# stage12_hf_eval —— HF 包装 + lm_eval 标准评测

**目标**:把自研模型接入 HF 生态与 lm-evaluation-harness,拿到**与 minimind
表可直接对照的标准评测数字**(其 README「客观评测」用的是同一套任务)。

## 三件套

| 文件 | 作用 |
|---|---|
| `hf_wrap.py` | `MiniGPTConfig(PreTrainedConfig)` + `MiniGPTForCausalLM(PreTrainedModel)` + `OurBPETokenizer(PreTrainedTokenizer)`(包装 bpe.py);`save_pretrained` 导出 `hf_11_4_chat/`(权重 + config + tokenizer + `chat_template.jinja`) |
| `check_parity.py` | **包装一致性自检**(守门员):权重逐位、logits 逐位、分词逐 id、chat 模板契约 |
| `run_lm_eval.py` | 注册自定义类 + `lm_eval.simple_evaluate`;`--tasks/--limit/--apply-chat-template/--out` |

## 包装期发现并修复的两个真 bug(自检的价值)

1. **`rope.py` 半精度 dtype 提升**:`q * cos`(half × fp32 表)会把 q 提升成
   fp32,而 v 仍是 half → fp16 推理时 attention 崩。修:旋转算完 `.to(q.dtype)`
   回原 dtype(fp32 下是 no-op;minimind 同款处理)。
2. **`model_modern.py` RoPE 表被清零**:transformers≥5 的 meta-device 初始化
   会丢**非持久 buffer**——HF 加载后 `freqs_cos` 全零,RoPE 变成无操作
   (logits 差异 8.09,逐层定位到 buffer 差异 1.0)。修:`freqs_cos[0,0]==0`
   时重算(minimind 源码里注释过的同一个坑,现在我们也踩过并防御)。

**一致性自检终值**:权重 max-diff **0.000**、logits max-diff **0.000**(fp32)、
分词逐 id 一致、chat 模板与训练渲染逐字符一致、fp16 前向正常。

## 标准评测结果(11_4_chat,加 chat 模板,lm-eval 0.4.13)

| 任务 | 我们 | minimind-3 64M | 随机线 | 判读 |
|---|---|---|---|---|
| **C-Eval**(53 科全量) | **23.03**(逐科宏平均 22.39) | 24.89 | ~25 | 双方贴随机线 |
| **ARC-Easy** | **28.70** | 28.49 | ~25 | 持平 |
| **PIQA** | **54.84** | 50.65 | 50 | **我们 +4.2** |
| **OpenBookQA** | 13.00 | 23.60 | ~25 | 我们明显偏低 |
| CMMLU | 未跑(见下) | 25.38 | ~25 | — |
| HellaSwag | 未跑 | 28.28 | ~25 | — |
| Social-IQA | 未跑 | 34.19 | ~33 | — |

## 口径与已知偏置(读表须知)

1. **判分**:选择题 logprob 法(比较候选项条件对数概率,取最大),无自由
   生成 —— 与 minimind 同法;我们**未做**它的 "exam" 格式对齐(它靠轻量
   LoRA 对齐选择题格式后 C-Eval **+6.1**,即这套分数里掺着"格式熟悉度")
2. **tokenizer 偏置(我们特有)**:BPE 在中文语料上训练,英文被过度切分,
   而 logprob 判定对 token 化敏感 → 英文项存在系统性偏置,OpenBookQA 13%
   (低于随机)可能一部分源于此,不全是"不会"
3. **量程**:该量级模型在这套评测上区分度低(±3 分内≈噪声)——它回答"与
   社区同尺子上的位置",不回答"自家模型是否变好"(后者用 20 题/held-out)

## CMMLU 未跑的原因与解法(已探明,待需要时执行)

- **原因**:`lmlmcat/cmmlu`(原 `haonan-li/cmmlu`)是**脚本数据集**
  (`cmmlu.py` + `cmmlu_v1_0_1.zip`),而 Spark 环境 `datasets==5.0.1`
  已硬移除脚本支持;minimind 能跑是因为它 pin 了 **`datasets==3.6.0`**
  (仍允许 `trust_remote_code` 加载脚本,已验源码)
- **解法**:镜像下 zip → 逐科转 Parquet → 程序化改写 lm_eval 的 67 个科目
  yaml(只改 `dataset_path: parquet` + `data_files`,判分字段一字不动)+
  拷贝组定义 `_cmmlu.yaml` → `TaskManager(include_path=...)` 加载 → `--tasks cmmlu`

## 复现命令(Spark)

```bash
PY=~/llm_study/.venv/bin/python
cd ~/llm_study/transformer_study/stage12_hf_eval

# 1) 包装 + 自检
$PY hf_wrap.py --ckpt ../stage11_datascale/ckpt_11_4_chat.pt \
    --bpe ../stage11_datascale/cache_mm10g/bpe.json --out hf_11_4_chat
$PY check_parity.py --ckpt ../stage11_datascale/ckpt_11_4_chat.pt \
    --bpe ../stage11_datascale/cache_mm10g/bpe.json --hf-dir hf_11_4_chat

# 2) 评测(HF_ENDPOINT 必须在 import 前设——踩过)
HF_ENDPOINT=https://hf-mirror.com $PY run_lm_eval.py --hf-dir hf_11_4_chat \
    --tasks ceval-valid --apply-chat-template --out results_lmeval/11_4_chat_ceval.json
HF_ENDPOINT=https://hf-mirror.com $PY run_lm_eval.py --hf-dir hf_11_4_chat \
    --tasks arc_easy,piqa,openbookqa --apply-chat-template \
    --out results_lmeval/11_4_chat_en3.json
```

环境:lm-eval 0.4.13 + accelerate 1.15 + transformers 5.16.1 + datasets 5.0.1;
`batch_size=1`(我们的 forward 不支持 padding,包装期决策)。

## 文件

| 文件 | 内容 |
|---|---|
| `hf_wrap.py` / `check_parity.py` / `run_lm_eval.py` | 三件套(见上) |
| `results_lmeval/11_4_chat_ceval.json` | C-Eval 逐科明细 + 汇总 |
| `results_lmeval/11_4_chat_en3.json` | ARC-Easy / PIQA / OpenBookQA |
| `hf_11_4_chat/`(Spark,不入库) | 导出的 HF 格式模型(可 `from_pretrained`) |

## 配套改动(本阶段顺带改的共享代码)

- `stage9_modern_gpt/rope.py`:旋转结果 cast 回原 dtype(修 fp16)
- `stage9_modern_gpt/model_modern.py`:RoPE buffer 清零防御(修 v5 加载)

两者对既有训练/评测是 no-op(fp32 路径),只影响半精度与 HF 加载路径。
