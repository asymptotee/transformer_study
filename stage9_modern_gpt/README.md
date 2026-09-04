# stage9_modern_gpt —— 9.0 定标尺:统一评估平台 + stage7 基线复测

看齐 minimind 的第一阶段(现代架构,见根目录 `ROADMAP.md`)。9.0 的任务:
把跨里程碑可比的三件标尺固定下来,并复测 stage7 基线,得到一份"以后所有
零件 A/B 都要对着比"的数字表。

## 双机拓扑(开发机 ↔ DGX Spark)

```
本机(代码源头,git 仓库)          DGX Spark(运行端,非 git 副本)
~/claude-code-dir/transformer_study   ~/llm_study/transformer_study
   ↑ 本机没有 torch,所有评估在 Spark 跑
   └─ rsync stage9_modern_gpt/ → Spark(如下)
```

- **代码同步**(本机执行):
  `rsync -az --exclude __pycache__ stage9_modern_gpt/ zhangxu@192.168.18.139:~/llm_study/transformer_study/stage9_modern_gpt/`
- **Spark 运行**:`~/llm_study/.venv/bin/python`(语料/ckpt/cache 都在 Spark:
  `stage7_gpu_scale/ckpt_large.pt` 371MB、`cache/tokens.pt` 353MB)
- 共享模型代码(stage3/4/5)两边 md5 一致,是同一份;stage7/8 脚本两边仅
  docstring 的 venv 路径不同,逻辑一致
- 结果回传:`rsync -az zhangxu@192.168.18.139:~/llm_study/transformer_study/stage9_modern_gpt/results/ results/`

## eval_harness.py —— 统一评估出口

一个命令出齐三组数字,每组都注明"与哪个旧脚本逐字同源"——**题面、模板、
判定、解码参数任何一个变了,跨里程碑的数字就不可比**(stage8 的假阳性教训):

| 段 | 内容 | 口径同源 |
|---|---|---|
| [1] val loss / ppl | 缓存 token 末 5% 当验证集,随机 N 个 batch(32×256),CE(ignore_index=0),bf16 | stage7 `train_large.py` 的 `val_loss()` |
| [2] 20 道常识题 | ask(问句)+ fill(补全)两种格式,贪心作答,回答含关键词判对 | stage8 `eval_facts.py`(HIGH/LOW/FILL 逐字拷贝) |
| [3] 固定 prompt | 6 个 prompt 续写,解码 t=0.7 / top-p=0.9 / 重复惩罚=1.2 | stage7 `eval_ckpt.py` |

用法(Spark):

```bash
PY=~/llm_study/.venv/bin/python
$PY -u eval_harness.py --ckpt ../stage7_gpu_scale/ckpt_large.pt \
    --label stage7-baseline --json-out results/stage7-baseline.json
# 常用开关:--no-val / --no-facts / --no-samples / --fact-format ask|fill
#           --val-batches 16(稳定口径,见下) / --device cpu
# 9.4 现代架构落地后加 --model-module model_modern,harness 本身不用改
```

输出:屏幕报告 + `--json-out` 机器可读结果(内含完整 cmd 与时间戳,可复现)。
**评估结果 `results/*.json` 入库**(每里程碑都留一份,攒出跨里程碑汇总表);
`*.log` 按仓库惯例忽略。

## 9.0 基线:stage7 `ckpt_large.pt`(best at step 8000)

| 指标 | stage7 记录 | harness 复测 | 说明 |
|---|---|---|---|
| val loss | 3.826(训练时 best) | **3.898**(16-batch 稳定口径);4-batch 单次 3.895~3.981 | 见下方噪声分析 |
| facts ask | 1/20\*(假阳) | 1/20(**假阳**) | 秦始皇题复读问题文本,"秦"关键词命中;真实 ≈ 0/20 |
| facts fill | 1/20\*(假阳) | 1/20(**假阳**,同一题) | 同上 |
| 生成样例 | stage7 README 表格 | 同水平:百科模板 + 乱参数(log 留存) | 例:北京是→"中国共产党在十八路军方的发动政变…" |

### val loss 噪声分析(对 9.1 之后所有 A/B 都成立)

- 4-batch(32,768 token)单次测量在 3.895~3.981 间波动,**极差 ≈ 0.09**;
  **小于 0.1 的差异在 4-batch 口径下不能算数**
- 16-batch 稳定口径:3.898(ppl 49.3),波动缩到 ~±0.02
- 训练记录 3.826 是"噪声曲线上的最小值":best-ckpt 是被低估噪声选出来的,
  同一份权重在稳定口径下 ≈ 3.90 属正常,不是 bug
- **结论:以后所有 A/B 默认 `--val-batches 16`;4-batch 只用于快速探路**

### 评估噪声之外的坑

1. **假阳性复现**:基座问"秦始皇…朝代"会原样复读问题(训练分布里"秦始
   皇统一六国"高频),关键词判定把它算对——stage8 已认过一次,harness 沿用
   同一判定以便纵向可比,但读表时要认得"复读型假阳"
2. **采样型指标**:prompt 续写带温度,两次运行文本不同属正常;比的是"水平",
   不是逐字复现

## 9.5 备料:minimind 官方权重下载指引

桥接实验(9.5)要用官方 110M 级 dense 权重。下载在 **Spark** 上做(网络在那边):

```bash
# 二选一(详见 minimind 仓库 README 下载区):
cd ~/llm_study/minimind            # 若不存在:git clone 对应仓库
modelscope download --model gongjy/minimind-3 --local_dir ./minimind-3
# 或 git clone https://huggingface.co/jingyaogong/minimind-3
```

权重与 tokenizer 备到 `stage9_modern_gpt/cache/minimind/`(cache/ 不入库),
9.5 的 `eval_minimind.py` 用。数据文件(`pretrain_t2t_mini.jsonl` 等)9.5 之后
若想跑 minimind 自己的训练再下,9.0~9.4 用不到。

## 本阶段文件

| 文件 | 内容 |
|---|---|
| `eval_harness.py` | 统一评估出口(见上) |
| `results/*.json` | 基线评估结果:stage7-baseline(-rep1/2/val16 为噪声量化跑) |
| `README.md` | 本文件(9.0 记录;9.1 起各里程碑在此追加) |

## 下一步

9.1:Norm + FFN 换代(RMSNorm / SwiGLU,各一次 A/B,3000 步 4-batch 快速探路 +
16-batch 定论),对照 minimind `model/model_minimind.py:50`、`:136`。见 ROADMAP。
