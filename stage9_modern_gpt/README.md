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

## 9.1 Norm + FFN 换代(RMSNorm / SwiGLU)

**目的**:把 stage7 基座的两个零件换成 minimind 的现代件,并用同预算 A/B
量化每个零件单独换的值。零件落点 `model_modern.py`(9.1 起步,9.2 起继续演进)。

### A/B 的干净性(本次新立的规矩,后面沿用)

1. **逐位同构自检**:model_modern 默认开关(layernorm/gelu)下,同一份 stage7
   权重、同一批窗口,**loss 六位小数全等、logits 差 = 0.0**(零噪声对比)。
   固化成了 `check_isomorphic.py`——9.2/9.3 每次改完 model_modern 先跑它,
   同构不过不做 A/B。
2. **同 seed 同数据流**:4 臂同 seed 42,训练 batch 与 val batch 完全一致,
   观察到的差异只能来自被换的零件。

### 实验(4 臂 × 3000 步 ≈ 22 分钟/臂,同 seed、同 96.6M 预算、d768/12 层)

| 臂 | 配置 | 参数 | 训练内 best val(4-batch 噪声) | **16-batch 稳定口径** |
|---|---|---|---|---|
| control | LN + GELU | 96.6M | 4.714 | **4.753 / 4.763**(≈4.76) |
| rms | **RMSNorm** + GELU | 96.6M | 4.706 | 4.783 / 4.769(≈4.78) |
| swiglu | LN + **SwiGLU** | 124.9M(+29%) | 4.546 | **4.609** |
| combo | **RMSNorm + SwiGLU** | 124.9M(+29%) | 4.550 | **4.659** |

常识题(ask/fill):各臂正例全部为关键词假阳(复读"秦始皇帝"、control 的
"长江→第一次世界大战"命中"第一"),**真实 ≈ 0/20**——3000 步模型知识层
本就没有,该零件不影响检索(和预期一致,不构成评估信号)。

### 结论

1. **SwiGLU:显著更好**(同预算 Δ≈−0.15,16-batch 噪声 ±0.02,信号真实)。
   **转正为默认 ff**。代价:同 d_ff 下参数 +29%(gate/up 双投影);
   不是严格同参对比——标准做法是把 SwiGLU 宽度缩到 2/3 对齐激活参数,
   9.4 全量时可用 `--d-ff 2048`(≈2.7×768)顺带对照(见 minimind 笔记)。
2. **RMSNorm:持平**(Δ+0.02,两次测量 [4.783/4.769] vs [4.753/4.763]
   有重叠,在噪声边缘;训练内 best 甚至略优 4.706 vs 4.714)。诚实结论:
   100M/3000 步预算下 **RMSNorm 没有可测收益,也没有可测损失**。
   附条件转正:minimind/LLaMA 选它不是因为小预算 loss,而是省偏置参数、
   bf16 数值更稳、深宽模型的已知偏好;9.4 的 10000 步全量对 combo 做终裁
   (对照组直接复用 stage7 的 3.898——LN+GELU 10000 步,不用重跑)。
3. **combo ≈ swiglu 单换**(4.659 vs 4.609,差在噪声内):两零件收益不叠加,
   Norm 那件在 3000 步里没贡献——进一步支持"RMSNorm 的账要到大预算再算"。

### 对照 minimind 源码的设计笔记(`model/model_minimind.py`)

| 它的实现 | 我们的差异 | 为什么 |
|---|---|---|
| RMSNorm:`x*rsqrt(mean(x²)+eps)`,float32 内算再 type_as(1e-5 类内默认;config `rms_norm_eps=1e-6`) | 同款写法;A/B 用 eps 1e-5(与 LN 的 eps 对齐,变量唯一) | eps 1e-5 vs 1e-6 在此量级无感(值域都远离 0),不必为此多跑一臂 |
| FFN:`intermediate_size = ⌈hidden·π/64⌉×64`(d768→**3904**≈2.7×d),gate/up/down 无 bias | A/B 沿用旧 d_ff=3072(=4×d),为了"只换激活结构" | minimind 没义务跟旧 FFN 同宽;它按 π 取整是"激活参数≈2.7 倍输入宽度"的近似 |
| 激活走 `ACT2FN[silu]` | 直接 `F.silu` | 无本质差异 |

### 踩坑记录

- 训练内 best val 是"噪声曲线的最小值",swiglu/combo 的 best(4.546/4.550)
  与 16-batch 复测(4.609/4.659)差 0.06~0.11;跨臂比一定要用同一口径
- 评估脚本里 grep 含 "✓" 的行会被 locale 干扰(✓ 占两个 token),提取数字
  用 `grep -oE "[0-9.]+$"` 一类写法
- SwiGLU ckpt 体积 501MB > control 388MB,正是 +28.3M 参数(fp32 4B)——
  文件大小差异本身是参数账的旁证

### 本里程碑文件

| 文件 | 内容 |
|---|---|
| `model_modern.py` | 现代模型(9.1:norm/ff 开关;默认 = 旧架构同构) |
| `train_large9.py` | 零件 A/B 训练(同 seed 同数据流约定) |
| `check_isomorphic.py` | 逐位同构自检(每次换零件后必跑) |
| `results/91-*.json` | 4 臂评估 + 重复测量 |

## 下一步

9.2:RoPE(换掉正弦位置编码 + 外推实验 + YaRN 推理期对照),先跑
`check_isomorphic.py` 再动手。见 ROADMAP。
