# stage9_modern_gpt —— 9.0 定标尺:统一评估平台 + stage7 基线复测

看齐 minimind 的第一阶段(现代架构,见根目录 `ROADMAP.md`)。9.0 的任务:
把跨里程碑可比的三件标尺固定下来,并复测 stage7 基线,得到一份"以后所有
零件 A/B 都要对着比"的数字表。

## 与前代对照(长期维护,9.4 更新)

同语料(150MB wiki)、同 BPE、同数据流(seed 42);**Δ 均对 stage7 旧架构**。

**表 A:10000 步全量家族(val = 16-batch 稳定口径)**

| 臂 | 配置(stage7 之上改了什么) | 参数 | val | ppl | Δ |
|---|---|---|---|---|---|
| stage7 | —(LN/GELU/正弦/MHA, d_ff 3072) | 96.6M | 3.898 | 49.3 | — |
| **94_full**(现代基座) | RMSNorm+SwiGLU+RoPE+GQA6kv,d_ff 3072 | 117.8M | **3.776** | 43.2 | **−0.12** |
| 94_2048(同参对照) | 同上,d_ff 2048 | 89.5M(−7%) | 3.860 | 47.4 | −0.04 |

**表 B:零件履历(3000 步 A/B,每个零件单独换、同 seed;val 同口径)**

| 零件(里程碑) | 判决 | 证据 | 说明 |
|---|---|---|---|
| RMSNorm(9.1) | 附条件转正 | 4.78 vs 4.76(Δ+0.02,噪声内,同参) | 无收益也无损失;省 bias、bf16 稳 |
| SwiGLU(9.1) | **转正** | 4.61 vs 4.76(Δ−0.15,+29% 参数) | 9.4 收口:加宽到 4×d 再买 −0.08 |
| RoPE(9.2) | **转正** | 4.42 vs 4.66(Δ−0.24,**严格同参**) | 附外推能力(见 9.2 节) |
| GQA 6kv(9.3) | **转正** | ≈4.41 vs ≈4.45(−5.7% 参数,略优) | KV 显存减半 |
| KV cache(9.3) | 采用(工程件) | 生成 1.8~2.1×,文本逐 token 一致 | 正确性自检通过 |

结论:现代零件本身在**少 7% 参数**时就值 −0.04(表 A 第三行),FFN 恢复到
4×d 宽度(与 stage7 同宽)再买 −0.08——9.1"参数 caveat"收口:SwiGLU 值得
宽。stage8 的检索之坎在新架构上画像不变(真实答对率 ≈0/20,见 9.4 节)。

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
| FFN:`intermediate_size = ⌈hidden·π/64⌉×64`(d768→**2432**≈3.17×d),gate/up/down 无 bias | A/B 沿用旧 d_ff=3072(=4×d),为了"只换激活结构" | minimind 没义务跟旧 FFN 同宽;π×d 取整到 64 倍是它的宽度配方(9.4 另跑 d_ff=2048 同参对照臂隔离 SwiGLU 多出的参数) |
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

## 9.2 位置编码换代(RoPE + 外推实验)

**目的**:把正弦位置编码换成 RoPE,并回答两个问题:
① 同预算下(训练长度内)RoPE 值不值?
② 短训长测——训练窗口只有 256,喂 512/1024/2048 会怎样?(正弦 vs RoPE vs
推理期换表的三档曲线)

### 实现与 A/B 干净性

- `rope.py`:`precompute_freqs_cis` / `apply_rotary_pos_emb`(含 YaRN 分支),
  与 minimind `model_minimind.py:62-84` 逐行对应;旋转表预计算到 4096
  (非持久 buffer,不入 state_dict——换表试验不用动权重)
- `model_modern.py` 加 `cfg.pos = sinusoidal | rope`;`RotaryAttention` 的
  参数布局与 stage1 MHA **完全一致** → 这轮 A/B 严格同参(124.9M 对 124.9M,
  9.1 的 SwiGLU 臂没这个待遇)
- 默认路径同构自检 ✓(loss 六位全等、logits 差 0.0);rope 路径另加多长度
  前向冒烟(256/1024/2048)——同构自检只覆盖默认路径,新路径必须先冒烟
- 对照臂直接复用 91_combo(rms/silu/sine,同配置同 seed 3000 步)——零额外成本

### A/B 结果(3000 步,同 seed,124.9M)

| | 正弦(91_combo) | **RoPE**(92_rope, θ=1e6) |
|---|---|---|
| 训练内 best val(4-batch 噪声) | 4.550 | **4.333** |
| 16-batch 稳定 val | 4.659(ppl 105) | **4.419(ppl 83)** |
| 墙钟 | 2.2 step/s | 1.8 step/s(+27% 耗时,旋转开销) |

**域内就赢 0.24**——RoPE 不是"只有外推价值";幅度超过 9.1 换 Norm/FFN 的
任何单项。常识题仍 1/20 假阳(复读型),真实 0/20,无信号。

### 外推实验(训练窗口 256,窗口长度 ppl,同窗口零噪声方案对比)

| ppl | 256 | 384 | 512 | 768 | 1024 | 1536 | 2048 |
|---|---|---|---|---|---|---|---|
| 正弦 | 111 | 141 | 152 | **越界崩**(表长 512) | — | — | — |
| RoPE plain(θ=1e6) | 95 | 94 | 97 | 96 | 127 | 157 | 154 |
| RoPE 换底 θ=1e4 | 154 | 164 | 162 | 158 | 203 | 240 | 223 |
| **YaRN×4** | 100 | 97 | 88 | 74 | 83 | 94 | 100 |
| YaRN×8 | 103 | 100 | 92 | 79 | 89 | 93 | **83** |

读表注意:每点是 ~16k token 估计,看趋势;plain rope 在 768 的"平台"是
3× 训练长内的真实外推能力,1024+ 开始"没学过"。

### 结论

1. **RoPE 转正**(域内 +0.24,同参),cfg.pos="rope" 成为后续标配(θ=1e6)
2. **位置编码的外推性分三档**,实测证据:
   - 正弦 = 查表:**硬墙**(表长 512 越界即崩),表内也单调劣化
   - RoPE plain = 连续函数:**软衰减**(3× 内平台,之后缓劣化不崩)
   - YaRN = 推理期插值:把"没学过的位置"压回训练范围,**2048 处 154→100**,
     域内只付 ~5% 代价;factor 是对齐目标长度的旋钮(×8 更长端更好)
3. **θ=1e4 换底全崩**(域内 95→154):频率体系与训练错配。反向印证 minimind
   选 θ=1e6 的外推动机——它的 YaRN(factor16/orig2048/beta32/1)与我们
   yarn8 同机制,只是把 2048→32k 的比例放大

### 对照 minimind 源码的设计笔记

| 它的实现 | 我们的差异 | 为什么 |
|---|---|---|
| 布局 (B,T,H,D),cos 按 `unsqueeze(1)` | 布局 (B,H,T,D),cos 按 `[None,None]` | 头维位置不同,**照搬 unsqueeze 位置会错位**(本次真踩的坑) |
| rotate_half 半拆分配对 (i, i+D/2),cos/sin 两半重复 | 同款(逐行对应) | 与 LLaMA 的相邻配对 (2i,2i+1) 差维度置换,训练等价,换权重不可混用 |
| rope_theta=1e6,max_position 32768,YaRN 推理期(f16/orig2048/beta 32/1) | θ=1e6、表长 4096、推理期 yarn4/8 | 它的 factor16 对应 2048→32k 的 16 倍目标;我们是 256→2k 的 4~8 倍 |
| 非持久 buffer 存 cos/sin | 同款 | 表由 config 重算,不入权重 |

### 踩坑记录

1. **`register_buffer` 与同名普通属性冲突**:先在 __init__ 赋了
   `self.freqs_cos=None` 再 register_buffer → KeyError。先想好属性归属
2. **cos/sin 的 unsqueeze 维度随 q/k 布局变**:仓库 (B,H,T,D) → `[None,None]`;
   minimind (B,T,H,D) → `unsqueeze(1)`。跨库"抄公式"要连布局一起抄
3. **同构自检只测默认路径**:rope 路径的 bug(尺寸错位)靠训练第一跑才炸——
   补了"多长度前向冒烟"进流程(256/1024/2048)
4. **nohup 失败尝试会留下陈旧 done 标记**:第一次启动崩了,但链尾的
   `echo done > *.done` 仍执行 → 等待器误判"完成"。清理后再挂
5. **θ=1e4 换底实验的教训**:推理期换表只该在"外推友好方向"(同 θ 更大/
   插值)做;换小 θ 是让位置分布整体错配

### 本里程碑文件

| 文件 | 内容 |
|---|---|
| `rope.py` | RoPE + YaRN(逐行对照 minimind :62-84) |
| `model_modern.py` | +pos 开关、RotaryAttention(同参 A/B) |
| `eval_extrap.py` | 外推曲线 + 方案对比(CSV/PNG) |
| `results/92-rope.json` | rope 臂 16-batch 评估 |
| `results/extrap_92-*.csv/png` | 两条外推曲线(正弦封顶 512) |

## 9.3 KV cache + GQA(先 cache 后 GQA)

**目的**:① 生成从"每步全量重算"改成增量解码(旧 token 的 K/V 不变,见
CONCEPTS 十二节);② GQA——既然 KV 要存,把它存小一点(8q/4kv 式压缩)。

### 实现

- `GPT.forward_cached(ids, past_kv)`:prefill(整段前向,造缓存)→ 逐 token
  增量前向(每层把新 K/V 拼到历史尾,只算新 token 的注意力);位置偏移
  自动取"past 的长度"(同 minimind `start_pos = past.shape[1]` 的思路)
- `RotaryAttention` 加 `n_kv_heads`(w_k/w_v 投影到 kv 头数,算注意力前
  `repeat_kv` 复制回 Q 头数——同 minimind `model_minimind.py:86`,布局差
  一个转置)
- cache 只落在 rope 路径;sinusoidal 默认路径未动 → 同构自检继续有效 ✓
- `generate_cached.py`:与旧版 `sampling.generate` **共用同一个 sample_next**,
  唯一差别是算力省在哪 → "两路生成文本一致"成为 cache 正确性的最强自检
- `bench_generate.py`:一致性自检 + 墙钟加速 + KV 显存账

### 一致性自检(先抓出过一个真 bug)

修前:第 2 个 token 起 logits 分歧(差 ~1.2-1.6)。根因:`_block_forward` 里
`x + attn_out + ff(norm2(x))`——Python 先求值 `norm2(x)`(加注意力**之前**的
x),pre-norm 结构要求 norm2 吃**加完注意力之后**的 x,导致 prefill 与 decode
两路语义不一致。修后:**同 seed 两路生成逐 token 一致 ✓**(92_rope 与
93_gqa 都过)。

### KV cache 加速(92_rope,3 次均值,同参数同 seed)

| 续写长度 | 无缓存 | 有缓存 | 加速 |
|---|---|---|---|
| 200 token | 2.0s(101 tok/s) | 1.1s(185 tok/s) | **1.8×** |
| 400 token | 5.5s(73 tok/s) | 2.6s(155 tok/s) | **2.1×** |

序列越长省得越多(重算量随序列平方增长,缓存后每步只算常数)——这正是
真实推理引擎普遍用 cache 的原因。**GQA 让同一账本的显存再减半**:
L=512 单序列 KV:MHA 18.0 MB → **GQA(6 kv 头)9.0 MB**。

### GQA A/B(12q/6kv = 2:1,同 minimind 比例;3000 步同 seed)

| | 92_rope(MHA, 12 kv) | 93_gqa(GQA, 6 kv) |
|---|---|---|
| 参数 | 124.9M | **117.8M(−7.1M, −5.7%)** |
| 训练内 best val(噪声) | 4.333 | 4.296 |
| 16-batch 稳定口径(多次) | 4.419 / 4.477(均 ≈4.45) | 4.384 / 4.447 / 4.397(均 ≈4.41) |
| KV 显存 @L=512 | 18.0 MB | **9.0 MB** |

**结论:GQA 以 −5.7% 参数取得 ≈ 甚至略优的 val(Δ≈−0.04,两次信号方向
一致但仍在噪声边缘),KV 显存直接减半——参数效率与推理显存双赢,转正**
(n_kv_heads=6 进 9.4 全量)。常识题仍 1/20 假阳/0 真阳,无信号。

### 对照 minimind 源码的设计笔记

| 它的实现 | 我们的差异 | 为什么 |
|---|---|---|
| `repeat_kv`(B,T,H,D) 布局 | 同函数,布局 (B,H,T,D) | 头维位置不同,展开维序跟着变 |
| generate:`past_len = past_key_values[0][0].shape[1]` | 等价:`off = past_kv[0][0].shape[2]` | 都靠"已缓存长度"当位置偏移 |
| n_heads=8 / n_kv_heads=4(2:1) | 12 / 6(2:1) | 同样的压缩比;6 能整除 12 |
| 逐层 past concat + 单 token 解码循环 | 同款 | — |

### 踩坑记录

1. **残差顺序 = Python 求值顺序**:`x + attn_out + ff(norm2(x))` 的 norm2
   吃的是旧 x——"加完再归一"写错成"归一再加",prefill/decode 两路语义
   分裂。**一致性自检的价值:它比"loss 差不多"强得多,直接抓语义 bug**
2. 一致性测试的隐藏前提:prompt 必须短于旧路径的截断上限(max_len=512),
   否则两路输入不等价(旧路径会丢开头) —— bench 里有 assert 保护
3. GQA 头数必须整除:n_heads % n_kv_heads == 0,否则 repeat_kv 半头都出不来
4. nohup + ssh:后台链输出重定向到文件后 ssh 仍可能挂满超时再返回(远端
   进程已脱离,不受影响);陈旧 done 标记的坑同 9.2(先 rm 再挂等待器)

### 本里程碑文件

| 文件 | 内容 |
|---|---|
| `model_modern.py` | RotaryAttention(n_kv_heads/past)+ forward_cached |
| `generate_cached.py` | KV cache 生成(与旧版共用采样器) |
| `bench_generate.py` | 一致性自检 + 加速 + KV 显存账 |
| `results/93-gqa*.json`、`92-rope-rep93.json` | GQA 臂评估与重复测量 |

## 9.4 合体重训 + 逐行 diff(产出 stage10 的现代基座)

**目的**:把 9.1~9.3 转正的零件合并成完整现代配置,10000 步全量训练,
在同语料同分词同预算下回答"架构换代值多少";并逐行对照 minimind
源码,把它的设计决策整理成笔记(部分散见于 9.1~9.3,这里补全并汇总)。

### 实验(两只臂 + stage7 作对照组,全部 10000 步、seed 42)

| | stage7(旧) | 94_full | 94_2048 |
|---|---|---|---|
| 配置 | LN/GELU/正弦/MHA | rms/silu/rope/GQA6kv | 同左 |
| d_ff | 3072 | 3072(同宽直比) | **2048(同参对照)** |
| 参数 | 96.6M | 117.8M(+22%) | **89.5M(−7%)** |
| 训练内 best val | 3.826 | 3.650 | 3.830 |
| 16-batch 稳定 val | 3.898 | **3.776**(ppl 43.2) | **3.860**(ppl 47.4) |
| 墙钟 | 64 min | 90.5 min(1.8 step/s) | 83 min(2.0 step/s) |

**拆解**:modern −7% 参数(−0.04)+ FFN 宽度 2048→3072(−0.08,即 9.1 的
SwiGLU 参数预算花在这里值)。零件代差与参数代差分离清楚。汇总见顶部表 A/B。

### 行为复测

- **20 常识题:真实答对率 ≈0/20,与 stage7 同画像**。ask 2/20、fill 1/20 全是
  假阳:①"中华人民共和国的首都是哪里?"→ 答出"习近平**在北京**召开记者会"
  (共现假阳——北京以高频共现出现,不是作为首都的回答);②秦始皇题复读型
  (顺带冒出"建立秦朝",仍非定点作答)。**stage8 的"记忆会、检索不会"在新
  架构、同数据量下依旧成立 → 检索之坎确实与架构无关,是规模/数据的账**
- **生成质量(定性,肉眼对比 stage7 10000 步日志)**:句子更密、实体更多、
  百科语气更足("鲁迅是著名小说家,于1956年加入…"——结构完美、内容依旧
  张冠李戴)。与 val loss 方向一致,但不是本阶段的形式化指标
- KV cache 直接可用(`generate_cached.py`,94_full 配置)→ stage10 的聊天
  脚本有现成高速解码

### 对照 minimind `model_minimind.py` 的设计决策笔记(全文件逐行,汇总)

| 设计 | minimind | 我们 | 笔记 |
|---|---|---|---|
| weight tying | `tie_word_embeddings=True` | 同款(stage3 起) | 词表大时省 embedding+head 双份,且共享语义空间 |
| head_dim | 独立字段 = hidden/n_heads | 等价(未显式) | 解耦后能脱离 n_heads 调 d_k(我们没用到) |
| dropout | 0.0 | 0.1(stage1 遗产) | **未 A/B 的遗留差异**;小模型防过拟合或有帮助,留待观察 |
| 注意力实现 | SDPA flash + 手写回退 | 手写 masked softmax | 数值可对照、教学清晰;速度慢(训练 1.8 vs 2.0~2.6 step/s 部分因此) |
| FFN 宽度 | ⌈h·π/64⌉×64 = 2432(d768) | 3072 / 2048(两只臂) | π 配方 ≈3.17×d;我们测了 4×d 与 2.67×d 两档,2048 档同参仍优于旧架构 |
| MoE 分支 | 可选(4E/top1 + aux loss) | 不实现 | 独立课题,进 11 可选池 |
| config 存储 | HF PretrainedConfig(json) | vars(cfg) 存 ckpt | 等价的"权重自带配置";9.5 桥接才需要 HF 格式 |
| rope_ctx / θ | 32768 / 1e6 + 推理期 YaRN | 4096 / 1e6 + YaRN | 9.2 已测:我们的外推曲线与机制一致 |
| generate | 手写循环 + eos/streamer | generate_cached(可选 eos) | 9.3 一致性自检保证语义等价 |
| 词表 | 6400(自训 BPE) | 15124(自研 BPE) | 9.0 拍板不重训:跨分词器不可比 |

### 踩坑记录

- 9.1 笔记里 minimind FFN 宽度算错过一次(写成 3904,应为 ⌈768π/64⌉×64=
  **2432**)——已修正;公式带 π 时先心算验证再落文档
- 训练内 best 的跨臂比较再次被"噪声最小值选择"干扰(94_full 3.650 看着比
  94_2048 的 3.830 好 0.18,稳定口径下差距是 0.08)——**跨臂比永远用 16-batch
  稳定口径,训练内数字只看走势**

### 本里程碑文件

| 文件 | 内容 |
|---|---|
| `results/94-full.json(-rep)` | 现代基座评估(16-batch + 20 题 + 样例) |
| `results/94-2048-rep*.json` | 同参对照臂稳定口径 |
| ckpt(不入库) | `ckpt_94_full.pt`(117.8M,val 3.776,主基座)、`ckpt_94_2048.pt`(89.5M,轻量替代) |

## 下一步

9.5:桥接实验——下载 minimind 官方权重(110M 级 dense + 其 tokenizer),用
同一套 harness 的 20 题与固定 prompt 打它,分离"架构/规模/数据量"三变量,
检验 stage8 归因。下载指引见 9.0 节。之后进入 10(偏好学习)。见 ROADMAP。
