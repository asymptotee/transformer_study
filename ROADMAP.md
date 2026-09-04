# Roadmap: Stage 9+ 看齐 minimind(学习路线图)

> 状态:2026-09-03 立项。里程碑按序推进,每个里程碑一次 commit。
> 参照仓库:`/home/zhangxu/claude-code-dir/minimind`(下称 **$MM**,上游开源项目,
> 本地无改动)。建议开一个终端窗口随时对照,或把 `$MM` 指过去。

## 0. 为什么是 minimind

两个工程是同一个目标的两条路:本仓库"每个零件从零手写、用对照实验搞懂原理",
minimind 是"现代 LLM 全链路(预训练 → SFT → DPO/PPO/GRPO → Agent)的工程化实现"。
前八阶段已走完"预训练 → SFT"前半程,与 minimind 的差距集中在三处:

1. **架构代差**:正弦 PE + LayerNorm + 普通 MHA + FFN(vs 它的
   RoPE + RMSNorm + SwiGLU + GQA,另有 MoE / YaRN 可选件)
2. **后训练只走了一半**:只有 SFT(格式已毕业),没有偏好学习/RL 机制
3. **工程外壳**:DDP、断点续训、swanlab、HF 生态(这些是工具,**不是学习目标**,按需再说)

看齐 = 把 1、2 补上;**看齐方式 ≠ 抄代码**,见方法论。

## 1. 方法论(沿用本项目八个阶段的规矩)

每个零件走四步,一步都不能省:

1. **从原理出发自己实现**——动机能讲清,再动手
2. **一次只换一个变量**,固定标尺做 A/B(否则结果无法归因)
3. **再逐行对照 minimind 源码**,把它的设计决策(rope_theta、intermediate 取整、
   tie embeddings……)记进 README——它替你踩过的坑属于"读答案"
4. 达到**毕业标准**才收尾:README 记录数字 + 结论 + 踩坑

预期管理:在 100M/44M token 预算下,单个零件 A/B **很可能 loss 差异很小甚至持平**
(stage4 的教训:很多技术要在规模里兑现)。差异小也是结论——重点是机制与行为
(外推能力、解码速度、参数效率),以及"为什么 minimind 仍选它"的设计笔记。

## 2. 对照标尺(9.0 定死,之后不换)

| 变量 | 固定值 | 理由 |
|---|---|---|
| 语料 | stage7 的 150MB 中文维基(43.9M token) | val loss 可直比 stage7 的 **3.826** |
| 分词 | **自研 BPE(15124 词表),不重训** | 词表一换 perplexity 不可比(跨分词器要按 BPB 比);重训 6400 词表是独立课题 |
| 算力 | GB10;全量 100M=64min/10000 步;A/B 用 3000 步(≈20min) | 每个实验一个下午量级 |
| 评估 | `eval_harness.py` 统一出口(见 9.0) | 所有里程碑数字严格可比 |

## 3. 总览

| 阶段 | 里程碑 | 学习内容 | 对照 $MM 源码 | 类型 |
|---|---|---|---|---|
| 9 现代架构 | 9.0 定标尺 | 基线复测 + 评估平台 | — | 必做(半天) |
| | 9.1 Norm+FFN 换代 | RMSNorm、SwiGLU | `model/model_minimind.py:50`、`:136` | 必做 |
| | 9.2 RoPE | RoPE + 外推、YaRN | `model/model_minimind.py:62-84`、`:32` | 必做 |
| | 9.3 GQA + KV cache | 分组注意力、增量解码 | `model/model_minimind.py:86`、`:120`、`:234-287` | 必做 |
| | 9.4 合体重训 + diff | 整体换代的真实收益 | `model/model_minimind.py` 全文件 | 必做 |
| | 9.5 桥接实验 | 官方权重跑同一套评估 | 官方权重 + 其 tokenizer | 推荐 |
| 10 偏好对齐 | 10.1 DPO | 偏好学习(最大新概念) | `trainer/train_dpo.py`、`dataset/lm_dataset.py` DPO 段 | 必做 |
| | 10.2 GRPO | 无 RM 的 RL 最小闭环 | `trainer/train_grpo.py` | 推荐 |
| | 10.3 SFT 格式看齐 | chat template / system / think | `dataset/lm_dataset.py` SFT 段 | 可选快进 |
| 11 可选池 | MoE / 蒸馏 / HF 生态 / PPO | 各自新机制 | 各自 trainer | 暂不排期 |

## 4. 阶段 9:现代架构看齐(目录 `stage9_modern_gpt/`)

### 9.0 定标尺(半天)

- **仓库产物**:`stage9_modern_gpt/eval_harness.py`——一键出全套数字
  (val loss + 20 常识题 + 固定 prompt 生成样例);基线复跑记录;官方权重下载说明
- **实验**:复跑 stage7 全量或取用其 log,确认基线(val loss ≈ 3.826、常识 0/20)
- **准备**:从 $MM README 下载区(HF `jingyaogong/minimind-3` / ModelScope,见其
  README ~L237-293)下载官方 dense 权重与 tokenizer,放 `stage9_modern_gpt/cache/`
  (不入库),9.5 用
- **毕业标准**:eval_harness 一次运行出齐全部数字,且与 stage7 记录一致

### 9.1 Norm+FFN 换代(半天~1 天)

- **实现**:RMSNorm(为什么去掉均值偏移)、SwiGLU(gate/up/down,为什么门控激活);
  与旧实现**同文件 flag 切换**(延续 stage2 `train_compare.py` 的对照风格)
- **实验**:两个 A/B 各 3000 步;对照组 = 9.0 基线的同预算数字
- **数字**:LayerNorm vs RMSNorm、GELU vs SwiGLU 的 val loss 表 + 生成样例
- **毕业结论**:实测量化小预算下差异(小是预期),解释 minimind 选它们的理由
  (数值稳定性、大模型/长序列收益);两零件转正为默认实现

### 9.2 RoPE(1~2 天,本阶段最重要的单个概念)

- **实现**:`rope.py`——`precompute_freqs_cis` + `apply_rotary_pos_emb`,
  位置编码第一次不再依赖 `max_len` 上限
- **实验**:① A/B:正弦 vs RoPE 同长度;② **外推**:256 训练 → 512/1024 长测
  (预期正弦崩溃、RoPE 软衰减);③ 推理期 YaRN 4 倍外推(等价 $MM
  `inference_rope_scaling`)vs 训更长的对照
- **数字**:三组困惑度/val loss 表 + 外推曲线图
- **对照笔记**:回答设计题——为什么 `rope_theta=1e6`?为什么 2 维一组拆 cos/sin?
  YaRN 的 ramp 在调什么?
- **联动**:CONCEPTS.md 新增"位置编码外推"节
- **毕业标准**:自己的实现与 $MM `precompute_freqs_cis` 逐行 diff 一致;外推结论成图

### 9.3 GQA + KV cache(1~2 天,先 KV cache 后 GQA)

- **实现**:① generate 增量解码(首轮算全序列,采样循环每步只算最后一个 token,
  逐层存/取 KV);② `repeat_kv` + GQA(8q/4kv)
- **实验**:KV cache 开/关的生成加速比 + **一致性自检**(两路生成文本必须一致,
  只是更快);MHA vs GQA 的参数量与 val loss
- **数字**:同长生成 wall-time 对照;cache 后 KV 显存(顺带解释 GQA 省显存的动机)
- **联动**:CONCEPTS.md 补"KV cache 实现"节(概念早已埋线,这里落地)
- **毕业标准**:带 cache 生成与旧实现文本一致且加速可测;GQA 参数账算清

### 9.4 合体重训 + 逐行 diff(1~2 天)

- **实现**:现代 GPT——d768/12 层,全部新零件(9.1~9.3)合并为默认配置
- **实验**:全量 10000 步(64min)→ 与 stage7 同语料同 tokenizer 直比
- **数字**:**3.826 → X.XXX**;生成样例对照;20 题复测(预期仍 ≈0——
  从这一刻起"答不出"再也赖不到架构头上)
- **对照笔记**:逐行 diff $MM `model_minimind.py`,记录设计决策:intermediate 取整、
  tie embeddings、head_dim 独立设定、dropout=0、SDPA(flash)自动回退、MoE 分支
  (本期不实现,只读笔记)
- **毕业标准**:README 顶部放"与前代对照"汇总表;**产出 stage10 的原料——现代基座**

### 9.5 桥接实验(推荐,半天)

- **实现**:`eval_minimind.py`——唯一允许 import transformers 的脚本;
  加载官方权重 + 它的 tokenizer,跑同一套 eval_harness 的问题
- **实验**:同一套 20 题 + 固定 prompt 打官方 110M 级 dense 权重
- **数字**:官方得分 vs 自己 0/20;它的典型回答样例
- **毕业结论**:三变量(架构/规模/数据量)分离后,**实锤或修正 stage8 的三层归因**
  ——"同样 100M 级,为什么它答得出而我答不出";结论升级写进 stage8/9 README

**阶段 9 整体成果**:现代基座 + 可复用的零件 A/B 平台(固定语料/评估标尺/flag 化
对照)+ 一份架构换代实验报告 + CONCEPTS 新增两节。

## 5. 阶段 10:后训练看齐 —— 偏好学习(目录 `stage10_preference/`)

### 10.1 DPO(必做,2~3 天;公式推导是大头)

- **推导**:Bradley-Terry → 隐式奖励 → DPO 闭式解 → 自己写 loss
  (参考模型冻结、logp 计算、KL 隐式在 loss 里)
- **数据**:`make_pref.py` 自造偏好对——同一问题上,正例 = 采样出的规则好回答
  (含关键实体/格式完整),负例 = 截断或跑题回答。**数据质量不是重点,机制是重点**
- **实验**:SFT → DPO 后:规则裁判的偏好命中率、与 SFT 的 KL 漂移曲线、
  温度对 DPO loss 的影响;20 题顺带复测(低分不意外,机制先行)
- **最强看齐标准**:**相同 batch 上,自己实现的 DPO loss 与 $MM `train_dpo.py`
  算出的数值一致(浮点误差内)**——机制懂了,不是"照着跑通"
- **联动**:CONCEPTS.md 新增"偏好学习"节

### 10.2 GRPO(推荐,1~2 天)

- **实现**:规则奖励集(关键词/长度/格式)+ group 内相对 advantage + 对参考模型的
  KL 约束;对照 $MM `train_grpo.py`
- **实验**:reward 上升曲线与 KL 受控情况;与 DPO 的对照表
- **毕业结论**:讲得清"无 RM 的在线策略 RL"最小组件集,并说得出它 vs DPO 的取舍
  (在线/离线采样、数据需求、稳定性)

### 10.3 SFT 格式看齐(可选快进)

- **内容**:chat template、system prompt 概率注入(20%)、assistant 段掩码写法、
  think 标签处理,对照 $MM `SFTDataset`
- **注意**:自己的 SFT 机制(stage6/8)已毕业;此项只在自己知识数据上重做一轮,
  验证模板换格式不破坏 test loss(对照 stage8 的 3.87),**不必单独成里程碑,
  做 DPO 数据时顺手完成**

**阶段 10 整体成果**:现代基座上跑通"偏好对齐"完整闭环(数据 → loss → 指标),
DPO loss 数值与 minimind 对齐;基座 → SFT → 偏好三连模型链打通。

## 6. 可选池(阶段 11,暂不排期,README 先记一笔)

| 选项 | 一句话成果 | 对照源码 |
|---|---|---|
| MoE | dense vs 4E/top1 同激活参数量 A/B + 专家负载均衡指标,理解 router aux loss 在调什么 | `model_minimind.py:148`(MOEFeedForward) |
| 蒸馏 | 官方权重 teacher → 30M student,CE+KL;student vs 同预算直接训的差距表 | `trainer/train_distillation.py` |
| HF 生态化 | 把现代基座包成 PreTrainedModel,save_pretrained + chat template + OpenAI 兼容服务(纯工程) | $MM `scripts/serve_openai_api.py` |
| PPO + Agent | 需要 RM + rollout + 工具环境,是 GRPO 的超集;有明确动机(如 tool calling)再开 | `trainer/train_ppo.py`、`train_agent.py` |

## 7. 拍板点(已按推荐默认)

1. **HF 依赖边界**:训练代码保持零依赖(自写哲学不污染);仅 9.5 桥接评估脚本
   `eval_minimind.py` 允许 import transformers
2. **目录与提交**:里程碑 = 一次 commit(消息形如 `stage9.2: RoPE 与外推实验`);
   阶段目录 `stage9_modern_gpt/`、`stage10_preference/`;权重/日志不入库(沿用 .gitignore)
3. **分词**:始终用自研 BPE;minimind 官方权重只在 9.5 由其自带 tokenizer 解码
4. **10.3 不单独成里程碑**,并入 10.1 的前置准备

## 8. 联动维护

- 每个里程碑收尾:更新该阶段 README(目标 / 实验 / 数字 / 结论 / 踩坑,沿用现有风格)
- CONCEPTS.md 规划新增:位置编码外推(9.2)、KV cache 实现(9.3)、偏好学习(10.1)
- stage9 README 顶部长期维护一张"与前代(stage7 基座)对照"汇总表,跨里程碑数字都进这一张表
