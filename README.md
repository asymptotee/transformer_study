# Transformer 自学笔记

从零手写 Transformer，并通过对照实验搞懂它的每一个零件和背后的原理。
按学习进度分阶段组织，每个阶段一个子目录。

## 目录结构

```
transformer_study/
├── README.md          ← 你在这里（总路线图）
├── CONCEPTS.md        ← 概念主线：QKV、KV cache、记忆vs泛化、容量（贯穿各阶段）
│
├── stage1_basics/     ← 阶段一：手写 Transformer 基础
│   ├── model.py         全部模型模块（核心）
│   ├── data.py          字符分词器、双任务数据管道
│   ├── train.py         训练循环 + 评估
│   ├── generate.py      自回归生成
│   ├── eval_ckpts.py    checkpoint 评估
│   ├── demo_matrices.py 真实矩阵演示（注意力反对角线）
│   ├── corpus.txt       小语料（唐诗+谚语，约 300 行）
│   ├── ARCHITECTURE.md  架构 + model.py 逐行讲解 + 形状流转 + abc→cba 实例
│   ├── README.md        阶段一：怎么跑 + 踩坑记录
│   └── ckpt*.pt         训好的检查点
│
├── stage2_scaling/    ← 阶段二：数据规模如何把"记忆"变成"泛化"
    ├── gen_corpus.py    生成结构化大语料（几千行）
    ├── train_compare.py 小数据 vs 大数据对照实验
    └── README.md        阶段二：实验设计与结论
│
├── stage3_gpt/        ← 阶段三：decoder-only GPT（语言模型）
│   ├── model_gpt.py     GPT 模型（= 阶段一 decoder 半边，next-token 预测）
│   ├── make_corpus.py   伪诗语料生成器
│   ├── train_lm.py      next-token 训练（nanoGPT 式）
│   ├── generate_text.py prompt 续写（温度 / top-k）
│   └── README.md        阶段三：从 seq2seq 到语言模型的范式转变
│
├── stage4_scaling_bpe/ ← 阶段四：小模型为什么不连贯，怎么改善
│   ├── scaling_experiment.py  缩放实验（大/中/小三规模，loss-参数曲线）
│   ├── bpe.py                 BPE 分词器
│   ├── train_bpe.py           BPE vs 字符级公平对比
│   └── README.md              阶段四：规模与分词的意义
│
├── stage5_capstone/   ← 阶段五：Capstone，把所学合起来推到 CPU 极限
│   ├── sampling.py            改进解码（top-p + 重复惩罚）
│   ├── train_best.py          最佳模型（BPE+大模型+长上下文）
│   ├── generate_best.py       改进解码续写
│   └── README.md              阶段五：组合技术与天花板
│
└── stage6_sft/        ← 阶段六：SFT 指令微调，从"续写"到"应答"
    ├── make_qa.py             构造鲁迅问答数据
    ├── lora.py                LoRA 低秩适配
    ├── train_sft.py           SFT 训练（loss 掩码，全量/LoRA）
    ├── compare.py             基座/全量/LoRA 三方对比
    └── README.md              阶段六：SFT 与 LoRA
│
├── stage7_gpu_scale/  ← 阶段七：GPU 规模训练（DGX Spark）
│   ├── build_corpus.py        zhwiki dump → 提取 → 繁转简 → 语料
│   ├── train_large.py         bf16 + warmup/cosine + 编码缓存
│   ├── eval_ckpt.py / chat7.py 评估与交互续写
│   ├── probe_v1/              探路实验存档（30M，可复现）
│   └── README.md              阶段七：规模翻越连贯之坎
│
└── stage8_knowledge_sft/ ← 阶段八：知识问答 SFT
    ├── make_qa.py             维基语料 → 补全式问答对
    ├── train_sft7.py          SFT（全量/LoRA，多进程编码）
    ├── eval_facts.py          20 道常识题评估（ask/fill 双格式）
    ├── chat8.py               交互式问答
    └── README.md              阶段八：记忆会，检索不会
```

## 学习路线

### 阶段一：搞懂 Transformer 怎么运作

从零实现 encoder-decoder Transformer，用两个任务体检同一套架构：

- **reverse（反转）**：有确定规则 → 模型学算法 → 可泛化（验证集 exact match 0.9+）
- **complete（续写）**：无通用规则 → 模型背语料 → 不泛化

核心收获：注意力机制、形状流转、teacher forcing、因果掩码，以及
**"任务性质由数据有没有可学结构决定"**。详见 `stage1_basics/`。

### 阶段二：数据规模引发的质变

承接阶段一留下的问题：**如果 complete 的数据大到超过参数容量，背不动了，会怎样？**

答案是：模型被迫去**压缩**数据，而压缩自然语言就是学习它的统计结构——
于是它从"背下每首诗"变成"学会诗怎么写"，开始能续写没见过的句子。
这正是真实大语言模型的诞生机制。详见 `stage2_scaling/`。

### 阶段三：迈向真正的 LLM（decoder-only GPT）

阶段一二是 encoder-decoder 的 seq2seq 范式；现代 LLM（GPT/Claude）全是 **decoder-only**。
阶段三把阶段一的 decoder 半边拿出来，改成"纯下一个 token 预测"，就是一个小 GPT：

- 没有 encoder、没有交叉注意力，只有因果自注意力
- 输入 = 输出错一位，对每个位置预测下一个 token
- 生成从"输入→输出"变成"prompt 续写"

这是从"序列映射"到"建模序列概率分布"的范式转变——`CONCEPTS.md` 第六节说的
LLM 范式，在这里第一次变成可以跑、可以续写的代码。详见 `stage3_gpt/`。

### 阶段四：小模型为什么不连贯，怎么改善

阶段三的鲁迅模型"风格像、句子不通"。本阶段用两个实验回答为什么、怎么办：

- **缩放实验**：同一语料训大/中/小三个模型——模型越大 loss 越低、生成越连贯
  （scaling law 的缩影）
- **BPE 分词**：字符级把语义打散，BPE 学出词级 token——小规模 loss 未必更好，
  但生成更顺，它的价值在规模化

共同教训：**很多技术的意义要在规模中才兑现**；小模型不连贯不是错，是还没大到
能压缩到语义那一层。详见 `stage4_scaling_bpe/`。

### 阶段五：Capstone —— 把所学合起来

把前四阶段的技术全部组合（BPE + 更大模型 + 更长上下文 + 改进解码），在 CPU 极限
内把鲁迅续写推到最好。生成质量逐级上台阶：字级乱接 → 对话 → 人物文体（模型甚至
学会了鲁迅笔下的人物和注释文体）。但句子间仍无逻辑叙事——**风格能学到极致，
连贯却跨不过去**，那道坎要靠规模涌现。详见 `stage5_capstone/`。

### 阶段六：SFT 指令微调（后训练）

把阶段五的基座做 **SFT**，从"只会续写"调教成"会应答"，并对比**全量 vs LoRA**：

- **loss 掩码**：只对答案部分算 loss，教模型"如何回答"而非"复述问题"
- **LoRA**：冻住基座、只训 1% 的低秩参数，效果接近全量（省约 108 倍参数）
- **知识边界**：SFT 教会了应答格式，但答不出语料外的东西——"SFT 激发行为，
  预训练注入知识"

至此走完"预训练（三~五）→ 后训练/SFT（六）"的 LLM 全流程。详见 `stage6_sft/`。

### 阶段七：GPU 规模训练 —— 规模翻越"连贯之坎"

从 CPU 换到 DGX Spark（NVIDIA GB10），语料从 1.8MB 鲁迅换到 **150MB 中文维基
百科**（1.8 万篇），模型从 3.5M 推到 **96.6M**：

- **探路（30M/2000步）**：val loss 5.23，生成停在"词级关联"（数学↔向量/矩阵）
- **全量（100M/10000步，64分钟）**：val loss **3.826**（困惑度降 4 倍），
  生成**句子全通**——"天文学家 Mark Oser 在 2003 年发表了《千年梦》"
- **连贯之坎跨过了**（阶段五遗留问题在规模上验证），但知识仍以"模板+高频
  填充"的模糊形式存在，答不出事实问题。详见 `stage7_gpu_scale/`。

### 阶段八：知识问答 SFT —— "记忆会，检索不会"

在 100M 百科基座上做 SFT（复用阶段六机制，全量 vs LoRA），用 20 道常识题评估：

- **格式照旧学会**（第三次验证），**LoRA 用 1% 参数再次接近全量**（test loss
  3.87 vs 4.03）
- **常识题全军覆没**（≈0/20）：归因有三——测试题不在维基的句式分布里、
  100M 模型没有"定点检索"能力、评估设计偏乐观
- 结论：**知识以模糊关联和原句复现存在（"故宫位于北京"能复现），但无法以
  问答形式取出**——"记忆会，检索不会"。要跨过这道坎：更大规模，或 RAG。
  详见 `stage8_knowledge_sft/`。

## 八个阶段的关系

```
阶段一：同一套代码，数据"有没有规则" → 学算法 vs 背答案
阶段二：同一套代码，数据"够不够多"   → 背答案 → 学规律（质变）
阶段三：换一种架构，目标"预测下一个" → 从 seq2seq 到语言模型（LLM 范式）
阶段四：追问"为什么不连贯"          → 规模 + 分词（scaling law 与 BPE）
阶段五：把技术合起来推到极限          → 风格学到极致，连贯仍需涌现
阶段六：后训练 SFT + LoRA           → 从"续写"到"应答"，行为可教、知识难注入
阶段七：GPU 规模训练（100M/150MB）  → 连贯之坎跨过，知识以模糊关联存在
阶段八：百科基座上的知识问答 SFT     → 记忆会、检索不会（知识边界画像）
```

`CONCEPTS.md` 是贯穿六个阶段的概念主线，建议在看完阶段一代码后阅读，
再带着它的结论进入后面各阶段的实验。

## 运行环境

- **Python 3.10+ 与 PyTorch**（`pip install torch`）。阶段一~六在 CPU 上即可
  完整复现（脚本会自动检测 GPU，有则用 GPU）；阶段七~八的数据提取与规模训练
  在 NVIDIA GPU 上完成（DGX Spark 实测，torch 2.13 + CUDA 13，任何
  Ampere 以上、显存 ≥ 16GB 的 GPU 亦可）。
- 各脚本 docstring 里的运行命令以 `python` 开头，请确保 torch 装在你
  激活的 Python 环境里。
- 阶段七需要从维基百科下载 dump（约 250MB/段），见 `stage7_gpu_scale/build_corpus.py`。
- 模型权重（`*.pt`）与训练日志不入库（见 `.gitignore`）——clone 后按各阶段
  README 重新训练即可复现（阶段五起约数小时 CPU / 数十分钟 GPU）。
