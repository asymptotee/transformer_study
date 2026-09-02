# 工程架构与 model.py 详解

本文档是对整个 `transformer_study` 工程的架构梳理，以及对核心文件 `model.py` 的
逐行讲解。面向自学：既讲"是什么"，也讲"为什么"。

目录：

1. [工程总览](#一工程总览)
2. [模块依赖关系](#二模块依赖关系)
3. [两条数据流](#三两条数据流)
4. [两个任务的差异](#四两个任务的差异)
5. [model.py 逐行讲解](#五modelpy-逐行讲解)
6. [torch API 速查表](#六torch-api-速查表)
7. [贯穿全文件的三条规律](#七贯穿全文件的三条规律)
8. [形状流转详解](#八形状流转详解)
9. [实例：abc→cba 全流程](#九实例abccba-全流程)

---

## 一、工程总览

从零手写一个完整的 encoder-decoder Transformer（《Attention Is All You Need》结构），
用字符级中文小语料训练，约 70 万参数，CPU 上几分钟训完。刻意不用
`torch.nn.Transformer` 高层封装，注意力、掩码、位置编码全部手写。

```
├── env/                 # PyTorch venv（阶段一~六 CPU 即可，阶段七~八需 GPU）
└── transformer_study/    # 核心：从零手写 Transformer
    ├── model.py          # 全部模型模块（核心，本文档重点）
    ├── data.py           # 字符分词器、双任务数据管道
    ├── train.py          # 训练循环（teacher forcing）+ 评估指标
    ├── generate.py       # 自回归生成（贪心 / 温度采样）
    ├── eval_ckpts.py     # 一次性评估两个 checkpoint
    ├── corpus.txt        # 小语料：唐诗 + 谚语
    ├── ckpt.pt           # reverse 任务检查点
    └── ckpt_complete.pt  # complete 任务检查点
```

运行时用装有 torch 的 Python 环境（可自建 venv：`python -m venv env && pip install torch`）：

```bash
PY=python
```

---

## 二、模块依赖关系

依赖是严格单向的：`model.py` 不依赖任何同目录模块；`data.py` 只用 torch；
三个入口脚本组合这两层。`eval_ckpts.py` 还复用了 `train.evaluate`。

```
corpus.txt（原始语料，唐诗+谚语）
     │
     ▼
┌─────────────────────────────────────────────────┐
│  data.py  —— 数据层                              │
│  CharTokenizer  字符↔id（词表 = 4特殊符 + 语料字符）│
│  make_pairs     按任务切 (src, tgt) 文本对         │
│  Seq2SeqDataset 编码成张量，tgt 加 <bos>/<eos>    │
│  collate        动态 padding 组 batch             │
│  random_lines   合成随机串（reverse 防背诵）        │
└────────────────────┬────────────────────────────┘
                     │ 被引用
     ┌───────────────┼───────────────┐
     ▼               ▼               ▼
┌──────────┐   ┌──────────┐   ┌──────────────┐
│ model.py │   │ train.py │   │ generate.py  │
│ 模型层    │◀──│ 训练入口  │   │ 推理入口      │
│ (无依赖)  │◀──┴──────────┘   │              │
└──────────┘   └──────┬───────┴──────────────┘
                      │ 复用 evaluate()
                      ▼
                ┌──────────────┐
                │ eval_ckpts.py│  一次性评估两个 ckpt
                └──────────────┘
```

### model.py 内部结构（自底向上）

```
TransformerConfig (dataclass, 全部超参)
        │
scaled_dot_product_attention()   ← 最底层函数：softmax(QKᵀ/√d)·V
        │
MultiHeadAttention               ← 拆头/投影/合并；一个类三种用途
        │                            （自注意力/因果自注意力/交叉注意力，只靠输入和掩码区分）
        ├── PositionalEncoding   ← sin/cos 固定编码，register_buffer
        ├── FeedForward          ← 两层 MLP + GELU（唯一的非线性来源）
        │
EncoderBlock = 自注意力 + FFN（各带 pre-norm 残差）
DecoderBlock = 因果自注意力 + 交叉注意力 + FFN
        │
Transformer（组装）
   ├── embedding（src/tgt 共享）+ lm_head（与 embedding 权重绑定）
   ├── pad_mask / causal_mask（两类掩码的工厂方法）
   ├── encode() / decode()（拆开是为了推理时 encoder 只跑一次）
   ├── forward()  → 训练：teacher forcing 并行
   └── generate() → 推理：自回归逐 token
```

两个相对原论文的现代化改动：**pre-norm**（残差路径干净、无需 warmup）和
**weight tying**（输出层与 embedding 共享权重）。

---

## 三、两条数据流

### 训练（train.py，teacher forcing）

```
corpus → split_train_val → make_pairs → Dataset → DataLoader
  → model(src, tgt[:, :-1])  → logits
  → cross_entropy(logits, tgt[:, 1:], ignore_index=PAD)
  → AdamW → 按 val_loss 或 train_loss 存最佳 ckpt
```

关键：输入右移一位、标签左移一位，所有位置并行算损失。checkpoint 里同时存了
`config + vocab + task`，推理时自包含。

### 推理（generate.py / model.generate）

```
prompt → encode → ys=[<bos>]
  循环：decode(ys) 取末位 logits → argmax 或温度采样 → 追加 → 遇 <eos> 停
```

---

## 四、两个任务的差异

整套代码没有一处 `if task` 分支在模型里——**任务差异全部外推到数据和评估策略**：

| | reverse | complete |
|---|---|---|
| `make_pairs` 切法 | `(line, line[::-1])` | `(line[:mid], line[mid:])` |
| 训练数据 | 混入 4000 条合成随机串 | 纯真实语料 |
| 本质 | 学对齐算法（可泛化） | 记忆语料（不泛化） |
| 挑 ckpt | `val_loss` | `train_loss` |

---

## 五、model.py 逐行讲解

### 5.1 文件头与导入（1–21 行）

```python
16  import math
17  from dataclasses import dataclass
18
19  import torch
20  import torch.nn as nn
21  import torch.nn.functional as F
```

- **16 `import math`**：Python 标准数学库，只为 `math.sqrt`（开方）和 `math.log`
  （自然对数）。这里只是算一个 Python 标量常数，不需要张量，所以用 math 而非 torch。
- **17 `dataclass`**：用来定义配置类（见下），自动生成 `__init__`，省得手写构造函数。
- **19 `torch`**：主库，提供张量、初始化、设备管理。
- **20 `nn`**：神经网络"积木层"——`Linear`、`Embedding`、`LayerNorm`、`Module` 等
  **带可学习参数**的组件。
- **21 `F`**：`nn` 的函数版——`softmax` 这类**纯计算、无参数**的操作。
  惯例：有参数用 `nn`，纯运算用 `F`。

### 5.2 配置类（24–35 行）

```python
24  @dataclass
25  class TransformerConfig:
28      vocab_size: int       # 词表大小
29      d_model: int = 128    # 模型宽度
30      n_heads: int = 4      # 注意力头数
31      n_layers: int = 2     # enc/dec 各自层数
32      d_ff: int = 256       # 前馈隐藏层维度
33      dropout: float = 0.1
34      max_len: int = 64     # 最大序列长度
35      pad_id: int = 0       # padding 的索引
```

- **24 `@dataclass`**：装饰器。让你只写字段声明（`名字: 类型 = 默认值`），它自动
  生成构造函数。于是 `TransformerConfig(vocab_size=300)` 就能用，其余字段取默认值。
- **28 `vocab_size: int`**：**唯一没有默认值**的字段，必须显式传——因为它由语料
  决定（`train.py:104` 传 `len(tokenizer)`）。
- 各超参含义：
  - `d_model=128`：每个 token 被表示成 128 维向量，是整个网络的"宽度"，所有层都
    必须保持这个维度。
  - `n_heads=4`：注意力拆成 4 个并行子空间，每个宽度 `d_k = 128/4 = 32`。
  - `n_layers=2`：encoder 和 decoder **各**堆 2 层。
  - `d_ff=256`：前馈层中间撑到 256 维再压回 128（惯例 2~4 倍）。
  - `max_len=64`：位置编码表只建 64 行，超过就崩。

> 这里没有 torch，纯 Python。

### 5.3 缩放点积注意力（38–66 行）—— 全文件最底层

```python
38  def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
51      d_k = q.size(-1)
54      scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
56      if mask is not None:
61          scores = scores.masked_fill(~mask, float("-inf"))
63      attn = F.softmax(scores, dim=-1)
64      if dropout is not None:
65          attn = dropout(attn)
66      return attn @ v, attn
```

这是注意力公式 `softmax(QKᵀ/√d_k)·V` 的直接翻译。输入 `q, k, v` 形状都是
`(batch, n_heads, seq_len, d_k)`。

- **51 `d_k = q.size(-1)`**：取最后一维的大小，即每个头的维度 `d_k`。`size(-1)` 的
  `-1` 表示倒数第一维。
- **54 `scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)`**：一行干了公式的前半段。
  - `k.transpose(-2, -1)`：把 k 的**最后两维对调**，`(B,H,T,d_k)` → `(B,H,d_k,T)`。
    这是为了下一步矩阵乘法能对齐。
  - `q @ ...`：`@` 是矩阵乘法运算符。`(B,H,T_q,d_k) @ (B,H,d_k,T_k)` →
    `(B,H,T_q,T_k)`。结果的 `[b,h,i,j]` 就是"第 i 个 query 和第 j 个 key 的点积相似度"。
  - `/ math.sqrt(d_k)`：除以 √d_k 缩放。维度越大点积方差越大，不缩放会把 softmax
    推进饱和区导致梯度消失。
- **61 `scores.masked_fill(~mask, float("-inf"))`**：把要屏蔽的位置填成负无穷。
  - `mask` 是 bool 张量，`True`=能看见。`~mask` 是逐元素取反，得到"要屏蔽的位置"。
  - `masked_fill(条件, 值)`：条件为 True 处填入该值。填 `-inf` 是因为下一步
    `e^(-inf)=0`，softmax 后这些位置权重彻底归零。
- **63 `F.softmax(scores, dim=-1)`**：沿最后一维（key 那一维）归一化成概率分布，
  每行加起来等于 1。这就是注意力权重。
- **65 `dropout(attn)`**：训练时随机把一部分权重置零，防过拟合。注意这里的 `dropout`
  是传进来的 `nn.Dropout` 对象，**当函数调用**（`nn.Module` 都实现了 `__call__`）。
- **66 `attn @ v`**：`(B,H,T_q,T_k) @ (B,H,T_k,d_k)` → `(B,H,T_q,d_k)`。按注意力权重
  对 value 加权求和——"汇总相关位置的信息"。返回 `(输出, 注意力权重)`，权重留着可可视化。

**本段 torch API**：`size`、`transpose`、`@`(matmul)、`masked_fill`、`~`(取反)、`F.softmax`。

### 5.4 多头注意力（69–107 行）

```python
69  class MultiHeadAttention(nn.Module):
82      def __init__(self, d_model, n_heads, dropout=0.1):
83          super().__init__()
84          assert d_model % n_heads == 0
85          self.n_heads = n_heads
86          self.d_k = d_model // n_heads
88          self.w_q = nn.Linear(d_model, d_model, bias=False)
89          self.w_k = nn.Linear(d_model, d_model, bias=False)
90          self.w_v = nn.Linear(d_model, d_model, bias=False)
91          self.w_o = nn.Linear(d_model, d_model, bias=False)
92          self.dropout = nn.Dropout(dropout)
```

- **69 `class ...(nn.Module)`**：继承 `nn.Module`——所有自定义网络层的基类。继承它，
  你注册的子层和参数才会被框架自动追踪。
- **83 `super().__init__()`**：**必须第一行调用**。它初始化 `nn.Module` 内部的参数
  登记簿；漏了它，后面注册的层全部失效。
- **84 `assert d_model % n_heads == 0`**：断言能整除，否则拆头时维度对不上，直接报错
  而非埋下隐患。
- **88–91 四个 `nn.Linear(d_model, d_model, bias=False)`**：四个全连接投影矩阵
  W_Q、W_K、W_V、W_O。`Linear(in, out)` 做 `y = xWᵀ`。`bias=False` 去掉偏置项——
  现代 LLM 的常见做法（原论文是带的）。
  - 注意前三个把 `d_model` 投到 `d_model`，**不是**投到 `d_k`：是先整体投影，再靠
    `view` 拆成多头（见下），这样更省代码且等价。
- **92 `nn.Dropout(dropout)`**：注册一个 dropout 层，待会儿用在注意力权重上。

```python
94      def forward(self, q, k, v, mask=None):
95          B = q.size(0)
98          q = self.w_q(q).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
99          k = self.w_k(k).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
100         v = self.w_v(v).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
102         out, attn = scaled_dot_product_attention(q, k, v, mask=mask, dropout=self.dropout)
106         out = out.transpose(1, 2).contiguous().view(B, -1, self.n_heads * self.d_k)
107         return self.w_o(out), attn
```

`forward` 定义前向计算。`nn.Module` 的约定：你只写 `forward`，调用时写 `layer(x)`
（框架会自动走钩子再进 `forward`）。

- **95 `B = q.size(0)`**：取 batch 大小（第 0 维）。
- **98 这一行是"拆头"的精髓**，分三步：
  1. `self.w_q(q)`：线性投影，`(B, T, d_model)` → `(B, T, d_model)`。
  2. `.view(B, -1, n_heads, d_k)`：把最后一维 128 **重新解释**成 `(4, 32)`，变成
     `(B, T, 4, 32)`。`view` 不改数据只改"看法"，`-1` 让 torch 自动推断（这里推出 T）。
  3. `.transpose(1, 2)`：把"头"那维挪到前面，`(B, T, 4, 32)` → `(B, 4, T, 32)`，
     即 `(B, H, T, d_k)`。这样所有头就能用一次批量矩阵乘并行算（上一段的 `@` 正是吃这个形状）。
- **102**：调用上一部分的底层函数，真正算注意力。
- **106 "合并多头"，拆头的逆操作**：
  1. `out.transpose(1, 2)`：`(B, H, T, d_k)` → `(B, T, H, d_k)`，把头挪回去。
  2. `.contiguous()`：**关键坑**。`transpose` 只改索引方式不改内存布局，此时内存是
     "乱序"的；而 `view` 要求内存连续，所以必须先 `contiguous()` 把数据在内存里重排整齐。
  3. `.view(B, -1, n_heads * d_k)`：把 `(H, d_k)` 两维拍平回 `d_model`，回到 `(B, T, d_model)`。
- **107 `self.w_o(out)`**：最后一个投影 W_O，把多头拼接结果混合成最终输出。

**本段 torch API**：`nn.Module`、`super().__init__`、`nn.Linear`、`nn.Dropout`、
`size`、`view`、`transpose`、`contiguous`。

### 5.5 位置编码（110–133 行）

```python
110 class PositionalEncoding(nn.Module):
120     def __init__(self, d_model, max_len=512, dropout=0.1):
121         super().__init__()
122         self.dropout = nn.Dropout(dropout)
123         pe = torch.zeros(max_len, d_model)
124         position = torch.arange(0, max_len).unsqueeze(1).float()
126         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
127         pe[:, 0::2] = torch.sin(position * div_term)
128         pe[:, 1::2] = torch.cos(position * div_term)
130         self.register_buffer("pe", pe.unsqueeze(0))
```

注意力本身对顺序不敏感（打乱输入输出不变），必须显式注入位置。这里在 `__init__` 里
**一次性算好整张位置表**（不是可学习参数，是固定公式）。

- **123 `torch.zeros(max_len, d_model)`**：建一个 `(64, 128)` 全 0 表，待会儿往里填。
- **124 `torch.arange(0, max_len).unsqueeze(1).float()`**：
  - `arange(0, 64)`：生成 `[0,1,2,...,63]`，形状 `(64,)`。
  - `.unsqueeze(1)`：在第 1 维插入一个大小为 1 的维度，`(64,)` → `(64, 1)`。这是为了下一步广播。
  - `.float()`：转成浮点型（arange 默认是整型，不能做三角函数）。
- **126 `div_term`**：公式里的 `1/10000^(2i/d_model)`，每个维度一个频率。
  - `arange(0, d_model, 2)`：`[0,2,4,...,126]`，共 64 个（步长 2）。
  - 用 `exp(-2i·ln(10000)/d_model)` 算 `1/10000^(2i/d)`——这是数值上更稳的等价写法
    （避免直接算大数幂）。
  - 结果形状 `(64,)`。
- **127–128 填表 + 广播**：
  - `position * div_term`：`(64,1) * (64,)` → 广播成 `(64,64)` 的外积矩阵，
    `[pos, i]` = `pos / 10000^(2i/d)`。
  - `torch.sin(...)` / `torch.cos(...)`：逐元素取正弦/余弦。
  - `pe[:, 0::2]`：**切片赋值**，`0::2` 表示偶数列（0,2,4...）填 sin；`1::2` 奇数列
    填 cos。对应公式 PE(pos,2i)=sin、PE(pos,2i+1)=cos。
- **130 `self.register_buffer("pe", pe.unsqueeze(0))`**：
  - `unsqueeze(0)`：前面加一维 batch，`(64,128)` → `(1,64,128)`，方便和 `(B,T,d)` 广播。
  - `register_buffer`：把它登记为"buffer"——**不是可学习参数**（优化器不会更新它），
    但会随模型一起 `.to(device)` 移动设备、存进 checkpoint。这是放固定常量的标准做法。

```python
132     def forward(self, x):
133         return self.dropout(x + self.pe[:, : x.size(1)])
```

- **133**：`x.size(1)` 是当前序列长度 T；`self.pe[:, :T]` 切出前 T 行位置编码，直接
  加到输入上（广播到 batch）。位置信息和词义信息从此融合在一起。

**本段 torch API**：`torch.zeros`、`torch.arange`、`unsqueeze`、`.float()`、
`torch.exp`、`torch.sin`/`torch.cos`、切片赋值、`register_buffer`、`size`。

### 5.6 前馈网络（136–154 行）

```python
136 class FeedForward(nn.Module):
142     def __init__(self, d_model, d_ff, dropout=0.1):
143         super().__init__()
144         self.net = nn.Sequential(
145             nn.Linear(d_model, d_ff),
146             nn.GELU(),
147             nn.Dropout(dropout),
148             nn.Linear(d_ff, d_model),
149             nn.Dropout(dropout),
150         )
153     def forward(self, x):
154         return self.net(x)
```

- **144 `nn.Sequential(...)`**：容器，把多个层**按顺序串起来**，数据依次流过。
  `self.net(x)` 一行就等于把这 5 层挨个作用一遍。
- **145 `Linear(d_model, d_ff)`**：128 → 256，先撑开。
- **146 `nn.GELU()`**：激活函数，引入非线性。**注意力层本质是线性加权，整个网络的
  非线性表达能力全靠这种激活层**。GELU 比原论文的 ReLU 更平滑，是现代选择。
- **148 `Linear(d_ff, d_model)`**：256 → 128，压回原宽度（这样残差才能相加）。
- 结构 `128→256→128`，像个沙漏，对每个 token 位置独立做同样的加工。

**本段 torch API**：`nn.Sequential`、`nn.Linear`、`nn.GELU`、`nn.Dropout`。

### 5.7 Encoder 单层（157–178 行）

```python
157 class EncoderBlock(nn.Module):
166     def __init__(self, d_model, n_heads, d_ff, dropout):
167         super().__init__()
168         self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
169         self.ff = FeedForward(d_model, d_ff, dropout)
170         self.norm1 = nn.LayerNorm(d_model)
171         self.norm2 = nn.LayerNorm(d_model)
173     def forward(self, x, src_mask):
174         h = self.norm1(x)
175         attn_out, _ = self.self_attn(h, h, h, mask=src_mask)
176         x = x + attn_out
177         x = x + self.ff(self.norm2(x))
178         return x
```

- **168–169**：注册两个子模块——一个多头注意力、一个前馈。
- **170–171 `nn.LayerNorm(d_model)`**：层归一化，把每个 token 的 128 维向量归一化到
  均值 0 方差 1，稳定训练。两个子层各配一个。
- **174–176 第一个子层（pre-norm 残差）**：
  - `h = self.norm1(x)`：先归一化。
  - `self.self_attn(h, h, h, ...)`：**q=k=v 都是 h**，这就是"自注意力"——序列自己
    和自己算注意力。`_` 丢弃返回的注意力权重。
  - `x = x + attn_out`：**残差连接**。注意加的是**没归一化的原始 x**——这是 pre-norm
    写法，残差路径是干净的恒等映射，梯度能直通到底，深层才训得动。
- **177 第二个子层**：`x + ff(norm2(x))`，同样的"归一化→子层→残差"模式，一行写完。

**本段 torch API**：`nn.LayerNorm`（外加复用前面的 `MultiHeadAttention`、`FeedForward`）。

### 5.8 Decoder 单层（181–211 行）

```python
181 class DecoderBlock(nn.Module):
192     def __init__(self, d_model, n_heads, d_ff, dropout):
193         super().__init__()
194         self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
195         self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
196         self.ff = FeedForward(d_model, d_ff, dropout)
197         self.norm1 = nn.LayerNorm(d_model)
198         self.norm2 = nn.LayerNorm(d_model)
199         self.norm3 = nn.LayerNorm(d_model)
201     def forward(self, x, enc_out, tgt_mask, src_mask):
202         h = self.norm1(x)
203         attn_out, _ = self.self_attn(h, h, h, mask=tgt_mask)
204         x = x + attn_out
205         h = self.norm2(x)
208         cross_out, _ = self.cross_attn(h, enc_out, enc_out, mask=src_mask)
209         x = x + cross_out
210         x = x + self.ff(self.norm3(x))
211         return x
```

比 encoder 多一个子层，共三个。注意 **194 和 195 是同一个类的两个独立实例**——
再次印证"三种注意力都是同一个 `MultiHeadAttention`，区别只在喂什么数据"。

- **203 因果自注意力**：`q=k=v=h`（decoder 序列自身），但 `tgt_mask` 是**因果掩码
  ∩ 填充掩码**，保证位置 t 只能看 ≤t 的位置，不能偷看未来。
- **208 交叉注意力（seq2seq 信息流动的关键）**：
  - `q = h`（来自 decoder），`k = v = enc_out`（来自 encoder 输出）。
  - decoder 借此"回头看"源序列：生成每个输出 token 时，去源序列里查该关注哪些位置。
    reverse 任务里，这里就是学习"第 i 个输出对齐第 L-1-i 个输入"的地方。
  - `src_mask` 屏蔽源序列的 padding。
- **210**：前馈子层，同 encoder。

**本段 torch API**：无新增（全是复用）。

### 5.9 完整 Transformer（214–328 行）

#### 组装（226–259 行）

```python
214 class Transformer(nn.Module):
226     def __init__(self, cfg: TransformerConfig):
227         super().__init__()
228         self.cfg = cfg
229         self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
230         self.pos_enc = PositionalEncoding(cfg.d_model, cfg.max_len, cfg.dropout)
231         self.encoder = nn.ModuleList(
232             EncoderBlock(...) for _ in range(cfg.n_layers)
233         )
235         self.decoder = nn.ModuleList(
236             DecoderBlock(...) for _ in range(cfg.n_layers)
238         )
240         self.enc_norm = nn.LayerNorm(cfg.d_model)
241         self.dec_norm = nn.LayerNorm(cfg.d_model)
242         self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
245         self.lm_head.weight = self.embedding.weight
246         self._init_weights(cfg)
```

- **229 `nn.Embedding(vocab_size, d_model, padding_idx=0)`**：嵌入层，本质一张
  `(词表大小, 128)` 的查找表——输入 token id，输出对应那行向量。`padding_idx=0` 让
  `<pad>` 那行恒为 0 且不更新。
- **231 `nn.ModuleList([...])`**：装子模块的列表。**不能用普通 Python list**——
  普通 list 里的层不会被 `nn.Module` 追踪，参数就丢了。`ModuleList` 保证 N 个
  `EncoderBlock` 的参数都被正确注册。这里 N=2。
- **240–241**：pre-norm 的收尾归一化。因为残差路径上的 x 从没归一化过，输出前补一次。
- **242 `lm_head = Linear(d_model, vocab_size)`**：输出头，把 128 维向量投回词表大小，
  得到每个词的 logits（未归一化分数）。
- **245 `self.lm_head.weight = self.embedding.weight`**：**权重绑定（weight tying）**。
  让输出头和 embedding 共享同一张表——"id→向量"和"向量→id"本该互逆。省一半这部分
  参数，还助泛化。注意是直接把同一个张量对象赋过去，不是复制。
- **246**：调用自定义初始化（见下）。

```python
248     def _init_weights(self, cfg):
256         nn.init.normal_(self.embedding.weight, mean=0.0, std=cfg.d_model ** -0.5)
257         with torch.no_grad():
259             self.embedding.weight[cfg.pad_id].fill_(0.0)
```

- **256 `nn.init.normal_(张量, mean, std)`**：用正态分布**原地**填充张量（结尾下划线
  `_` 是 torch 约定：原地修改）。std 取 `1/√d_model`——这样乘上后面的 `√d_model`
  缩放后量级回到 1，和位置编码（±1）平衡。**这是训练能否启动的关键**：用 PyTorch
  默认 N(0,1) 会让注意力分数爆炸、softmax 饱和、梯度归零（README 坑 1）。
- **257 `with torch.no_grad():`**：上下文管理器，临时关闭梯度追踪。这里是纯改数值
  不需要算梯度，关掉省内存。
- **259 `.fill_(0.0)`**：原地填 0。因为 256 行的 `normal_` 覆盖了 `padding_idx` 的
  零初始化约定，这里手动把 `<pad>` 那行补回全 0。

#### 掩码工厂（263–272 行）

```python
263     def pad_mask(self, ids):
266         return (ids != self.cfg.pad_id).view(ids.size(0), 1, 1, ids.size(1))
268     @staticmethod
269     def causal_mask(size, device):
272         return torch.tril(torch.ones(size, size, dtype=torch.bool, device=device)).view(1, 1, size, size)
```

- **266 填充掩码**：
  - `ids != pad_id`：逐元素比较，得到 bool 张量，真实 token 处为 True。
  - `.view(B, 1, 1, T)`：reshape 成 `(B,1,1,T)`，中间两个 1 是为了**广播**到注意力
    分数的 `(B,H,T_q,T_k)`——同一 batch 内所有头、所有 query 共享这套 key 掩码。
- **268 `@staticmethod`**：静态方法，不依赖实例状态（因果掩码只和长度有关）。
- **272 因果掩码**：
  - `torch.ones(size, size, dtype=torch.bool, device=device)`：建 `(T,T)` 全 True 方阵。
    `dtype=torch.bool` 直接建 bool 型；`device` 保证和模型在同一设备。
  - `torch.tril(...)`：取**下三角**（lower triangle），对角线以上全变 False——
    位置 t 只能看 ≤t 的位置。
  - `.view(1,1,T,T)`：加两个广播维。

#### 前向（276–301 行）

```python
276     def embed(self, ids):
278         return self.pos_enc(self.embedding(ids) * math.sqrt(self.cfg.d_model))
280     def encode(self, src):
283         src_mask = self.pad_mask(src)
284         x = self.embed(src)
285         for block in self.encoder:
286             x = block(x, src_mask)
287         return self.enc_norm(x), src_mask
289     def decode(self, tgt, enc_out, src_mask):
292         tgt_mask = self.causal_mask(tgt.size(1), tgt.device) & self.pad_mask(tgt)
293         x = self.embed(tgt)
294         for block in self.decoder:
295             x = block(x, enc_out, tgt_mask, src_mask)
296         return self.lm_head(self.dec_norm(x))
298     def forward(self, src, tgt):
300         enc_out, src_mask = self.encode(src)
301         return self.decode(tgt, enc_out, src_mask)
```

- **278 `embed`**：查 embedding → 乘 `√d_model`（让量级和位置编码匹配）→ 加位置编码。
  src 和 tgt 共用这套。
- **285–286**：数据依次流过 N 个 encoder block（`ModuleList` 可直接 for 循环）。
- **287**：收尾归一化，返回 `(encoder输出, 源掩码)`。**单独拆出 `encode` 是为了推理
  时 encoder 只跑一次、结果反复用**。
- **292 `&`**：两个 bool 掩码**逐元素与**——decoder 自注意力同时受因果掩码和填充掩码
  约束。靠广播自动对齐形状。
- **296**：decoder 输出过 `lm_head`，得到 `(B, T, vocab_size)` 的 logits。
- **298–301 `forward`**：训练入口。`model(src, tgt)` 实际就调这里（teacher forcing）。

#### 推理生成（305–328 行）

```python
305     @torch.no_grad()
306     def generate(self, src, bos_id, eos_id, max_new_tokens, temperature=0.0):
315         enc_out, src_mask = self.encode(src)
316         B = src.size(0)
317         ys = torch.full((B, 1), bos_id, dtype=torch.long, device=src.device)
318         for _ in range(max_new_tokens):
319             logits = self.decode(ys, enc_out, src_mask)[:, -1]
320             if temperature <= 0:
321                 next_id = logits.argmax(dim=-1, keepdim=True)
322             else:
323                 probs = F.softmax(logits / temperature, dim=-1)
324                 next_id = torch.multinomial(probs, num_samples=1)
325             ys = torch.cat([ys, next_id], dim=1)
326             if (next_id == eos_id).all():
327                 break
328         return ys[:, 1:]
```

- **305 `@torch.no_grad()`**：装饰器版，整个函数内关闭梯度——推理不需要反向传播，
  省内存省计算。
- **315**：encoder 只编码**一次**，结果在循环里反复用。
- **317 `torch.full((B,1), bos_id, dtype=torch.long, ...)`**：造一个 `(B,1)` 的张量
  全填 `<bos>`，作为生成起点。`dtype=torch.long` 因为 token id 是整数。
- **319 `self.decode(...)[:, -1]`**：decode 整个已生成序列，但**只取最后一个位置**
  的 logits——只有它才预测"下一个 token"。`[:, -1]` 是切片：所有 batch、倒数第一行。
- **320–324 两种解码策略**：
  - **贪心**（temperature≤0）：`argmax(dim=-1, keepdim=True)` 取分数最大的 id。
    `keepdim=True` 保留被压缩的那一维（结果 `(B,1)` 而非 `(B,)`），方便下一步拼接。
  - **温度采样**：`logits / temperature` 后再 softmax。温度越高分布越平、越随机；
    `torch.multinomial(probs, num_samples=1)` 按概率分布抽样 1 个。
- **325 `torch.cat([ys, next_id], dim=1)`**：沿序列维（dim=1）把新 token 拼到已生成序列尾部。
- **326 `(next_id == eos_id).all()`**：`.all()` 判断是否**所有** batch 样本都生成了
  结束符，是则提前停。
- **328 `ys[:, 1:]`**：切掉开头的 `<bos>`，只返回真正生成的内容。

**本段 torch API**：`nn.Embedding`、`nn.ModuleList`、`nn.init.normal_`、
`torch.no_grad`、`.fill_`、`torch.tril`、`torch.ones`、`&`、`torch.full`、`argmax`、
`torch.multinomial`、`torch.cat`、`.all()`、切片。

---

## 六、torch API 速查表

按用途分类，这就是整个 `model.py` 用到的全部 torch 能力。

### 张量创建

| API | 作用 | 出现处 |
|---|---|---|
| `torch.zeros(shape)` | 全 0 张量 | 123 |
| `torch.ones(shape, dtype, device)` | 全 1 张量 | 272 |
| `torch.full(shape, 值)` | 全填指定值 | 317 |
| `torch.arange(start, stop, step)` | 等差序列 `[0,1,2,...]` | 124, 126 |
| `torch.tril(x)` | 取下三角 | 272 |

### 张量运算

| API | 作用 | 出现处 |
|---|---|---|
| `@` | 矩阵乘法 | 54, 66 |
| `torch.sin` / `torch.cos` / `torch.exp` | 逐元素函数 | 127, 128, 126 |
| `x.transpose(a, b)` | 交换两维 | 54, 98, 106 |
| `x.view(shape)` | 改形状不改数据（要求内存连续） | 98, 106, 266, 272 |
| `x.unsqueeze(dim)` | 插入一个大小为 1 的维 | 124, 130 |
| `x.contiguous()` | 内存重排整齐（view 前置） | 106 |
| `x.size(dim)` | 取某维大小 | 51, 95, 133, 292, 316 |
| `x.masked_fill(条件, 值)` | 条件处填值 | 61 |
| `torch.cat([a,b], dim)` | 沿维拼接 | 325 |
| `x.argmax(dim, keepdim)` | 取最大值索引 | 321 |
| `torch.multinomial(probs, n)` | 按概率抽样 | 324 |
| `x.all()` | 是否全部为真 | 326 |
| `x.fill_(值)` | 原地填值 | 259 |
| `~x` / `&` / `!=` | bool 取反/与/比较 | 61, 266, 292, 326 |
| 切片 `[:, -1]`、`[:, 0::2]` | 取子集 | 127, 319, 328 |

### 神经网络组件（`nn`，带参数）

| API | 作用 | 出现处 |
|---|---|---|
| `nn.Module` | 所有自定义层的基类 | 69 等 |
| `nn.Linear(in, out, bias)` | 全连接层 `y=xWᵀ` | 88-91, 145, 148, 242 |
| `nn.Embedding(n, d, padding_idx)` | id→向量查找表 | 229 |
| `nn.LayerNorm(d)` | 层归一化 | 170, 240 等 |
| `nn.Dropout(p)` | 随机置零防过拟合 | 92, 122 等 |
| `nn.GELU()` | 激活函数（非线性） | 146 |
| `nn.Sequential(...)` | 顺序容器 | 144 |
| `nn.ModuleList([...])` | 可被追踪的层列表 | 231, 235 |
| `nn.init.normal_(t, mean, std)` | 正态初始化 | 256 |
| `register_buffer(name, t)` | 登记非参数张量 | 130 |

### 函数式（`F`，无参数）

| API | 作用 | 出现处 |
|---|---|---|
| `F.softmax(x, dim)` | 归一化成概率分布 | 63, 323 |

### 梯度控制

| API | 作用 | 出现处 |
|---|---|---|
| `torch.no_grad()` | 关闭梯度追踪（推理/改数值） | 257, 305 |

---

## 七、贯穿全文件的三条规律

1. **形状流转是读懂的关键**：所有操作都在摆弄 `(B, T, d_model)`、`(B, H, T, d_k)`、
   `(B, H, T_q, T_k)` 这几种形状，`view`/`transpose`/`unsqueeze` 就是在它们之间切换。
2. **`nn` vs `F` 的分工**：要学习/保存的用 `nn`（Linear、Embedding），纯计算的用
   `F`（softmax）。
3. **带下划线的方法原地修改**：`normal_`、`fill_` 直接改张量本身，不返回新张量。

### 张量形状完整流转（一次训练前向）

```
src: (B, T_src)  整数 id
  │ embedding
  ▼
(B, T_src, d_model)
  │ ×√d_model + 位置编码
  ▼
(B, T_src, d_model)
  │ EncoderBlock ×N（拆头时内部变 (B, H, T, d_k)，算完合回）
  ▼
enc_out: (B, T_src, d_model)
  │
  │   tgt: (B, T_tgt) → embed → (B, T_tgt, d_model)
  │                                │ DecoderBlock ×N
  │         cross_attn 用它做 k/v ◀┘
  ▼
(B, T_tgt, d_model)
  │ lm_head
  ▼
logits: (B, T_tgt, vocab_size)
```

---

## 八、形状流转详解

第七节提到"形状流转是读懂的关键"。这一节用项目真实数字
（`d_model=128, n_heads=4, d_k=32`）配一个具体 batch（`B=2, T=5`）把它讲透。

### 8.1 先认字母：每个维度是什么

| 字母 | 含义 | 项目里的值 | 类比 |
|---|---|---|---|
| `B` | batch size，一次喂几个句子 | 2 | 一摞作业本的本数 |
| `T` | sequence length，句子有几个 token | 5 | 每本作业有几行 |
| `d_model` | 模型宽度，每个 token 的向量维度 | 128 | 每行用多少个数字描述 |
| `H` | n_heads，注意力头数 | 4 | 把描述拆成几个"视角" |
| `d_k` | 每个头的维度 = d_model / H | 32 | 每个视角用多少个数字 |
| `T_q` / `T_k` | query 序列长 / key 序列长 | 5 / 5 | 提问方长度 / 被查方长度 |

记住一个恒等式：**`d_model = H × d_k`**（128 = 4 × 32）。整个形状游戏就是把它
**拆开**再**合上**。

### 8.2 三种"标准形状"各自为什么存在

**形状 ①：`(B, T, d_model)` —— 层与层之间的"resting 态"**

```
batch 里第 b 个句子：
  token0 → [128 个数]
  token1 → [128 个数]
  ...
  token4 → [128 个数]
```

这是数据在**层之间流动**的形状。Embedding 输出是它，每个 block 的输入输出也是它。
每个 token 被压成一个 128 维向量，"这个词的含义"全在这 128 个数里。

**形状 ②：`(B, H, T, d_k)` —— 算注意力时的"工作态"**

```
把每个 token 的 128 维，拆成 4 个头 × 32 维：
  头0：token0→[32个数], token1→[32个数], ...
  头1：...
  头2：...
  头3：...
```

为什么要拆？因为**多头注意力要让 4 个头各自独立算**。把 H 维挪到前面（第 1 维），
4 个头就成了 batch 的一部分，一次矩阵乘法就能并行算完所有头——而不是 for 循环算 4 次。

**形状 ③：`(B, H, T_q, T_k)` —— 注意力分数/权重**

```
每个头里，一张 T_q × T_k 的"关注度表格"：
        key0  key1  key2  key3  key4
  q0  [ 0.1   0.5   0.2   0.1   0.1 ]   ← q0 把注意力分给各 key
  q1  [ 0.3   0.1   0.1   0.4   0.1 ]
  q2  [ ...                         ]
  ...
```

这是注意力的**核心产物**：`[b, h, i, j]` = "第 b 句、第 h 个头里，第 i 个 query 对
第 j 个 key 的关注权重"。每行加起来等于 1（softmax 归一化）。reverse 任务训好后，
这张表会呈现清晰的对角线（位置 i 关注位置 L-1-i）。

### 8.3 三个切换操作：view / transpose / unsqueeze

要理解它们的区别，得先知道**张量在内存里是一维摊平的**。

**内存模型（关键前提）**

一个 `(2, 3)` 的张量：

```
逻辑上看：            内存里实际是一维：
[[a, b, c],           a b c d e f
 [d, e, f]]           └─第0行─┘└─第1行─┘
```

张量靠 **stride（步幅）** 记录"沿某一维走一步，内存里要跳过几个元素"：
- 沿 dim 0 走一步（换一行）：跳 3 个 → stride[0] = 3
- 沿 dim 1 走一步（换一列）：跳 1 个 → stride[1] = 1

**"连续（contiguous）"** 的意思是：按逻辑顺序读（a,b,c,d,e,f）正好等于内存顺序。

**`view`：只改"看法"，不动数据**

```python
(B, T, 128)  →  view(B, T, 4, 32)
```

把最后一维 128 **重新解释**成 4×32。内存里一个数都没动，只是告诉 torch：
"原来每 128 个数一组，现在看成 4 组、每组 32 个"。

```
原来：[......128 个数......]  一个 token 的向量
现在：[..32..][..32..][..32..][..32..]  拆成 4 个头
```

**前提：内存必须连续。** 因为 view 假设"逻辑顺序 = 内存顺序"，否则就拆错了。

**`transpose`：交换两个轴，不动数据（只改 stride）**

```python
(B, T, H, d_k)  →  transpose(1, 2)  →  (B, H, T, d_k)
```

把"头"那维从第 2 位挪到第 1 位。**注意：内存里的数据一个没动**，只是把 stride 里
对应两维的步幅对调了。逻辑上现在按 (B,H,T,d_k) 读，但内存还是按 (B,T,H,d_k) 排的。

这就导致一个问题：**transpose 之后内存不连续了**。逻辑顺序和内存顺序对不上了。

```
transpose 前的内存：  头0头1头2头3 | 头0头1头2头3 | ...   (按 token 排)
transpose 后逻辑想读：头0头0头0... | 头1头1头1...         (按头读)
                      ↑ 内存里根本不是这个顺序！
```

所以 `transpose` 后面如果跟 `view`，必须先 `contiguous()`。

**`contiguous`：把内存真正重排整齐**

```python
out.transpose(1, 2).contiguous().view(...)
```

`contiguous()` 会**真正搬动数据**，按当前的逻辑顺序重新在内存里排一遍，让它再次连续。
之后 `view` 才能安全工作。这是 `model.py:106` 那行的由来。

**`unsqueeze`：插入一个大小为 1 的维度**

```python
(B, T)  →  view(B, 1, 1, T)     # 填充掩码
(T, T)  →  view(1, 1, T, T)     # 因果掩码
```

加这些"1 维"纯粹是为了**广播（broadcasting）**。注意力分数是 `(B, H, T_q, T_k)`，
而填充掩码只想表达"哪些 key 是 padding"——这个信息**对所有头、所有 query 都一样**，
所以把 H 和 T_q 那两维设成 1，让 torch 自动把它"复制"到 `(B, H, T_q, T_k)` 上：

```
掩码   (B, 1, 1, T_k)      分数   (B, H, T_q, T_k)
        ↓ 广播（1 自动扩展）
       (B, H, T_q, T_k)    对齐   (B, H, T_q, T_k)   ✓ 可以逐元素运算了
```

### 8.4 完整走一遍：MultiHeadAttention 里的形状流转

用 `B=2, T=5, d_model=128, H=4, d_k=32`，对照 `model.py:94-107`：

```
输入 q: (2, 5, 128)                          ← 形状①，resting 态
   │
   │ self.w_q(q)   线性投影，维度不变
   ▼
(2, 5, 128)
   │
   │ .view(2, -1, 4, 32)   把 128 拆成 4×32
   ▼
(2, 5, 4, 32)                                ← (B, T, H, d_k)
   │
   │ .transpose(1, 2)   把头挪到前面
   ▼
(2, 4, 5, 32)                                ← 形状②，工作态 (B, H, T, d_k)
   │
   │ ══════ 进入 scaled_dot_product_attention ══════
   │
   │ k.transpose(-2, -1):  (2,4,5,32) → (2,4,32,5)   最后两维对调
   │ q @ kᵀ:  (2,4,5,32) @ (2,4,32,5)
   ▼
(2, 4, 5, 5)                                 ← 形状③，注意力分数 (B,H,T_q,T_k)
   │
   │ masked_fill + F.softmax(dim=-1)   每行归一化成权重
   ▼
(2, 4, 5, 5)                                 ← 形状③，注意力权重
   │
   │ attn @ v:  (2,4,5,5) @ (2,4,5,32)
   ▼
(2, 4, 5, 32)                                ← 回到形状② (B, H, T, d_k)
   │
   │ ══════ 回到 MultiHeadAttention ══════
   │
   │ .transpose(1, 2)   把头挪回去
   ▼
(2, 5, 4, 32)                                ← (B, T, H, d_k)，但内存不连续！
   │
   │ .contiguous()   真正重排内存
   ▼
(2, 5, 4, 32)                                ← 形状不变，内存已连续
   │
   │ .view(2, -1, 128)   把 4×32 拍平回 128
   ▼
(2, 5, 128)                                  ← 回到形状①，resting 态
   │
   │ self.w_o(out)   混合多头，维度不变
   ▼
(2, 5, 128)                                  ← 输出，交给残差相加
```

**整个过程的骨架**：

```
①resting  ──view拆头──▶  (B,T,H,d_k)  ──transpose──▶  ②工作态
   ▲                                                      │
   │                                                      ▼
   └──view合头──  (B,T,H,d_k)  ◀──transpose──  ②工作态 ──▶ ③分数 ──▶ ②
      (先 contiguous!)
```

中间鼓起的部分是"打开来看每个头"，两头是"打包好交给下一层"。

### 8.5 为什么非要这么折腾？

一句话：**为了用一次批量矩阵乘算完所有头，而不是循环 4 次。**

- 如果不拆头，你得 `for h in range(4)` 逐个算注意力——慢，且 Python 循环是性能杀手。
- 把 H 挪到前面当 batch 维，`@` 运算符会自动对 `(B, H, ...)` 的每个 `(B,H)` 切片
  并行做矩阵乘——4 个头、2 个句子，8 个注意力计算**一次全完成**。GPU/CPU 的批量
  矩阵乘高度优化，这比循环快几个数量级。

所以 `view`/`transpose`/`contiguous` 这些"形状体操"不是炫技，而是**把数据摆成
批量矩阵乘要求的姿势**。读懂了这条主线，整个 `model.py` 的形状变化就都顺了。

---

## 九、实例：abc→cba 全流程

用一个具体例子把矩阵摆出来，走一遍训练和推理的完整过程。

先说明：真实模型 `d_model=128` 太大写不下，下面用**缩小版参数 + 示意数字**。
数字是为了看清结构编的（训练好的模型大致长这样），**重点看矩阵的形状和模式，
不要纠结具体数值**。

### 9.0 设定

| 项 | 值 |
|---|---|
| 词表 | `<pad>=0, <bos>=1, <eos>=2, a=3, b=4, c=5`（共 6 个） |
| `d_model` | 4 |
| `n_heads` / `d_k` | 2 / 2 |
| 任务 | 反转 `"abc"` → `"cba"` |

为简洁，下面省略 batch 维（当作 `B=1`）。

### 9.1 训练（teacher forcing，一次并行算完）

**① 分词**

```
src = "abc"  →  [3, 4, 5]
tgt = "cba"  →  加特殊符 → [<bos>, c, b, a, <eos>] = [1, 5, 4, 3, 2]
```

代码里（`train.py:133`）会把 tgt 错开一位：

```
decoder 输入 = tgt[:, :-1] = [1, 5, 4, 3]      (bos, c, b, a)   长度 4
标签 labels  = tgt[:, 1:]  = [5, 4, 3, 2]      (c, b, a, eos)   长度 4
```

**第 i 个输入负责预测第 i 个标签**：`bos→c`、`c→b`、`b→a`、`a→eos`。

**② Embedding 查表**

embedding 是一张 `(6, 4)` 的表，每个 token 一行：

```
E =          d0     d1     d2     d3
  <pad>  [  0.00   0.00   0.00   0.00 ]
  <bos>  [  0.10  -0.20   0.30   0.05 ]
  <eos>  [ -0.15   0.25  -0.05   0.10 ]
   a     [  0.21  -0.12   0.53   0.31 ]
   b     [  0.14   0.42  -0.23   0.08 ]
   c     [ -0.31   0.27   0.15   0.62 ]
```

查 src 的 id `[3,4,5]`，取出对应 3 行，得到 `(3, 4)`：

```
X =   a [  0.21  -0.12   0.53   0.31 ]
      b [  0.14   0.42  -0.23   0.08 ]
      c [ -0.31   0.27   0.15   0.62 ]
```

**③ 乘 √d_model + 位置编码**

`X × 2`（√4=2）再逐行加上位置编码 PE(0)、PE(1)、PE(2)，得到进入 encoder 的 `(3,4)`：

```
X' =  pos0(a) [  0.5   -0.2    1.1    0.7 ]
      pos1(b) [  0.3    0.9   -0.4    0.2 ]
      pos2(c) [ -0.5    0.6    0.3    1.3 ]
```

> 注意：此时"a 在第 0 位、b 在第 1 位"这个**顺序信息**已经融进向量里了。没有位置
> 编码，模型根本不知道谁在前谁在后。

**④ Encoder 自注意力**

每个 token 投影出 Q、K、V（`Q = X'·W_Q` 等），算 `softmax(QKᵀ/√d_k)` 得到 `(3,3)`
权重表——"每个位置关注哪些位置"：

```
A_enc =        看a    看b    看c
   a 查询  [  0.50   0.30   0.20 ]
   b 查询  [  0.25   0.50   0.25 ]
   c 查询  [  0.20   0.30   0.50 ]
   （每行加起来 = 1）
```

加权求和后输出 `enc_out`，仍是 `(3, 4)`——每个 token 的向量里现在"掺入了"其他位置的信息。

**⑤ Decoder 因果自注意力**

decoder 输入 `[bos, c, b, a]`（4 个 token）。因果掩码是下三角，**禁止看未来**：

```
因果掩码 =      bos   c    b    a
   bos      [   1    0    0    0 ]   bos 只能看自己
   c        [   1    1    0    0 ]   c 能看 bos、c
   b        [   1    1    1    0 ]
   a        [   1    1    1    1 ]   a 能看全部
```

应用掩码后（禁止位填 `-inf`、softmax 后变 0），自注意力权重也是下三角：

```
A_causal =     bos    c     b     a
   bos     [  1.00   0     0     0   ]
   c       [  0.40  0.60   0     0   ]
   b       [  0.20  0.30  0.50   0   ]
   a       [  0.15  0.25  0.30  0.30]
```

**⑥ Decoder 交叉注意力 —— ⭐ 反转的本质在这里**

decoder 的 Q 去查 encoder 的 K/V：**"要生成这个输出，该看源序列的哪个位置？"**
权重表是 `(4 个 decoder 位置) × (3 个 src 位置)`：

```
A_cross =          src:a   src:b   src:c
   pos0(预测 c) [   0.05    0.05    0.90 ]  ← 盯着 src 的 c
   pos1(预测 b) [   0.05    0.90    0.05 ]  ← 盯着 src 的 b
   pos2(预测 a) [   0.90    0.05    0.05 ]  ← 盯着 src 的 a
   pos3(预测eos)[   0.33    0.33    0.34 ]  ← 结束符，看得很散
```

**看这条从右上到左下的反对角线！** 这就是模型学到的反转算法：
- 输出第 0 位 → 看输入第 2 位（`L-1-0`）
- 输出第 1 位 → 看输入第 1 位（`L-1-1`）
- 输出第 2 位 → 看输入第 0 位（`L-1-2`）

`README` 里说"reverse 训好后注意力热力图能看到清晰对角线"，指的就是这张表
（准确说是反对角线）。**模型不是背下了 abc→cba，而是学会了"第 i 个输出对齐第
L-1-i 个输入"这条规则**——所以没见过的字符串也能反转。

**⑦ 输出头 → logits**

decoder 输出过 `lm_head`（`(4维) → (6维词表)`），得到 `(4, 6)` 的 logits——
每个位置对 6 个词的打分：

```
logits =          pad   bos   eos    a     b     c     argmax  标签
  pos0(预测c)  [ -2.0  -3.0  -1.0   0.1  -1.0   3.5 ]   → c  ✓  c
  pos1(预测b)  [ -2.0  -3.0  -1.0  -1.0   3.2   0.0 ]   → b  ✓  b
  pos2(预测a)  [ -2.0  -3.0  -1.0   3.4  -1.0  -2.0 ]   → a  ✓  a
  pos3(预测eos)[ -2.0  -3.0   3.6  -1.0  -2.0  -1.0 ]   → eos ✓  eos
```

这是一个**训好了**的模型，所以每行最大值（argmax）正好等于标签。

**⑧ 损失与反向传播**

对每个位置：softmax 把 logits 变成概率，取**正确标签那个词的概率** `p`，
损失 = `-log(p)`。

以 pos0 为例：`c` 的 logit 是 3.5，softmax 后概率 ≈ 0.94，`-log(0.94) ≈ 0.06`
（很小，因为答对了且很自信）。

```
总 loss = 四个位置的 -log(p) 取平均
```

- **没训练时**：logits 是随机的，每个词概率约 1/6，`-log(1/6) ≈ 1.79`，loss 很高。
- **训练就是**：反向传播算出每个权重矩阵（W_Q、W_K、embedding、lm_head……）该往哪
  调一点，让正确词的概率更高。AdamW 更新所有矩阵，重复几百轮 → 交叉注意力逐渐
  sharpen 成反对角线 → loss 降到接近 0。

### 9.2 推理（自回归，一个 token 一个 token 长出来）

训练时把整个答案 `[bos,c,b,a]` 一次性喂进去并行算（teacher forcing）；
推理时**没有答案可抄**，只能从 `<bos>` 开始，每次只生成一个：

**先编码一次**：`src="abc"` → encoder → `enc_out (3,4)`，之后反复用。

然后循环（对照 `model.py:306` 的 `generate`）：

```
步骤   已生成 ys          decode 取末位 logits   argmax   新 ys             说明
─────────────────────────────────────────────────────────────────────────────
 0     [bos]              [..., 3.5]→c          c(5)     [bos, c]          第一个输出
 1     [bos, c]           [..., 3.2]→b          b(4)     [bos, c, b]
 2     [bos, c, b]        [..., 3.4]→a          a(3)     [bos, c, b, a]
 3     [bos, c, b, a]     [..., 3.6]→eos        eos(2)   遇到 eos，停止
─────────────────────────────────────────────────────────────────────────────
输出 = 去掉 bos = [c, b, a] = "cba"  ✓
```

每一步：把当前 `ys` 送进 decoder → **只看最后一个位置的 logits**（只有它预测
"下一个"）→ 选出 token → 追加到 `ys` → 直到吐出 `<eos>`。

### 9.3 训练和推理为什么能一致？

训练时位置 1（输入 `c`）因为**因果掩码**，本来就只看得到 `[bos, c]`、看不到后面的
`[b, a]`——这跟推理时位置 1 手上只有 `[bos, c]` 是**完全相同的处境**。

所以训练时并行算出的"位置 1 预测 b"，和推理时逐步生成到位置 1 预测 b，用的是同一套
权重、同一种可见范围。**因果掩码就是保证这两件事等价的机关**——去掉它（README
进阶实验 1），训练会"抄答案"loss 降得飞快，但推理时没有答案可抄，立刻露馅。

### 9.4 一页小结

```
训练（并行）：
  src[3,4,5] ─embed+PE→ encoder ─→ enc_out
                                        ╲
  tgt[1,5,4,3] ─embed+PE→ decoder ←──────╳ 交叉注意力（反对角线=反转规则）
                              │
                          lm_head → logits(4×6) → 和 labels[5,4,3,2] 算 loss → 反向传播

推理（逐步）：
  src 编码一次 → enc_out
  [bos] → 预测 c → [bos,c] → 预测 b → [bos,c,b] → 预测 a → [...] → 预测 eos → 停
  输出 "cba"
```

### 9.5 真实矩阵（用 demo_matrices.py 跑出来的）

上面 9.1~9.4 是示意数字。`demo_matrices.py` 复用本目录的 `model.py`，搭了一个
迷你版（`d_model=16, 2 头, 2 层`，约 1.1 万参数），在随机字符串上真训练反转任务，
并用 forward hook 抓取**真实**注意力矩阵。运行：

```bash
python demo_matrices.py
```

**核心看点：交叉注意力从噪声 → 反对角线。**

训练前（接近均匀，模型"哪都看一点"）：

```
        a     b     c     d     e
<bos> [ 0.15  0.16  0.24  0.23  0.22 ]    一片噪声
  e   [ 0.23  0.12  0.31  0.13  0.21 ]
  d   [ 0.24  0.20  0.35  0.10  0.12 ]
  ...
```

训练 2500 步后（loss 0.89 → 0.09），头 1 几乎完美对齐成反对角线：

```
        a     b     c     d     e
<bos> [ 0.00  0.00  0.00  0.01  0.99 ]  █  预测 e → 死盯 src 的 e（最右）
  e   [ 0.00  0.00  0.01  0.78  0.21 ]  ▓  预测 d → 盯 d
  d   [ 0.00  0.02  0.83  0.15  0.01 ]  █  预测 c → 盯 c
  c   [ 0.03  0.86  0.11  0.00  0.00 ]  █  预测 b → 盯 b
  b   [ 0.94  0.06  0.00  0.00  0.00 ]  █  预测 a → 盯 a（最左）
  a   [ 1.00  0.00  0.00  0.00  0.00 ]  █
```

高亮块从右上（0.99）滑到左下（1.00）——这就是"第 i 个输出对齐第 L-1-i 个输入"
这条反转规则在权重里的物理形态。两个头不约而同收敛到了几乎相同的解。

**真实 logits（训练后，src='abcde'）**，6 个位置 argmax 全对，正确词分数远高于其他：

```
位置  输入→标签    ...  a    b    c    d    e   argmax
 0    <bos>→e     ... -1.3 -2.5 -2.3 -0.6  5.0   e ✓
 1    e→d         ...  1.1  0.4  0.5  7.0  1.5   d ✓
 2    d→c         ... -0.8 -0.6  5.6 -0.1 -1.9   c ✓
 3    c→b         ...  1.3  7.0  0.9  0.6  0.5   b ✓
 4    b→a         ...  6.0 -0.3 -0.3 -0.5 -0.5   a ✓
 5    a→<eos>     ...  2.9  0.2 -0.3 -0.3 -1.0   <eos> ✓
```

**真实自回归轨迹（src='abc'）**，每步 logits 最大值依次 c(5.3)→b(7.0)→a(6.1)→eos(10.6)：

```
步骤  已生成 ys    → 选出
 0    <bos>        → c
 1    <bos>c       → b
 2    <bos>cb      → a
 3    <bos>cba     → <eos>   输出 'cba' ✓
```

**一个有教育意义的失败**：泛化测试里 `'ab' -> 'baa'` 错了。原因是训练数据长度
范围是 3~6（脚本 `randint(3, 6)`），长度 2 的串从未出现，属于分布外输入，模型在
结尾犹豫、多吐了一个 `a`。这和 README 坑 4（OOV 字符）同源——**模型只会处理训练
分布内的输入**。把长度范围改成 `randint(2, 6)` 重训即可修复。
