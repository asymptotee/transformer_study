# 阶段三：decoder-only GPT（语言模型）

## 核心洞察

> **GPT 就是你已经写好的 decoder 那一半。**

把阶段一的 `Transformer` 去掉 encoder 和交叉注意力，只保留"因果自注意力 + FFN"，
训练目标改成"纯下一个 token 预测"——这就是 GPT。所以本阶段的模型大量**复用
阶段一的零件**（`MultiHeadAttention`、`FeedForward`、`PositionalEncoding`），
新写的只有 `GPTBlock`（= DecoderBlock 去掉交叉注意力）和 `GPT`（组装 + LM 前向）。

```
阶段一 Transformer:   src→Encoder ─┐
                    tgt→Decoder(因果自注意力+交叉注意力+FFN)→logits    （seq2seq）
阶段三 GPT:           ids→(因果自注意力+FFN)×N→logits                 （语言模型）
```

## 从 seq2seq 到语言模型：范式转变

这是本阶段最重要的概念升级：

| | 阶段一 (seq2seq) | 阶段三 (GPT/语言模型) |
|---|---|---|
| 结构 | encoder + decoder + 交叉注意力 | 只有因果 decoder |
| 目标 | 输入 src → 输出 tgt | **纯下一个 token 预测** |
| 数据 | (src, tgt) 对 | 一段文本，**输入 = 输出错一位** |
| 损失 | 只算 tgt 位置 | 每个位置都算"预测下一个" |
| 生成 | 输入→输出 | **prompt 续写** |
| 范式 | 序列到序列的映射 | **建模序列的概率分布**（LLM 的本质） |

训练代码的关键差异（`train_lm.py`）：

```python
# 语言模型：输入和目标就是同一段文本错一位
x = text[i : i+T]        # 输入
y = text[i+1 : i+T+1]    # 目标 = 输入右移一位
loss = cross_entropy(model(x), y)   # 每个位置都预测下一个 token
```

没有 src/tgt 之分，没有 encoder，整段文本拼成一条流、随机切窗口训练（nanoGPT 做法）。
**"预测下一个词"这个简单目标，正是 `CONCEPTS.md` 第六节说的 LLM 范式**——放大
万亿倍就是你今天用的每个大模型。

## 运行

```bash
cd stage3_gpt
PY=python

$PY make_corpus.py        # 生成伪诗语料 corpus.txt（约 12000 行）
$PY train_lm.py           # 训练（CPU 约 2~3 分钟）
$PY generate_text.py --prompt "山峰" --temperature 0.8    # 续写
$PY generate_text.py --prompt $'\n' --max-new-tokens 60   # 另起几行新"诗"
```

**用真实文本**（效果更地道）：

```bash
$PY train_lm.py --data /path/to/你的中文.txt
```

代码是字符级、语料无关的，任何中文文本都能训。

## 真实结果

**配置**：语料 12000 行 / 83,820 字符 / 词表 101；模型 54 万参数
（d_model=128, 4 层, 4 头）；训练 2000 步（batch 32, block 64, lr 3e-4）。

**loss 曲线**：

```
step    1 | train 7.72 | val 7.82     ← 起始（≈ln(词表) 的随机水平）
step  200 | train 2.74 | val 2.68     ← 快速学会结构
step 2000 | train 2.23 | val 2.21     ← 趋平
```

**loss 为什么停在 2.2 而不是更低？这不是没训好，而是到了语料的熵下限。**
伪诗里主题内的字是**均匀随机**的（每个位置在 8 个同主题字里等概率选），
所以即使完美模型，每个位置也只能预测到"8 选 1"，不可消除的熵 = ln 8 ≈ 2.08。
模型学到 2.2 已经把**所有可学的结构**（主题一致、前景→后情、五言/七言、换行）
都学到了，剩下的纯粹是数据本身的随机性。

另一个好信号：**train_loss(2.23) ≈ val_loss(2.21)**，几乎不过拟合——
模型学到的是真结构，不是死背训练集（呼应 `CONCEPTS.md` 的记忆 vs 泛化）。

**生成样例**（模型续写，每行都主题一致）：

```
涛河溪江浩流流      ← 水主题（涛河溪江=前景，浩流=后情，七言）
杏荷荷桃香茂        ← 花主题
岫嶂岭峻峻          ← 山主题（五言）
霏霏霞散淡          ← 云主题
月月月圆晶          ← 从"月"续写，月主题
```

模型显然学会了：给定几个字，判断出主题，然后按"前景→后情"的节奏续出同主题的字，
并在合适处换行。这就是语言模型"建模序列分布"的具体体现。

## 文件

| 文件 | 内容 |
|---|---|
| `model_gpt.py` | GPT 模型（复用阶段一组件，去掉 encoder；含温度+top-k 生成） |
| `make_corpus.py` | 伪诗语料生成器（离线可复现的默认语料） |
| `train_lm.py` | next-token 训练（整段文本切窗口，nanoGPT 式） |
| `generate_text.py` | prompt 续写（温度 / top-k） |
| `corpus.txt` | 生成的训练语料 |
| `ckpt_gpt.pt` | 训好的检查点 |

## 局限与下一步

- **字符级 + 小语料**：真实 LLM 用 BPE 分词、海量文本。字符级序列长、效率低，
  但最适合看清原理。
- **合成语料**：伪诗结构规整，生成的"语文味"有限。换真实文本（`--data`）会地道得多。
- **下一步可做**：
  - **BPE 分词**：根治字符级的 OOV 和长序列问题（README 坑 4 的根本解）
  - **scaling 实验**：变模型大小和数据量，画 loss 的幂律曲线（scaling law）
  - **实现 KV cache**：给 `generate` 加速，实测 O(n²)→O(n)（`CONCEPTS.md` 第二节）
