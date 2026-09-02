# Transformer 学习实现

从零手写一个完整的 encoder-decoder Transformer（《Attention Is All You Need》结构），
用字符级中文小语料训练，目的是搞懂内部每一个零件。约 70 万参数，CPU 上几分钟训完。

## 文件结构

| 文件 | 内容 |
|---|---|
| `model.py` | 全部模型模块（核心，注释最密集，建议从这里读起） |
| `data.py` | 字符分词器、双任务数据管道 |
| `train.py` | 训练循环（teacher forcing）+ 评估指标 |
| `generate.py` | 自回归生成（贪心 / 温度采样） |
| `corpus.txt` | 小语料：唐诗 + 谚语，约 300 行 |

## 架构总览

```
        src（源序列）                     tgt（目标序列，右移一位）
            │                                   │
     ┌──────▼───────┐                    ┌──────▼───────┐
     │ Embed ×√d    │                    │ Embed ×√d    │  ← 两侧共享同一张 embedding 表
     │ + 位置编码    │                    │ + 位置编码    │
     └──────┬───────┘                    └──────┬───────┘
            │                                   │
     ┌──────▼───────┐ ×N                 ┌──────▼───────┐ ×N
     │ 自注意力      │                    │ 因果自注意力  │  ← 下三角掩码，禁止看未来
     │ + FFN        │                    │      ↓       │
     │ (残差+Norm)  │                    │ 交叉注意力 ──┼─── 接收 encoder 输出
     └──────┬───────┘                    │      ↓       │
            │ enc_out ──────────────────▶│ FFN          │
            │                            │ (残差+Norm)  │
            │                            └──────┬───────┘
            │                                   │
            │                            ┌──────▼───────┐
            └───────────────────────────▶│ Linear head  │  ← 与 embedding 权重绑定
                                         └──────┬───────┘
                                                ▼
                                        logits（词表分布）
```

## 核心概念速查

**注意力公式**（`scaled_dot_product_attention`）：

    Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V

- Q·Kᵀ：query 和每个 key 算相似度（点积）
- ÷√d_k：维度越大点积方差越大，不缩放会让 softmax 饱和、梯度消失
- softmax：归一成概率权重
- ·V：按权重对 value 加权求和，"汇总相关位置的信息"

**三种注意力都是同一个类**（`MultiHeadAttention`），区别只在输入：

| 用途 | q 来自 | k/v 来自 | 掩码 |
|---|---|---|---|
| encoder 自注意力 | 源序列 | 源序列 | 填充掩码 |
| decoder 因果自注意力 | 目标序列 | 目标序列 | 填充 ∩ 因果（下三角） |
| 交叉注意力 | 目标序列 | encoder 输出 | 填充掩码 |

**两类掩码**：
- 填充掩码：屏蔽 `<pad>`，让 batch 内长短序列可以拼在一起算
- 因果掩码：下三角阵，保证位置 t 只能看见 ≤t 的位置。没有它，decoder 训练时
  会直接抄到答案，自回归生成时立刻露馅。实现上把禁止位置的分数填成 `-inf`，
  softmax 后权重即为 0

**位置编码**：注意力对顺序不敏感（打乱输入输出不变），必须显式注入位置。
本实现用论文的 sin/cos 固定编码，好处是相对位置可表示为线性关系。

**残差 + LayerNorm**：每个子层写成 `x = x + Sublayer(Norm(x))`（pre-norm），
残差路径保证梯度直通，深层才训得动。

**teacher forcing**：训练时把真实目标序列整体喂给 decoder，所有位置并行算损失；
推理时没有答案可抄，只能一个 token 一个 token 自回归生成（见 `generate`）。

## 运行

使用本目录外的 venv（已装 PyTorch）：

```bash
cd transformer_study
PY=python

# 0. 冒烟测试：单 batch 过拟合，loss 应降到 ~0（先证明管道没 bug）
$PY train.py --task reverse --smoke

# 1. 训练反转任务：val_exact（整句精确匹配）可到 0.8~0.9+
#    （训练集默认混入 4000 条随机字符串增强，原因见下文"坑 2"）
$PY train.py --task reverse --epochs 30

# 2. 试生成
echo "一二三四五" | $PY generate.py          # 期望: 五四三二一
echo "床前明月光" | $PY generate.py          # 期望: 光月明前床

# 3. 训练续写任务并试用（靠记忆语料，按 train_loss 挑 checkpoint，见"坑 3"）
$PY train.py --task complete --epochs 80 --select-by train_loss --ckpt ckpt_complete.pt
echo "床前" | $PY generate.py --ckpt ckpt_complete.pt   # 期望: 明月光
```

注意：两个任务的 checkpoint 要分开存（`--ckpt`），`generate.py` 会从 checkpoint
里读出任务类型。

## 建议的学习路径

1. 通读 `model.py`，对照上面的架构图走一遍数据流
2. 跑冒烟测试，确认 loss 能降到 ~0
3. 训练 reverse，观察 `val_exact` 从 0 爬到 0.9 附近的过程
4. 用 `generate.py` 玩一玩，注意 greedy 和 `--temperature 0.8` 的区别

## 踩过的坑（都是真实调试记录）

**坑 1：初始化不对，训练根本启动不了。**
PyTorch 的 embedding 默认初始化为 N(0,1)，再乘上 √d_model 缩放后量级约 11，
注意力分数随之爆炸，softmax 饱和成 one-hot，梯度几乎为零 ——
冒烟测试的 loss 从 118 开始，300 步只降到 6.6（还是随机水平）。
修复：embedding 用 std = 1/√d_model 初始化（论文官方实现 tensor2tensor 的技巧），
缩放后量级回到 1，和位置编码平衡，loss 立刻正常下降。见 `model.py` 的 `_init_weights`。

**坑 2：reverse 在真实语料上"学会"了，但验证集全错。**
300 行语料对 75 万参数的模型来说背下来毫不费力，于是它走了捷径：
逐行记忆，而不是学习"第 i 个输出看第 L-1-i 个输入"的对齐规则。
train_loss 降到 0.25，val_exact 却是 0。修复：用字符集随机生成 4000 条字符串
混入训练集（`data.random_lines`）——空间大到背不下来，模型被迫学算法，
val_exact 随后爬到 0.9+。验证集保持真实语料，才是诚实的泛化测试。

**坑 3：给 complete 按 val_loss 挑 checkpoint，挑出了最差的模型。**
complete（给前半句续后半句）没有可学的通用算法，本质是记忆语料，
验证集上的陌生句子永远答不对，val_loss 越早（越没记住）反而越低。
修复：加 `--select-by train_loss`，会泛化的任务看验证、靠记忆的任务看训练 ——
**checkpoint 的挑选标准取决于任务性质**。

**坑 4：输入里出现词表外的字，输出就乱了。**
测试 `一二三四五` 时得到 `杨杨杨四三二一`：原语料里根本没有"五"这个字，
它被编码成 `<unk>`，而模型训练时从没见过 `<unk>`，对应位置只能吐乱码。
修复：给语料补了《一去二三里》（顺带补齐缺失的数字字符）。
启示：词表覆盖率决定模型能处理什么输入，真实系统用 BPE 这类永不 OOV 的分词。

## 进阶实验

- **去掉因果掩码**：把 `decode` 里的 `causal_mask` 去掉，训练 loss 会降得更快
  （因为能抄答案），但生成输出会崩坏 —— 体会掩码的作用
- **改头数**：`n_heads=1` vs `4` vs `8`（保持 d_model 不变），看收敛速度
- **看注意力权重**：`MultiHeadAttention.forward` 已返回 `attn`，把它存下来用
  matplotlib 画热力图，reverse 任务应该能看到清晰的对角线（位置 i 关注 L-1-i）
- **换 post-norm**：把 block 改成 `norm(x + sublayer(x))`，对比训练稳定性
- **加长语料/加大模型**：观察过拟合迹象，再试着调 dropout
