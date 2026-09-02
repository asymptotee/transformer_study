"""一个简单的 Transformer 实现（encoder-decoder 结构）

从零复现论文《Attention Is All You Need》的核心架构：

    encoder: N × (多头自注意力 + 前馈网络)，带残差连接和 LayerNorm
    decoder: N × (因果自注意力 + 交叉注意力 + 前馈网络)

出于学习目的，代码刻意不使用 torch.nn.Transformer 这类高层封装，
注意力计算、掩码构建、位置编码全部手写，并在关键处解释"为什么"。

两个与原始论文不同、但更现代的设计选择（已在对应位置注释说明）：
- pre-norm（子层之前做 LayerNorm），训练更稳定，无需学习率 warmup
- 输出投影层与 embedding 权重绑定（weight tying），省参数、助泛化
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    """模型超参数。默认值刻意设得很小，CPU 上几分钟就能训练完。"""

    vocab_size: int       # 词表大小（由语料的字符集决定）
    d_model: int = 128    # 模型宽度：每个 token 的向量维度
    n_heads: int = 4      # 注意力头数（要求 d_model 能被 n_heads 整除）
    n_layers: int = 2     # encoder 和 decoder 各自的层数
    d_ff: int = 256       # 前馈网络隐藏层维度（惯例为 2~4 倍 d_model）
    dropout: float = 0.1
    max_len: int = 64     # 支持的最大序列长度（位置编码的上限）
    pad_id: int = 0       # padding token 的索引，用于构建填充掩码


def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    """缩放点积注意力：softmax(Q·Kᵀ / √d_k) · V

    直觉：每个 query 和所有 key 算相似度（点积），softmax 归一成权重，
    再对 value 加权求和 —— "把注意力分配到相关位置上并汇总信息"。

    Args:
        q, k, v: (batch, n_heads, seq_len, d_k)
        mask:    bool 掩码，True 的位置可以互相看见，False 的位置会被屏蔽
    Returns:
        out:  (batch, n_heads, seq_len, d_k)
        attn: 注意力权重 (batch, n_heads, query_len, key_len)，可用于可视化
    """
    d_k = q.size(-1)
    # Q·Kᵀ: (B, H, T_q, d_k) @ (B, H, d_k, T_k) -> (B, H, T_q, T_k)
    # 除以 √d_k：维度越大点积的方差越大，会把 softmax 推进饱和区使梯度消失
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        # 为什么用 -inf：softmax 后 e^(-inf) = 0，等于让这些位置的权重彻底为零。
        # 常见两类掩码：
        #   填充掩码 (padding mask)：屏蔽 <pad> 位置，形状 (B,1,1,T_k)，广播到所有头
        #   因果掩码 (causal mask)：下三角阵，保证位置 t 只能看见 ≤t 的位置
        scores = scores.masked_fill(~mask, float("-inf"))

    attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    return attn @ v, attn


class MultiHeadAttention(nn.Module):
    """多头注意力

    把 d_model 拆成 n_heads 个 d_k = d_model/n_heads 的子空间，各自独立算注意力
    再拼接投影回去。不同的头可以关注不同类型的关系（相邻位置、对称结构、
    语义呼应……），相当于用多个"视角"看同一个序列。

    这一个类同时承担三种用途，仅靠传入的数据和掩码区分：
      自注意力 (self-attention)：  q = k = v = 序列自身
      因果自注意力：              同上，但额外传入下三角因果掩码
      交叉注意力 (cross-attention)：q 来自 decoder，k/v 来自 encoder 输出
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        # Q/K/V 各用一个线性层投影；bias=False 是现代 LLM 的常见做法（原论文带 bias）
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        # 投影后拆头：(B, T, d_model) -> (B, T, H, d_k) -> (B, H, T, d_k)
        # 把头的维度挪到前面，所有头就能用一次批量矩阵乘法并行计算
        q = self.w_q(q).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        out, attn = scaled_dot_product_attention(q, k, v, mask=mask, dropout=self.dropout)

        # 合并多头：(B, H, T, d_k) -> (B, T, H, d_k) -> (B, T, d_model)
        # transpose 之后内存不连续，view 之前必须 contiguous()
        out = out.transpose(1, 2).contiguous().view(B, -1, self.n_heads * self.d_k)
        return self.w_o(out), attn


class PositionalEncoding(nn.Module):
    """正弦位置编码（论文公式 1、2）

    注意力计算本身对顺序不敏感（打乱输入结果不变），必须额外注入位置信息。
    公式：PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
          PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    好处：任意固定偏移 k 的 PE(pos+k) 可表示为 PE(pos) 的线性函数，
    模型容易学到"相对位置"关系；且可外推到训练时没见过的长度。
    """

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
        # 每个维度对用一个不同频率：exp(-2i·ln(10000)/d_model) = 1/10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维用 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维用 cos
        # register_buffer：不是可学习参数，但会随模型一起移动设备、存进 checkpoint
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class FeedForward(nn.Module):
    """逐位置前馈网络：FFN(x) = Linear(GELU(Linear(x)))

    注意力层只做加权求和（本质是线性的），非线性变换能力全靠这一层。
    它对每个位置独立施加同样的两层 MLP，相当于"逐 token 的特征加工"。
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    """encoder 单层：自注意力 + 前馈，各配一个残差连接和 LayerNorm

    采用 pre-norm 写法：x = x + Sublayer(LayerNorm(x))
    原论文是 post-norm：x = LayerNorm(x + Sublayer(x))。
    区别：pre-norm 残差路径是"干净"的恒等映射，梯度回传更稳，
    不需要学习率 warmup 就能训练，是目前的主流做法。
    """

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, src_mask):
        h = self.norm1(x)
        attn_out, _ = self.self_attn(h, h, h, mask=src_mask)  # q=k=v=x：自注意力
        x = x + attn_out                                      # 残差
        x = x + self.ff(self.norm2(x))
        return x


class DecoderBlock(nn.Module):
    """decoder 单层：因果自注意力 + 交叉注意力 + 前馈

    三个子层，比 encoder 多一个交叉注意力：
    1. 因果自注意力：带下三角掩码，位置 t 只能看已生成的 ≤t 的位置，
       保证训练和推理（逐 token 生成）的行为一致
    2. 交叉注意力：q 来自 decoder，k/v 来自 encoder 输出 ——
       decoder 借此"回头看"源序列，这是 seq2seq 信息流动的关键
    3. 前馈网络：同 encoder
    """

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, tgt_mask, src_mask):
        h = self.norm1(x)
        attn_out, _ = self.self_attn(h, h, h, mask=tgt_mask)
        x = x + attn_out
        h = self.norm2(x)
        # 交叉注意力：q 来自 decoder 侧，k/v 来自 encoder 输出
        # （encoder 末尾已有 LayerNorm，这里 enc_out 不再归一化）
        cross_out, _ = self.cross_attn(h, enc_out, enc_out, mask=src_mask)
        x = x + cross_out
        x = x + self.ff(self.norm3(x))
        return x


class Transformer(nn.Module):
    """完整的 encoder-decoder Transformer

    数据流：
        src ─▶ Embed+PE ─▶ Encoder×N ─┐
                                      ├─▶ enc_out
        tgt ─▶ Embed+PE ─▶ Decoder×N ─┘─▶ Linear ─▶ logits（词表上的分布）

    src 和 tgt 共用同一套字符词表，因此共享一张 embedding 表；
    输出层 Linear 也与 embedding 共享权重（weight tying）。
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_enc = PositionalEncoding(cfg.d_model, cfg.max_len, cfg.dropout)
        self.encoder = nn.ModuleList(
            EncoderBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layers)
        )
        self.decoder = nn.ModuleList(
            DecoderBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layers)
        )
        # pre-norm 结构的收尾：残差路径上的 x 从未归一化，输出前补一次
        self.enc_norm = nn.LayerNorm(cfg.d_model)
        self.dec_norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # 权重绑定：输出投影和 embedding 是同一张表。
        # 直觉："把 token id 映射成向量"和"把向量映射回 token id"本该互逆
        self.lm_head.weight = self.embedding.weight
        self._init_weights(cfg)

    def _init_weights(self, cfg):
        """初始化是 Transformer 训练稳定性的关键细节之一。

        embedding 用 std = 1/√d_model 的正态分布（论文官方实现 tensor2tensor 的技巧）：
        乘上 √d_model 缩放后量级回到 1 左右，和位置编码（±1）平衡。
        若用 PyTorch 默认的 N(0,1) 初始化，缩放后 embedding 量级约 11，
        注意力分数会大到让 softmax 饱和成 one-hot，梯度几乎为零，训练极难启动。
        """
        nn.init.normal_(self.embedding.weight, mean=0.0, std=cfg.d_model ** -0.5)
        with torch.no_grad():
            # normal_ 会覆盖 padding_idx 的零初始化约定，手动补回
            self.embedding.weight[cfg.pad_id].fill_(0.0)

    # ---------- 掩码 ----------

    def pad_mask(self, ids):
        """填充掩码：真实 token 为 True，<pad> 为 False。
        形状 (B,1,1,T)，可广播到注意力分数的 (B,H,T_q,T_k) 上。"""
        return (ids != self.cfg.pad_id).view(ids.size(0), 1, 1, ids.size(1))

    @staticmethod
    def causal_mask(size, device):
        """因果掩码（下三角阵）：位置 t 只能看见 ≤t 的位置。
        没有它，decoder 训练时会直接"抄"到未来的答案，生成时就露馅了。"""
        return torch.tril(torch.ones(size, size, dtype=torch.bool, device=device)).view(1, 1, size, size)

    # ---------- 前向 ----------

    def embed(self, ids):
        # 原论文把 embedding 乘以 √d_model：让它的量级和位置编码（±1 之间）匹配
        return self.pos_enc(self.embedding(ids) * math.sqrt(self.cfg.d_model))

    def encode(self, src):
        """编码源序列，返回 (encoder 输出, 填充掩码)。
        生成时 encoder 只需跑一次，结果可被反复使用，所以单独拆出来。"""
        src_mask = self.pad_mask(src)
        x = self.embed(src)
        for block in self.encoder:
            x = block(x, src_mask)
        return self.enc_norm(x), src_mask

    def decode(self, tgt, enc_out, src_mask):
        """解码目标序列，返回词表上的 logits。"""
        # decoder 自注意力的掩码 = 因果掩码 ∩ 填充掩码（逐元素与，靠广播对齐）
        tgt_mask = self.causal_mask(tgt.size(1), tgt.device) & self.pad_mask(tgt)
        x = self.embed(tgt)
        for block in self.decoder:
            x = block(x, enc_out, tgt_mask, src_mask)
        return self.lm_head(self.dec_norm(x))

    def forward(self, src, tgt):
        """训练入口：teacher forcing，把真实 tgt 整体喂给 decoder 并行算所有位置。"""
        enc_out, src_mask = self.encode(src)
        return self.decode(tgt, enc_out, src_mask)

    # ---------- 推理 ----------

    @torch.no_grad()
    def generate(self, src, bos_id, eos_id, max_new_tokens, temperature=0.0):
        """自回归生成：encoder 只编码一次，decoder 每次只"长出"一个 token

        每一步：把已生成序列送进 decoder → 取最后一个位置的 logits →
        选出下一个 token → 追加到序列 → 直到出现 <eos> 或达到长度上限。

        temperature <= 0 时贪心（取 argmax）；否则 logits 除以温度再采样，
        温度越高输出越随机多样。
        """
        enc_out, src_mask = self.encode(src)
        B = src.size(0)
        ys = torch.full((B, 1), bos_id, dtype=torch.long, device=src.device)
        for _ in range(max_new_tokens):
            logits = self.decode(ys, enc_out, src_mask)[:, -1]  # 只看最后一个位置
            if temperature <= 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            ys = torch.cat([ys, next_id], dim=1)
            if (next_id == eos_id).all():  # 所有样本都生成了结束符
                break
        return ys[:, 1:]  # 去掉开头的 <bos>，只返回生成内容
