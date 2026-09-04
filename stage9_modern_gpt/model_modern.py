"""model_modern.py —— stage9 现代化模型(9.1: Norm/FFN;9.2: RoPE)

骨架逐字来自 stage3 model_gpt.py(它再复用了 stage1 的注意力与位置编码),
**默认开关下与旧版逐字节同构**——每次换零件前先跑 check_isomorphic.py
证明"默认路径没被碰坏",再做 A/B,差异才能只归因于被换的零件。

  零件开关(对照 minimind `model/model_minimind.py`):
    cfg.norm = "layernorm" | "rms"      RMSNorm(9.1):去掉均值中心化
    cfg.ff   = "gelu" | "silu"          SwiGLU(9.1):门控 FFN
    cfg.pos  = "sinusoidal" | "rope"    RoPE(9.2):Q/K 旋转位置编码
                                        rope 路径下 cfg.max_len(正弦表长)退役,
                                        改用 cfg.rope_ctx 预计算表长(可远超训练
                                        长度,外推测试直接查表);rope_theta 同
                                        minimind 默认 1e6;推理期可另配 YaRN
                                        (rope.py 的 rope_scaling,改表不改权重)

9.3 起 GQA / KV cache 继续在这个文件里演进;stage3 模型保持不动,它是
stage7/8 与所有对照实验的参照物。ckpt 格式不变:
{"model": state_dict, "config": vars(cfg), "step": N}——config 自带零件开关。
"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "stage1_basics"))
from model import (FeedForward, MultiHeadAttention, PositionalEncoding,  # noqa: E402
                   scaled_dot_product_attention)
from rope import apply_rotary_pos_emb, precompute_freqs_cis               # noqa: E402


@dataclass
class GPTConfig:
    vocab_size: int        # 词表大小(由语料字符集决定)
    d_model: int = 128     # 模型宽度
    n_heads: int = 4       # 注意力头数
    n_layers: int = 4      # GPT 层数
    d_ff: int = 256        # 前馈隐藏层维度
    dropout: float = 0.1
    max_len: int = 256     # 位置编码上限(sinusoidal 用;rope 路径下退役)
    pad_id: int = 0
    # ---- 9.1 零件开关(默认 = 旧架构,保证同构) ----
    norm: str = "layernorm"    # layernorm | rms
    ff: str = "gelu"           # gelu | silu(=SwiGLU)
    # ---- 9.2 位置编码开关 ----
    pos: str = "sinusoidal"    # sinusoidal | rope
    rope_theta: float = 1e6    # RoPE 基频底数(同 minimind 默认)
    rope_ctx: int = 4096       # RoPE 预计算表长(外推测试上限,非训练长度)


class RMSNorm(nn.Module):
    """RMSNorm: x̂ = x / RMS(x),再乘可学习缩放 weight。(9.1,见 README)"""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)


class SwiGLU(nn.Module):
    """门控 FFN:down(silu(gate(x)) ⊙ up(x))。(9.1,见 README)"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        g = F.silu(self.gate(x))
        return self.dropout(self.down(self.dropout(g * self.up(x))))


class RotaryAttention(nn.Module):
    """RoPE 版自注意力(9.2):结构与 stage1 MultiHeadAttention 完全一致
    (w_q/k/v/o 四个无 bias 线性层、同样的拆分/合并/dropout),唯一区别是
    拆头之后、算注意力之前,把 q、k 按绝对位置旋转(见 rope.py)。

    参数布局与 MHA 相同 → rope 臂与 sinusoidal 臂参数**严格相等**,
    这是 9.1 的 SwiGLU A/B 没有的待遇(那次的差异还带着 +29% 参数)。
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, cos, sin):
        B = x.size(0)
        q = self.w_q(x).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos[:x.size(1)], sin[:x.size(1)])
        out, attn = scaled_dot_product_attention(q, k, v, mask=mask,
                                                 dropout=self.dropout)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.n_heads * self.d_k)
        return self.w_o(out), attn


class GPTBlock(nn.Module):
    """与 stage3 GPTBlock 同结构,只按 cfg 选 Norm/FFN/注意力零件。"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        if cfg.pos == "rope":
            self.self_attn = RotaryAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        else:
            self.self_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        if cfg.ff == "gelu":
            self.ff = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
        else:
            self.ff = SwiGLU(cfg.d_model, cfg.d_ff, cfg.dropout)
        Norm = RMSNorm if cfg.norm == "rms" else nn.LayerNorm
        self.norm1 = Norm(cfg.d_model)
        self.norm2 = Norm(cfg.d_model)

    def forward(self, x, mask, cos=None, sin=None):
        h = self.norm1(x)
        if cos is not None:                      # rope 路径:旋转 q/k
            attn_out, _ = self.self_attn(h, mask, cos, sin)
        else:                                    # sinusoidal 路径:原样
            attn_out, _ = self.self_attn(h, h, h, mask=mask)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class GPT(nn.Module):
    """与 stage3 GPT 同构,pos=rope 时把正弦表换成 RoPE 频率表(无学习参数)。"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        if cfg.pos == "rope":
            head_dim = cfg.d_model // cfg.n_heads
            cos, sin = precompute_freqs_cis(head_dim, cfg.rope_ctx, cfg.rope_theta)
            # 非持久 buffer:不入 state_dict(表可由 cfg 随时重算,如推理期换 YaRN)
            self.register_buffer("freqs_cos", cos, persistent=False)
            self.register_buffer("freqs_sin", sin, persistent=False)
            self.pos_enc = None
        else:
            self.pos_enc = PositionalEncoding(cfg.d_model, cfg.max_len, cfg.dropout)
            self.freqs_cos = None                # 非 rope 路径:标记无旋转表
            self.freqs_sin = None
        self.blocks = nn.ModuleList(GPTBlock(cfg) for _ in range(cfg.n_layers))
        Norm = RMSNorm if cfg.norm == "rms" else nn.LayerNorm
        self.norm = Norm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self._init_weights(cfg)

    def _init_weights(self, cfg):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=cfg.d_model ** -0.5)
        with torch.no_grad():
            self.embedding.weight[cfg.pad_id].fill_(0.0)

    @staticmethod
    def causal_mask(size, device):
        return torch.tril(torch.ones(size, size, dtype=torch.bool, device=device)).view(1, 1, size, size)

    def embed(self, ids):
        x = self.embedding(ids) * math.sqrt(self.cfg.d_model)
        return self.pos_enc(x) if self.pos_enc is not None else x

    def forward(self, ids):
        mask = self.causal_mask(ids.size(1), ids.device)
        x = self.embed(ids)
        for block in self.blocks:
            if self.freqs_cos is not None:
                x = block(x, mask, self.freqs_cos, self.freqs_sin)
            else:
                x = block(x, mask)
        return self.lm_head(self.norm(x))

    @torch.no_grad()
    def generate(self, ids, max_new_tokens, temperature=0.8, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            ctx = ids if ids.size(1) <= self.cfg.max_len else ids[:, -self.cfg.max_len:]
            logits = self.forward(ctx)[:, -1]
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            if temperature <= 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, nxt], dim=1)
        return ids
