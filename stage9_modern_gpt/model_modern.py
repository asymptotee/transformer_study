"""model_modern.py —— stage9 现代化模型(9.1 Norm/FFN,9.2 RoPE,9.3 KV cache+GQA)

骨架逐字来自 stage3 model_gpt.py(它再复用了 stage1 的注意力与位置编码),
**默认开关下与旧版逐字节同构**——每次换零件前先跑 check_isomorphic.py
证明"默认路径没被碰坏",再做 A/B,差异才能只归因于被换的零件。

  零件开关(对照 minimind `model/model_minimind.py`):
    cfg.norm = "layernorm" | "rms"      RMSNorm(9.1)
    cfg.ff   = "gelu" | "silu"          SwiGLU(9.1)
    cfg.pos  = "sinusoidal" | "rope"    RoPE(9.2);rope 路径下 max_len 退役,
                                        用 rope_ctx(预计算表长)与 rope_theta
    cfg.n_kv_heads                      GQA(9.3):注意力里 K/V 头数,
                                        None = 与 Q 同头数(MHA)
    9.3 的 KV cache 只落在 rope 路径:GPT.forward(ids) 保持原行为(整窗前向),
    GPT.forward_cached(ids, past_kv) 走增量前向(逐层存/取 K/V,返回新 past)。
    正弦路径不改 → 同构自检继续覆盖默认分支。

ckpt 格式不变:{"model": state_dict, "config": vars(cfg), "step": N}
注意:92_rope(9.2)的 ckpt 在 n_kv_heads=None 时仍可直接加载(参数字典一致)。
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
    n_heads: int = 4       # 注意力 Q 头数
    n_layers: int = 4      # GPT 层数
    d_ff: int = 256        # 前馈隐藏层维度
    dropout: float = 0.1
    max_len: int = 256     # 位置编码上限(sinusoidal 用;rope 路径下退役)
    pad_id: int = 0
    # ---- 9.1 零件开关(默认 = 旧架构,保证同构) ----
    norm: str = "layernorm"    # layernorm | rms
    ff: str = "gelu"           # gelu | silu(=SwiGLU)
    # ---- 9.2 位置编码 ----
    pos: str = "sinusoidal"    # sinusoidal | rope
    rope_theta: float = 1e6    # RoPE 基频底数(同 minimind 默认)
    rope_ctx: int = 4096       # RoPE 预计算表长(外推测试上限,非训练长度)
    # ---- 9.3 注意力 ----
    n_kv_heads: int = None     # GQA 的 K/V 头数;None = 与 Q 同头数(MHA)


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


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA:把 n_kv 个头的 K/V 复制成 n_heads 份(同 minimind model_minimind.py:86,
    只是它的布局是 (B, T, H, D),本仓库是 (B, H, T, D))。n_rep=1 时原样返回。"""
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    return (x[:, :, None, :, :].expand(b, h, n_rep, t, d)
            .reshape(b, h * n_rep, t, d))


class RotaryAttention(nn.Module):
    """RoPE + GQA 版自注意力(9.2/9.3)。

    9.2:拆头后按绝对位置旋转 q/k(见 rope.py),参数布局与 stage1 MHA 一致
    9.3:w_k/w_v 只投影到 n_kv_heads×head_dim(GQA),算注意力前用 repeat_kv
        把 K/V 头复制回 Q 的头数;返回 (out, present),present = 本层新的
        (k, v)——KV cache 的原料。past_kv 给定时走增量前向:
        把新 token 的 k/v 拼到历史后,直接算注意力(增量步只有一个 query,
        天然因果,无需掩码)。

    布局约定(q/k/v 都是 (B, H, T, head_dim),与 minimind 的 (B, T, H, D)
    不同——9.2 已踩过 unsqueeze 维度的坑,详见 README)。
    """

    def __init__(self, d_model, n_heads, dropout=0.1, n_kv_heads=None):
        super().__init__()
        assert d_model % n_heads == 0
        if n_kv_heads is not None:
            assert n_heads % n_kv_heads == 0, "Q 头数必须能整除 K/V 头数"
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = d_model // n_heads
        kv_dim = self.head_dim * self.n_kv_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, kv_dim, bias=False)
        self.w_v = nn.Linear(d_model, kv_dim, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, cos, sin, past_kv=None):
        B, T, _ = x.shape
        q = self.w_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_kv is not None:                    # 增量步:拼上历史 K/V
            k = torch.cat([past_kv[0], k], dim=2)  # 沿时间维
            v = torch.cat([past_kv[1], v], dim=2)
        present = (k, v)                            # 新缓存(K/V 都是 kv 头数)
        k = repeat_kv(k, self.n_heads // self.n_kv_heads)
        v = repeat_kv(v, self.n_heads // self.n_kv_heads)
        out, _ = scaled_dot_product_attention(q, k, v, mask=mask,
                                              dropout=self.dropout)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.w_o(out), present


class GPTBlock(nn.Module):
    """与 stage3 GPTBlock 同结构,只按 cfg 选 Norm/FFN/注意力零件。"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.is_rope = cfg.pos == "rope"
        if self.is_rope:
            self.self_attn = RotaryAttention(cfg.d_model, cfg.n_heads,
                                             cfg.dropout, cfg.n_kv_heads)
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
        """整窗前向(训练/评估;rope 无缓存)。sinusoidal:cos 为 None 走原样。"""
        h = self.norm1(x)
        if self.is_rope:
            attn_out, _ = self.self_attn(h, mask, cos, sin)
        else:
            attn_out, _ = self.self_attn(h, h, h, mask=mask)
        x = x + attn_out
        return x + self.ff(self.norm2(x))

    def forward_cached(self, x, cos, sin, past_kv):
        """增量前向(9.3):只算新 token,返回 (新 hidden, 本层新 present)。
        只对 rope 路径定义;sinusoidal 不支持 cache。"""
        assert self.is_rope, "KV cache 只在 rope 路径实现"
        h = self.norm1(x)
        attn_out, present = self.self_attn(h, None, cos, sin, past_kv)
        x = x + attn_out
        return x + self.ff(self.norm2(x)), present


class GPT(nn.Module):
    """与 stage3 GPT 同构;pos=rope 时加旋转表与增量前向(forward_cached)。"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.is_rope = cfg.pos == "rope"
        self.pos_enc = None
        if self.is_rope:
            head_dim = cfg.d_model // cfg.n_heads
            cos, sin = precompute_freqs_cis(head_dim, cfg.rope_ctx, cfg.rope_theta)
            # 非持久 buffer:不入 state_dict(表由 cfg 随时可重算,如推理期换 YaRN)
            self.register_buffer("freqs_cos", cos, persistent=False)
            self.register_buffer("freqs_sin", sin, persistent=False)
        else:
            self.pos_enc = PositionalEncoding(cfg.d_model, cfg.max_len, cfg.dropout)
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
        """整窗前向(与 stage3 同接口):ids (B,T) → logits (B,T,vocab)。"""
        return self._run(ids, None)[0]

    @torch.no_grad()
    def forward_cached(self, ids, past_kv):
        """增量前向(9.3,只 rope):ids 为**下一个 token**(B,1)或 prefill 段。

        past_kv: 每层一个 (k, v),形状 (B, n_kv_heads, T_old, head_dim);
                 None 表示 prefill(整段前向,返回全部层的初始缓存)。
        返回 (logits, new_past_kv)。
        """
        return self._run(ids, past_kv)

    def _run(self, ids, past_kv):
        T = ids.size(1)
        x = self.embed(ids)
        if not self.is_rope:
            assert past_kv is None, "sinusoidal 路径不支持 cache"
            mask = self.causal_mask(T, ids.device)
            for block in self.blocks:
                x = block(x, mask)
            return self.lm_head(self.norm(x)), None

        # rope 路径
        off = past_kv[0][0].shape[2] if past_kv else 0    # 已缓存的时间长度
        cos = self.freqs_cos[off: off + T]
        sin = self.freqs_sin[off: off + T]
        if past_kv is None:
            mask = self.causal_mask(T, ids.device)          # prefill:整段因果
            presents = [None] * len(self.blocks)
            for i, block in enumerate(self.blocks):
                x, presents[i] = self._block_forward(block, x, mask, cos, sin)
            return self.lm_head(self.norm(x)), presents
        else:
            assert T == 1, "增量步一次只喂一个 token(否则需拼接偏移的因果掩码)"
            presents = []
            for i, block in enumerate(self.blocks):
                x, p = block.forward_cached(x, cos, sin, past_kv[i])
                presents.append(p)
            return self.lm_head(self.norm(x)), presents

    def _block_forward(self, block, x, mask, cos, sin):
        """rope 路径整窗前向也要收集 present(prefill 造缓存)。

        残差顺序必须与 block.forward 逐字一致(pre-norm):
        norm2 的输入是**加完注意力之后**的 x——9.3 曾把这里写成
        `x + attn_out + ff(norm2(x))`,Python 先求值 norm2(旧 x),
        导致 prefill 路径与 decode 路径语义不一致(一致性自检抓出)。
        """
        h = block.norm1(x)
        attn_out, present = block.self_attn(h, mask, cos, sin)
        x = x + attn_out
        return x + block.ff(block.norm2(x)), present

    @torch.no_grad()
    def generate(self, ids, max_new_tokens, temperature=0.8, top_k=None):
        """无缓存的简单生成(与 stage3 同款,兼容旧调用;性能见 generate_cached)。"""
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
