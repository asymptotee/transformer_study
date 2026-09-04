"""model_modern.py —— stage9 现代化模型(9.1 起步:Norm / FFN 可切换)

骨架逐字来自 stage3 model_gpt.py(它再复用了 stage1 的注意力与位置编码),
**默认开关下与旧版逐字节同构**——这样 A/B 只差被测的那个零件。

  9.1 加的开关(对照 minimind `model/model_minimind.py`):
    cfg.norm = "layernorm" | "rms"      RMSNorm:去掉均值中心化,只留缩放
                                        (LayerNorm 的 bias/mean 收益随宽度递减;
                                        RMSNorm 省参数、数值更稳——LLaMA 系标配)
    cfg.ff   = "gelu" | "silu"          silu = SwiGLU:gate/up/down 三投影,
                                        silu(gate(x))·up(x) 再做 down
                                        (门控让 FFN 有"按 token 选择性通过"
                                        的能力,现代 LLM 的标准 FFN)

9.2 起 RoPE / GQA / KV cache 继续在这个文件里演进;stage3 模型保持不动,
它是 stage7/8 与所有对照实验的参照物。

ckpt 格式与旧版一致:{"model": state_dict, "config": vars(cfg), "step": N}
vars(cfg) 会把 norm/ff 一起存进 config —— 换零件训练的 ckpt 自带说明。
"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "stage1_basics"))
from model import FeedForward, MultiHeadAttention, PositionalEncoding  # noqa: E402


@dataclass
class GPTConfig:
    vocab_size: int        # 词表大小(由语料字符集决定)
    d_model: int = 128     # 模型宽度
    n_heads: int = 4       # 注意力头数
    n_layers: int = 4      # GPT 层数
    d_ff: int = 256        # 前馈隐藏层维度
    dropout: float = 0.1
    max_len: int = 256     # 位置编码上限(9.2 换 RoPE 后退役)
    pad_id: int = 0
    # ---- 9.1 零件开关(默认 = 旧架构,保证同构) ----
    norm: str = "layernorm"    # layernorm | rms
    ff: str = "gelu"           # gelu | silu(=SwiGLU)


class RMSNorm(nn.Module):
    """RMSNorm: x̂ = x / RMS(x),再乘可学习缩放 weight。

    与 LayerNorm 的唯一差别:不做均值中心化(不减 mean、无 bias)。
    动机:残差流里均值信息已在别处表达,中心化收益随宽度增大而递减;
    去掉后省一组 bias 参数、少一次归约,数值更稳。LLaMA/Qwen 系标配。
    数值上 float32 内算再回原 dtype(minimind 同款写法,bf16 下更稳)。
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)


class SwiGLU(nn.Module):
    """SwiGLU FFN: down(silu(gate(x)) ⊙ up(x)),三个投影、无 bias。

    对照旧 FFN(Linear→GELU→Linear):多了个 gate 分支,gate(x) 是逐 token
    的"开关",silu 保证开关在 0 附近连续可导。效果 = FFN 能按输入选择
    只激活部分神经元。现代 LLM(LLaMA 3 等)的 FFN 标准形。
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        g = F.silu(self.gate(x))
        return self.dropout(self.down(self.dropout(g * self.up(x))))


class GPTBlock(nn.Module):
    """与 stage3 GPTBlock 同结构,只按 cfg 选 Norm 与 FFN 零件。"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        if cfg.ff == "gelu":
            self.ff = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
        else:
            self.ff = SwiGLU(cfg.d_model, cfg.d_ff, cfg.dropout)
        Norm = RMSNorm if cfg.norm == "rms" else nn.LayerNorm
        self.norm1 = Norm(cfg.d_model)
        self.norm2 = Norm(cfg.d_model)

    def forward(self, x, mask):
        h = self.norm1(x)
        attn_out, _ = self.self_attn(h, h, h, mask=mask)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class GPT(nn.Module):
    """与 stage3 GPT 完全同构(embed×√d + 正弦位置编码 + weight tying)。"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_enc = PositionalEncoding(cfg.d_model, cfg.max_len, cfg.dropout)
        self.blocks = nn.ModuleList(GPTBlock(cfg) for _ in range(cfg.n_layers))
        Norm = RMSNorm if cfg.norm == "rms" else nn.LayerNorm
        self.norm = Norm(cfg.d_model)          # 收尾归一化(pre-norm 结构)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight    # weight tying
        self._init_weights(cfg)

    def _init_weights(self, cfg):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=cfg.d_model ** -0.5)
        with torch.no_grad():
            self.embedding.weight[cfg.pad_id].fill_(0.0)

    @staticmethod
    def causal_mask(size, device):
        return torch.tril(torch.ones(size, size, dtype=torch.bool, device=device)).view(1, 1, size, size)

    def embed(self, ids):
        return self.pos_enc(self.embedding(ids) * math.sqrt(self.cfg.d_model))

    def forward(self, ids):
        mask = self.causal_mask(ids.size(1), ids.device)
        x = self.embed(ids)
        for block in self.blocks:
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
