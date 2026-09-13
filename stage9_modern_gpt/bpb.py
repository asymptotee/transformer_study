"""bpb.py —— bits-per-byte:词表无关的 loss 口径(stage14 引入,源自 nanochat)

问题:按 token 平均的 CE loss 依赖分词器——同一段文本,词表切得越细、
每 token 越好猜、loss 越低。所以我们历史上"wiki 模型 3.776 vs 官方语料
模型 1.799"这类跨词表比较一直不可比。

bpb 把 loss 归一到**字节**:

    bpb = Σ(−log₂ p(token)) / Σ(bytes(token))

同一段文本字节数固定,与分词器无关 → 跨词表可比(仍需同一验证集;
换语料不是同一任务,只可作粗参照)。

实现要点(对齐 nanochat loss_eval.py 的口径):
  · 特殊 token(<pad>/<bos>/<eos>/<unk>)字节数记 0 → 自动被分母排除
  · ignore_index(-100/-0)位置的 loss 与字节都不计
  · 本仓 ignore_index=0 恰好是 <pad>(真实 \\x00 字符是 id 4),字节表 0 也扣掉
"""

import math

import torch
import torch.nn.functional as F


def build_token_bytes(tok, n_special=4):
    """token id → UTF-8 字节数;前 n_special 个特殊 token 记 0(不计入分母)。"""
    tb = torch.zeros(len(tok), dtype=torch.long)
    for i in range(n_special, len(tok)):
        tb[i] = len(tok.itos[i].encode("utf-8"))
    return tb


def ce_nats_and_bytes(logits, y, token_bytes, ignore_index=0):
    """返回 (Σ nats, Σ 字节)。logits (B,T,V),y (B,T)。"""
    V = logits.size(-1)
    token_bytes = token_bytes.to(y.device)
    per_tok = F.cross_entropy(logits.reshape(-1, V).float(),
                              y.reshape(-1),
                              ignore_index=ignore_index, reduction="none")
    nats = per_tok.sum().item()
    idx = y.reshape(-1).clamp(min=0)
    nbytes = token_bytes[idx].sum().item()          # 特殊 token 字节为 0,天然排除
    return nats, nbytes


def bpb(nats, nbytes):
    return (nats / math.log(2)) / max(nbytes, 1)
