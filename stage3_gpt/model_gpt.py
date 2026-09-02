"""model_gpt.py —— decoder-only GPT（语言模型）

核心洞察：**GPT 就是阶段一 Transformer 的 decoder 半边。**
  去掉 encoder 和交叉注意力，只保留"因果自注意力 + FFN"，
  训练目标改成"纯下一个 token 预测"——这就是 GPT。

所以本文件大量复用 stage1_basics/model.py 的零件：
  MultiHeadAttention（带因果掩码用）、FeedForward、PositionalEncoding。
新写的只有：GPTBlock（= DecoderBlock 去掉交叉注意力）和 GPT（组装 + LM 前向）。

与阶段一 Transformer 的结构对比：
  Transformer:  src→Encoder ─┐
              tgt→Decoder(因果自注意力+交叉注意力+FFN)→logits   （seq2seq）
  GPT:          ids→(因果自注意力+FFN)×N→logits                （语言模型，无 encoder）
"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "stage1_basics"))
from model import FeedForward, MultiHeadAttention, PositionalEncoding  # 复用阶段一组件


@dataclass
class GPTConfig:
    vocab_size: int        # 词表大小（由语料字符集决定）
    d_model: int = 128     # 模型宽度
    n_heads: int = 4       # 注意力头数
    n_layers: int = 4      # GPT 层数（只有 decoder 那种层）
    d_ff: int = 256        # 前馈隐藏层维度
    dropout: float = 0.1
    max_len: int = 256     # 位置编码上限（支持的最长序列）
    pad_id: int = 0


class GPTBlock(nn.Module):
    """GPT 单层 = 因果自注意力 + FFN（各带 pre-norm 残差）。

    对照阶段一的 DecoderBlock：就是它**去掉交叉注意力那个子层**。
    GPT 没有 encoder 可看，信息只来自序列自身（因果掩码保证只看过去）。
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ff = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)

    def forward(self, x, mask):
        h = self.norm1(x)
        attn_out, _ = self.self_attn(h, h, h, mask=mask)  # q=k=v：自注意力
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class GPT(nn.Module):
    """decoder-only 语言模型。

    数据流：
        ids ─▶ Embed×√d + 位置编码 ─▶ GPTBlock×N ─▶ LayerNorm ─▶ lm_head ─▶ logits
    logits[:, t, :] 预测的是位置 t+1 的 token（下一个词）。
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_enc = PositionalEncoding(cfg.d_model, cfg.max_len, cfg.dropout)
        self.blocks = nn.ModuleList(GPTBlock(cfg) for _ in range(cfg.n_layers))
        self.norm = nn.LayerNorm(cfg.d_model)          # pre-norm 结构的收尾归一化
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight    # weight tying（同阶段一）
        self._init_weights(cfg)

    def _init_weights(self, cfg):
        # embedding 用 std=1/√d_model 初始化（同阶段一，避免注意力分数爆炸）
        nn.init.normal_(self.embedding.weight, mean=0.0, std=cfg.d_model ** -0.5)
        with torch.no_grad():
            self.embedding.weight[cfg.pad_id].fill_(0.0)

    @staticmethod
    def causal_mask(size, device):
        """因果掩码（下三角）：位置 t 只能看 ≤t 的位置。GPT 唯一的掩码。"""
        return torch.tril(torch.ones(size, size, dtype=torch.bool, device=device)).view(1, 1, size, size)

    def embed(self, ids):
        return self.pos_enc(self.embedding(ids) * math.sqrt(self.cfg.d_model))

    def forward(self, ids):
        """ids: (B, T)。返回 logits (B, T, vocab)，logits[:, t] 预测位置 t+1。"""
        mask = self.causal_mask(ids.size(1), ids.device)
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x, mask)
        return self.lm_head(self.norm(x))

    @torch.no_grad()
    def generate(self, ids, max_new_tokens, temperature=0.8, top_k=None):
        """从 prompt（ids）自回归续写。

        temperature<=0 贪心；否则按温度采样。top_k 只在最高的 k 个候选里采样。
        这是 LLM 生成的标准做法，和阶段一 generate 同源，只是这里是"续写"而非"输入→输出"。
        """
        self.eval()
        for _ in range(max_new_tokens):
            # 序列超过 max_len 时截断最早的（位置编码表只有 max_len 行）
            ctx = ids if ids.size(1) <= self.cfg.max_len else ids[:, -self.cfg.max_len:]
            logits = self.forward(ctx)[:, -1]              # 只看最后一个位置
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")  # 砍掉 top-k 以外
            if temperature <= 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, nxt], dim=1)
        return ids
