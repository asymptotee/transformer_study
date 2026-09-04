"""rope.py —— RoPE 旋转位置编码 + YaRN 长度外推(9.2)

为什么要换掉正弦位置编码(stage1 起一直在用)?
  正弦编码把位置信息**加**进 embedding,一次注入后各层只能"读到"那个静态值;
  RoPE 把位置信息**旋**进 Q/K:q、k 先按各自绝对位置旋转,再做点积——
  旋转后的内积正好等于"原内积 × 相对位置的旋转角":

      (R_p q)·(R_p' k) = qᵀ R_{p'-p} k          (旋转矩阵的差 = 相对旋转)

  于是注意力天然只依赖**相对位置**,而且:
  ① 不用学位置参数(可外推到训练长度之外——正弦表是查表,越界即崩;
     RoPE 的频率是连续函数,任意长的位置都能算);
  ② 位置信息进到每一层的注意力里,而不是只在输入层加一次。

实现与 minimind `model/model_minimind.py` 的 precompute_freqs_cis /
apply_rotary_pos_emb(:62-84)逐行对应(见 README 9.2 的对照笔记),以便
逐行 diff。注意它的 rotate_half 配对是 (i, i+dim/2) 半拆分式
(GPT-J/NeoX 风格),cos/sin 表按两半重复;与 LLaMA 的相邻配对 (2i, 2i+1)
差一个维度置换——两者都是合法旋转,对训练等价,只是换权重时不能混用。
"""

import math

import torch


def precompute_freqs_cis(dim: int, end: int, theta: float = 1e6,
                         rope_scaling: dict = None):
    """预计算位置表:返回 cos/sin,形状 (end, dim)。

    dim   = head_dim(每头维度);dim/2 个基频,第 j 个频率 = theta^{-2j/dim}
    theta = 基频底数(rope_theta)。越大 → 高频越少、低频分量周期越长。
            训练于 theta=1e6 时,低维分量在短序列内几乎不动 → 位置信息
            主要靠高维分量;这也是它能"撑"更长序列的原因之一。
    end   = 表长。训练 256 时也可预计算到 4096:外推测试直接查表,无需重训。
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))

    # ---- YaRN(推理期长度外推,不重训)----
    # 动机:长度变长后,高维分量的频率太高,周期 < 1 个 token → 相邻位置
    # 角度混叠,模型分不清相邻与更远。粗暴办法是全体插值(freq/factor),
    # 但那会破坏短距离分辨力;YaRN 只对"高频段"(维度下标大)插值,
    # 低频段保持原频,中间用 ramp 平滑过渡 —— ramp 区间由 beta_fast/beta_slow
    # 这两个"周期"边界反解出维度位置。语义同 minimind:62-78。
    if rope_scaling is not None:
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)
        factor = rope_scaling.get("factor", 16.0)
        beta_fast = rope_scaling.get("beta_fast", 32.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)
        if end / orig_max > 1.0:
            def inv_dim(beta):
                return (dim * math.log(orig_max / (2 * math.pi * beta))
                        / (2 * math.log(theta)))
            low = max(math.floor(inv_dim(beta_fast)), 0)
            high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp(
                (torch.arange(dim // 2).float() - low) / max(high - low, 0.001),
                0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)

    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()              # (end, dim/2)
    # cos/sin 两半重复:rotate_half 按 (i, i+dim/2) 配对旋转
    cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return cos, sin                                     # (end, dim)


def apply_rotary_pos_emb(q, k, cos, sin):
    """q/k 形状 (B, H, T, D),cos/sin (T, D)→ 就地按绝对位置旋转。

    rotate_half 把后一半取负放前一半:配合两半重复的 cos/sin,
    (i, i+D/2) 两个分量互相转成对方 → 等价于 2D 旋转一个角度 θ_p。
    注意本仓库 q/k 布局是 (B, H, T, D)(头在第二维),cos 要扩成
    (1, 1, T, D);minimind 的布局是 (B, T, H, D),它的 unsqueeze(1)
    等价位置不同——照搬要小心(9.2 踩坑,见 README)。
    """

    def rotate_half(x):
        d = x.shape[-1]
        return torch.cat((-x[..., d // 2:], x[..., : d // 2]), dim=-1)

    cos = cos[None, None]                    # (1, 1, T, D) 广播到 B、H
    sin = sin[None, None]
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k
