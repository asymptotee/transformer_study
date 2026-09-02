"""lora.py —— LoRA（Low-Rank Adaptation）低秩适配

原理（CONCEPTS 第十节）：
  微调时权重变化量 ΔW 是低秩的，所以不直接学完整的 ΔW（d×k），
  而是分解成两个瘦矩阵 ΔW = B·A（B: d×r, A: r×k, r<<d,k）。
  原权重 W 冻住，只训 A、B → 只动约 1% 的参数。

  前向:  h = W·x + (α/r)·B·A·x
  初始化: A 随机、B=0 → 起点 ΔW=0，模型保持预训练原样。

这里实现：
  · LoRALinear：包住一个冻结的 Linear，加可训练的低秩 A、B
  · inject_lora：把 GPT 每层注意力的 W_Q、W_V 换成 LoRALinear，冻住其余参数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """用低秩增量替代对原 Linear 的全量微调。"""

    def __init__(self, linear: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.r = r
        self.scaling = alpha / r
        # 冻结原权重（克隆一份，不参与梯度）
        self.weight = nn.Parameter(linear.weight.data.clone(), requires_grad=False)
        # 可训练的低秩矩阵：A (r×in)，B (out×r)——建在宿主 Linear 所在设备，
        # 否则先 model.to(cuda) 再注入时 A/B 会留在 CPU（GPU 上会报设备不一致）
        self.A = nn.Parameter(torch.empty(r, linear.in_features,
                                          device=linear.weight.device))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        self.B = nn.Parameter(torch.zeros(linear.out_features, r,
                                          device=linear.weight.device))  # B=0 → 起点 ΔW=0

    def forward(self, x):
        base = F.linear(x, self.weight)                       # x·Wᵀ（冻结）
        lora = F.linear(F.linear(x, self.A), self.B)          # x·Aᵀ·Bᵀ
        return base + lora * self.scaling


def inject_lora(model, r=8, alpha=16):
    """给 GPT 每层注意力的 W_Q、W_V 注入 LoRA，冻住其余所有参数。

    返回可训练参数总数（应远小于总参数）。
    """
    for p in model.parameters():
        p.requires_grad = False                     # 先全部冻住
    for block in model.blocks:
        attn = block.self_attn
        attn.w_q = LoRALinear(attn.w_q, r=r, alpha=alpha)
        attn.w_v = LoRALinear(attn.w_v, r=r, alpha=alpha)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable


if __name__ == "__main__":
    # 自测：注入 LoRA 后起点输出应与原模型一致（B=0）
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "stage3_gpt"))
    from model_gpt import GPT, GPTConfig

    cfg = GPTConfig(vocab_size=100, d_model=64, n_layers=2, n_heads=4, d_ff=128)
    m = GPT(cfg).eval()
    x = torch.randint(0, 100, (2, 10))
    with torch.no_grad():
        before = m(x)
    n_train = inject_lora(m, r=8)
    with torch.no_grad():
        after = m(x)
    total = sum(p.numel() for p in m.parameters())
    print(f"总参数 {total:,} | LoRA 可训练 {n_train:,} ({n_train/total*100:.2f}%)")
    print(f"注入后输出与原模型一致: {torch.allclose(before, after)}  (B=0 起点)")
