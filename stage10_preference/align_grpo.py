"""align_grpo.py —— 10.2 看齐自检:GRPO loss 与 minimind 实现数值一致

同 10.1 的做法:同一批合成张量上,我们的 grpo 管线(per_token_logps +
completion_mask + kl + min-clip loss,见 grpo.py)与 minimind train_grpo.py
的张量管线(逐字移植其 logp_pos / completion_mask / kl_div / min-clip,
见下方 minimind_loss)各算一遍,逐 token loss 应完全一致。

合成批要覆盖索引陷阱:prompt 长度不一致、有/无 EOS 的行、padding、
回答长于 R 的截断——索引约定错了就会在某个角落暴露。

用法(Spark,轻量,可 CPU):
  CUDA_VISIBLE_DEVICES="" ~/llm_study/.venv/bin/python align_grpo.py
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from grpo import (completion_mask, compute_advantages, grpo_loss,  # noqa: E402
                  kl_penalty, per_token_logps)

DEV = "cpu"
EOS, PAD = 2, 0


# ---------- minimind train_grpo.py 的张量管线(逐字移植 :101-143) ----------
def minimind_loss(model_out_logits, outputs, prompt_lens, old_logps,
                  completion_ids, eos_id, pad_id, rewards, num_gen,
                  beta=0.1, epsilon=0.2):
    per_token_logps_mm = F.log_softmax(model_out_logits[:, :-1, :], dim=-1)\
        .gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1)\
        .gather(1, prompt_lens.unsqueeze(1) - 1 +
                torch.arange(completion_ids.size(1), device=outputs.device)
                .unsqueeze(0))
    grouped = rewards.view(-1, num_gen)
    mean_r = grouped.mean(dim=1).repeat_interleave(num_gen)
    std_r = grouped.std(dim=1, unbiased=False).repeat_interleave(num_gen)
    advantages = (rewards - mean_r) / (std_r + 1e-4)
    completion_pad_mask = (completion_ids != pad_id).bool()
    is_eos = (completion_ids == eos_id) & completion_pad_mask
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1) - 1,
                         dtype=torch.long, device=outputs.device)
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    completion_mask_mm = ((torch.arange(is_eos.size(1), device=outputs.device)
                           .expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1))
                          & completion_pad_mask).int()
    kl_div = old_logps - per_token_logps_mm      # 用同一份 old 当"ref"
    per_token_kl = torch.exp(kl_div) - kl_div - 1
    ratio = torch.exp(per_token_logps_mm - old_logps)
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    per_token_loss1 = ratio * advantages.unsqueeze(1)
    per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
    per_token_loss = -(torch.min(per_token_loss1, per_token_loss2)
                       - beta * per_token_kl)
    return ((per_token_loss * completion_mask_mm).sum(dim=1)
            / completion_mask_mm.sum(dim=1).clamp(min=1)).mean()


def main():
    torch.manual_seed(0)
    B, T, R, V, K = 6, 40, 16, 100, 3          # 2 个 prompt × 3 生成
    # 造行:两档 prompt 长 + 有 EOS 与无 EOS + 回答长不一 + pad
    prompt_lens = torch.tensor([18, 22, 18, 22, 18, 22])
    full = torch.randint(3, V, (B, T))
    comp = torch.full((B, R), PAD)
    for i in range(B):
        n = 12 if i % 2 == 0 else R
        comp[i, :n] = torch.randint(3, V, (n,))
        if i < 4:                                # 4 行有 EOS(位置各异)
            comp[i, 5 + i] = EOS
        full[i, prompt_lens[i]:prompt_lens[i] + n] = comp[i, :n]
    lens = prompt_lens + comp.ne(PAD).sum(dim=1)
    for i in range(B):                           # 补回 full 的行尾 pad
        full[i, lens[i]:] = PAD
    logits = torch.randn(B, T, V)
    rewards = torch.randn(B) * 0.5 + 1.0
    old_logps = torch.randn(B, R).abs() * 0.5    # 任意"采样时 logp"

    # 我们的管线
    R_ = comp.shape[1]
    ours_mask = completion_mask(comp, EOS, PAD).float()
    ours_adv = compute_advantages(rewards, K)
    ours_lp = per_token_logps(logits, full, prompt_lens, R_)
    ours = grpo_loss(ours_lp, old_logps, old_logps, ours_adv, ours_mask)
    # minimind 管线(把 per_token_logps_mm 当 policy、old_logps 当 ref/old)
    mm = minimind_loss(logits, full, prompt_lens, old_logps, comp, EOS, PAD,
                       rewards, K)
    a, b = ours.mean().item(), mm.item()
    print(f"our grpo loss      = {a:.8f}")
    print(f"minimind grpo loss = {b:.8f}")
    print(f"Δ = {abs(a - b):.2e}  {'一致 ✓' if abs(a - b) < 1e-5 else '不一致 ✗'}")
    # 掩码一致性单独核一遍
    mm_mask = ((torch.arange(R_, device=comp.device).expand(B, -1) <=
                (torch.where((comp == EOS).any(1),
                             (comp == EOS).int().argmax(1),
                             torch.full((B,), R_ - 1, dtype=torch.long)))
                .unsqueeze(1)) & (comp != PAD)).int().float()
    print(f"mask Δ = {(ours_mask - mm_mask).abs().max().item():.0f}")


if __name__ == "__main__":
    main()
