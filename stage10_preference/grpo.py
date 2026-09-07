"""grpo.py —— GRPO 的推导与最小实现(10.2 核心)

目标:在可验证(规则)奖励下,不学奖励模型(critic),直接在线优化 policy。

推导(DeepSeekMath 式):
1) 对每个 prompt x,policy 采样一组 K 个回答 {y_1..y_K},规则给分 r_k。
   绝对分不可比(不同 prompt 的分数尺度不同)→ 组内归一化成 advantage:
      A_k = (r_k − mean_k) / (std_k + ε)
   "这个回答比同组的其他 K−1 个好多少"——只需要相对排名,不学 RM。

2) 策略梯度(PG)更新 π_θ,y 是 θ_old 采的 → 用重要性比修正:
      ratio_t = π_θ(y_t|x) / π_θ_old(y_t|x) = exp(new_logp − old_logp)
   PPO 的 clip 防止单步走太远:
      L = −E[ min(ratio·A, clip(ratio, 1±ε)·A) ]
   min 取小 = 对"让变差的步子"不给好处:ratio 超过 [1−ε,1+ε] 时增益封顶。
   (重要:clip 只有在**同一批采样数据被多次复用**(micro-epoch)时才真正
   绑定——单遍更新时 new==old、ratio≡1、clip 空转。minimind 的 trainer
   每步只更新一遍,clip 结构上失效;我们默认 inner_epochs=2 让 clip 真
   正工作,这也是比照时发现的机制点,见 README。)

3) KL 刹车:在线 RL 会把 π 推离 SFT 起点(π_ref)→ 逐 token 惩罚,公式
   (与 minimind train_grpo.py:132-133 逐行对应):
      δ = log π_ref − log π      (逐 token)
      kl = exp(δ) − δ − 1         (对"π 跑得比 ref 高或低太多"都惩罚)
   loss = −E[ min(ratio·A, clip(ratio)·A) − β·kl ]   (逐 token,回答区)

公式与 minimind 的差异只有一处:它单遍更新(clip 空转),我们加 micro-epoch。
逐公式数值对齐见 align_grpo.py(用它的张量管线喂我们/它的 loss)。
"""

import torch
import torch.nn.functional as F


def compute_advantages(rewards, num_gen, eps=1e-4):
    """rewards (B·K,) → advantages (B·K,)。组 = 每 num_gen 个。"""
    g = rewards.view(-1, num_gen)
    mean_r = g.mean(dim=1, keepdim=True)
    std_r = g.std(dim=1, unbiased=False, keepdim=True)
    adv = (g - mean_r) / (std_r + eps)
    return adv.reshape(-1)


def per_token_logps(logits, ids, prompt_lens, R):
    """logits (B,T,V),ids (B,T) → 回答区每位置 logp (B,R)。

    与 minimind train_grpo.py:101 同约定:logits[:, :-1] 对齐 ids[:, 1:],
    logp_pos = prompt_lens−1+arange(R)——取"回答第一个 token 起"的 R 个。
    R 是补零对齐后的回答最大长(短回答/已停行的行,padding 区由 mask 管)。
    """
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    gathered = torch.gather(logp, 2, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    pos = (prompt_lens.unsqueeze(1) - 1 +
           torch.arange(R, device=ids.device).unsqueeze(0))
    return gathered.gather(1, pos.clamp(max=gathered.shape[1] - 1))


def completion_mask(completion_ids, eos_id, pad_id=0):
    """回答 token 矩阵 (B,R) 的掩码:非 pad ∧ 不越过第一个 EOS。

    与 minimind train_grpo.py:126-130 同语义(它管 completion_ids +
    completion_mask,这里合成一个函数)。
    """
    R = completion_ids.shape[1]
    is_eos = completion_ids == eos_id
    eos_idx = torch.full((is_eos.shape[0],), R - 1, dtype=torch.long,
                         device=completion_ids.device)
    has = is_eos.any(dim=1)
    eos_idx[has] = is_eos.int().argmax(dim=1)[has]
    ar = torch.arange(R, device=completion_ids.device).unsqueeze(0)
    return ((ar <= eos_idx.unsqueeze(1)) &
            (completion_ids != pad_id)).int()


def kl_penalty(ref_logps, pi_logps):
    """逐 token KL 项(与 minimind :132-133 同式)。"""
    d = ref_logps - pi_logps
    return torch.exp(d) - d - 1


def grpo_loss(pi_logps, old_logps, ref_logps, advantages, mask, beta=0.1,
              epsilon=0.2):
    """经典 PPO-clip 版 GRPO loss(回答区逐 token 平均)。"""
    ratio = torch.exp(pi_logps - old_logps)
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    adv = advantages.unsqueeze(1)
    l1 = ratio * adv
    l2 = clipped * adv
    kl = kl_penalty(ref_logps, pi_logps)
    per_tok = -(torch.min(l1, l2) - beta * kl) * mask
    return per_tok.sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def rule_reward(prompt_ids, ans_ids, stopped_eos):
    """10.2 的规则奖励(校准过,SFT 基线均值 ≈ +1.0/上限 2.0,有余量)。

    组件全部连续,方向与 10.1 的三条规则一致,但把"阈值打分"改成
    "连续惩罚/带奖励"让梯度更平滑:
      len_band : 8 ≤ len ≤ 40 → +1;过短 −1;过长 −0.5(惩罚长尾游荡)
      eos_bonus: 提前以 EOS 收尾 +0.5
      rep_pen  : 重复率 dup → −min(2·dup, 1)
      echo_pen : 与前 8 个 prompt token 的重叠率 → −min(2·echo, 1)
    """
    n = len(ans_ids)
    r = 1.0 if 8 <= n <= 40 else (-1.0 if n < 8 else -0.5)
    if stopped_eos:
        r += 0.5
    if ans_ids:
        dup = 1 - len(set(ans_ids)) / n
        r -= min(2 * dup, 1.0)
        pset = set(prompt_ids[-8:])
        echo = len(set(ans_ids) & pset) / max(1, len(pset))
        r -= min(2 * echo, 1.0)
    return r
