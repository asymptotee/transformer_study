"""dpo.py —— DPO 的推导与最小实现(10.1 核心,机制学习)

从偏好到目标函数,自己推一遍(不依赖任何库的现成实现):

1) Bradley-Terry:人类偏好服从 p(y_w > y_l | x) = σ(r(x,y_w) − r(x,y_l)),
   其中 r 是(未知的)奖励函数,σ 是 sigmoid。SFT 后想对齐偏好,经典做法是
   RLHF:先学 r 再 PPO 最大化 r。DPO 的洞察:这两步可以合并——

2) RLHF 的目标 π* = argmax E[r(x,y)] − β·KL(π‖π_ref),闭式解是
      π*(y|x) = π_ref(y|x) · exp(r(x,y)/β) / Z(x)
   反解出奖励:r(x,y) = β·log(π*(y|x)/π_ref(y|x)) + β·log Z(x)

3) 把 r 的这个"隐式形式"代回 Bradley-Terry(配分函数 Z(x) 与 y 无关,
   会约掉),得到 **DPO loss**(只依赖当前 policy π 与冻结的 π_ref,无 RM、
   无采样):

      L = −E [ log σ( β·( log π(y_w|x)/π_ref(y_w|x)
                       − log π(y_l|x)/π_ref(y_l|x) ) ) ]

   直觉:σ 的输入 = policy 相对参考模型"偏好胜出多少"的对数比。
   胜出越多 loss 越小;若 policy 变成和 ref 一样(初始态),logit=0,
   loss = log 2 ≈ 0.693——训练就是从 0.693 往下压。

4) 实现(与 minimind `trainer/train_dpo.py` 的 dpo_loss 逐行对应,β 默认
   0.15,loss 相同 batch 上数值一致见 align 脚本):
      · x/y:同一序列的输入/下一个 token 标签(chosen 与 rejected 各自
        prompt+回答;prompt 与 padding 位置的 mask=0)
      · logp:log_softmax(logits)[y] 逐 token 取到,再按 mask 求和
      · policy 与 ref 用同一份 x 前向;ref 冻结、no_grad

数值注意:log 域全程 fp32 累积(求和前 dtype 统一),避免 β×logit 的
bf16 损失;这是"与 minimind 数值一致"能成立的前提之一(它们没做,但
对齐脚本里两边都按同一实现算)。
"""

import torch
import torch.nn.functional as F


def logits_to_log_probs(logits, labels):
    """logits (B,T,V) → 每位置真实标签的 logp (B,T)。"""
    log_probs = F.log_softmax(logits.float(), dim=2)   # float 域算,避免 bf16
    return torch.gather(log_probs, 2, labels.unsqueeze(2)).squeeze(-1)


def dpo_loss(policy_logps, ref_logps, mask, beta=0.15):
    """minimind train_dpo.py 的同款结构:chosen 在 batch 前半,rejected 后半。

    policy_logps / ref_logps:(2B, T) 逐 token logp(已对 y 对齐);
    mask:(2B, T) 回答区为 1。返回标量 loss。
    """
    policy_sum = (policy_logps * mask).sum(dim=1)      # (2B,)
    ref_sum = (ref_logps * mask).sum(dim=1)
    B2 = policy_sum.shape[0]
    pi_w, pi_l = policy_sum[:B2 // 2], policy_sum[B2 // 2:]
    ref_w, ref_l = ref_sum[:B2 // 2], ref_sum[B2 // 2:]
    pi_logratios = pi_w - pi_l
    ref_logratios = ref_w - ref_l
    logits = pi_logratios - ref_logratios
    return -F.logsigmoid(beta * logits).mean()


@torch.no_grad()
def sequence_logps(model, tok, x, y, mask):
    """一次性算 (policy 或 ref) 在一批 (x,y,mask) 上的逐 token logp。"""
    logits = model(x)
    logps = logits_to_log_probs(logits, y)
    logps = logps * mask
    return logps, logits


def build_batch(pairs, tok, eos_id, max_len=128, pad_id=0):
    """pairs: [(prompt, chosen, rejected)] → 训练用 (x, y, mask)。

    full = prompt + answer + EOS;x = full[:-1],y = full[1:];
    mask 只在"预测回答 token"的位置为 1(与 stage8 的 CE 掩码同约定,
    换成 0/1 版供 DPO 乘用;padding 的 y 填 pad_id,但 mask=0 使其不参与)。
    """
    xs, ys, masks = [], [], []
    for prompt, answer in pairs:
        p, a = tok.encode(prompt), tok.encode(answer)
        full = p + a + [eos_id]
        if len(full) > max_len or len(full) < 4:
            continue
        x = torch.tensor(full[:-1], dtype=torch.long)
        y = torch.tensor(full[1:], dtype=torch.long)
        m = torch.zeros(len(y), dtype=torch.float32)
        m[len(p) - 1:] = 1.0                       # 回答区(含 EOS 预测)
        xs.append(x); ys.append(y); masks.append(m)
    L = max(len(t) for t in xs)
    def pad(ts, v):
        return torch.stack([torch.cat([t, torch.full((L - len(t),), v,
                                                     dtype=t.dtype)])
                            for t in ts])
    return (pad(xs, pad_id).to(DEVICE), pad(ys, pad_id).to(DEVICE),
            pad(masks, 0.0).to(DEVICE))


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
