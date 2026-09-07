"""train_grpo10.py —— 10.2:GRPO 在线训练(纯规则奖励,无 RM)

每步结构(对照 minimind train_grpo.py,差异见注释):
  1. rollout:一个 batch 的 B 个 prompt,各采样 K 个回答(温度 0.9,
     generate_cached,逐个序列——行数少,顺序即可)
  2. 规则奖励 rule_reward(校准见 grpo.py)→ 组内 advantage
  3. 同权重前向拿 old logps(ref 冻结拿 ref logps)——步骤内"采样即旧",
     单遍更新时 ratio≡1;**我们加 inner_epochs>1(同批数据多次更新),
     让 clip 真正绑定**(minimind 单遍更新,clip 结构上空转,见 README)
  4. loss = −E[min(ratio·A, clip(ratio)·A) − β·kl],回答区逐 token 平均

日志每步:reward / KL(ref vs π,逐 token)/ adv mean+std / 平均长度——
这些是判断"在线 RL 有没有推飞模型"的仪表盘。

用法(Spark):
  ~/llm_study/.venv/bin/python train_grpo10.py --steps 150 --epochs 1
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).parent
STAGE8 = HERE.parent / "stage8_knowledge_sft"
STAGE9 = HERE.parent / "stage9_modern_gpt"
STAGE7 = HERE.parent / "stage7_gpu_scale"
sys.path.insert(0, str(HERE.parent / "stage4_scaling_bpe"))
sys.path.insert(0, str(STAGE9))
from bpe import BPETokenizer, EOS_ID, PAD_ID                  # noqa: E402
from generate_cached import generate_cached_ids               # noqa: E402
from grpo import (completion_mask, compute_advantages, grpo_loss,  # noqa: E402
                  per_token_logps, rule_reward)
from model_modern import GPT, GPTConfig                       # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path)
    cfg = GPTConfig(**ckpt["config"])
    m = GPT(cfg).to(DEVICE)
    m.load_state_dict(ckpt["model"])
    return m, cfg


def rollout(prompts, tok, model, k, max_new, temperature):
    """对每个 prompt 采 k 个回答。返回补零对齐的张量与元信息。"""
    rows_p, rows_a, stopped = [], [], []
    lens = []
    for p in prompts:
        pid = tok.encode(p)
        for _ in range(k):
            ids = generate_cached_ids(model, tok, p, max_new,
                                      temperature=temperature, top_k=0,
                                      top_p=0.9, rep_penalty=1.1,
                                      eos_id=EOS_ID)
            rows_p.append(pid); rows_a.append(ids); lens.append(len(pid))
            stopped.append(len(ids) < max_new)
    R = max(len(a) for a in rows_a) or 1
    T = max(l + R for l, a in zip(lens, rows_a))
    full = []
    comp = []
    for pid, a in zip(rows_p, rows_a):
        seq = (pid + a)[:T]
        full.append(seq + [PAD_ID] * (T - len(seq)))
        comp.append(a + [PAD_ID] * (R - len(a)))
    return (torch.tensor(full, dtype=torch.long, device=DEVICE),
            torch.tensor(comp, dtype=torch.long, device=DEVICE),
            torch.tensor(lens, dtype=torch.long, device=DEVICE),
            stopped, rows_a)      # rows_a:原始回答(未 padding),奖励用它


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-ckpt", default=str(HERE / "ckpt_10_sft.pt"))
    ap.add_argument("--n-prompts", type=int, default=400, help="prompt 池大小")
    ap.add_argument("--k", type=int, default=6, help="每 prompt 采样数(组大小)")
    ap.add_argument("--b", type=int, default=4, help="每步 prompt 数")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--inner-epochs", type=int, default=2, help="同批数据复用次数")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--epsilon", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ckpt-out", default=str(HERE / "ckpt_10_grpo.pt"))
    args = ap.parse_args()

    global DEVICE
    if args.device != "auto":
        DEVICE = args.device
    torch.manual_seed(args.seed); random.seed(args.seed)

    tok = BPETokenizer.load(STAGE7 / "cache" / "bpe.json")
    pool = []
    with open(STAGE8 / "qa_train.jsonl", encoding="utf-8") as f:
        for ln in f:
            pool.append(json.loads(ln)["prompt"])
    random.shuffle(pool)
    pool = pool[:args.n_prompts]

    policy, cfg = load_model(args.sft_ckpt)
    ref, _ = load_model(args.sft_ckpt)
    ref.eval(); ref.requires_grad_(False)
    print(f"policy/ref: {sum(p.numel() for p in policy.parameters()):,} | "
          f"prompt 池 {len(pool)} | k={args.k} b={args.b} "
          f"inner={args.inner_epochs} β={args.beta} ε={args.epsilon}",
          flush=True)

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    steps_per_epoch = max(1, len(pool) // args.b)
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        random.shuffle(pool)
        for si in range(steps_per_epoch):
            batch_p = pool[si * args.b:(si + 1) * args.b]
            step += 1

            # ---- 1. rollout ----
            full, comp, lens, stopped, raw_a = rollout(batch_p, tok, policy,
                                                       args.k, args.max_new, 0.9)
            # ---- 2. 奖励与 advantage(用未 padding 的原始回答)----
            rew = []
            for idx in range(len(raw_a)):
                rew.append(rule_reward(full[idx, :lens[idx]].tolist(),
                                       raw_a[idx], stopped[idx]))
            rewards = torch.tensor(rew, device=DEVICE)
            adv = compute_advantages(rewards, args.k)
            mask = completion_mask(comp, EOS_ID).to(DEVICE).float()
            R = comp.shape[1]

            # ---- 3. old / ref logps(同权重,单遍时 old==new)----
            with torch.no_grad():
                old_logps = per_token_logps(policy(full), full, lens, R)
                ref_logps = per_token_logps(ref(full), full, lens, R)
                # 已停行:停在 EOS 之后的行 token 是 pad,logp 无意义,掩码兜底
                old_logps = old_logps.detach()

            # ---- 4. inner 更新(同批数据复用,让 clip 绑定)----
            loss_val = 0.0
            for _ in range(args.inner_epochs):
                pi_logps = per_token_logps(policy(full), full, lens, R)
                per_row = grpo_loss(pi_logps, old_logps, ref_logps, adv,
                                    mask, args.beta, args.epsilon)
                loss = per_row.mean()
                opt.zero_grad(); loss.backward(); opt.step()
                loss_val = loss.item()

            if step % 5 == 0 or step == 1:
                kl = ((ref_logps - pi_logps) * mask).sum() / mask.sum()
                n_ok = mask.sum().item()
                print(f"step {step:4d} | reward {rewards.mean().item():+.3f} "
                      f"| adv {adv.mean():+.3f}±{adv.std():.3f} | "
                      f"KL {kl.item():+.4f} | len {n_ok / (len(batch_p)*args.k):.1f} "
                      f"| loss {loss_val:.4f} | {time.time()-t0:.0f}s", flush=True)

    torch.save({"model": policy.state_dict(), "config": vars(cfg),
                "step": step, "beta": args.beta, "grpo_of": args.sft_ckpt},
               args.ckpt_out)
    print(f"已保存 -> {args.ckpt_out}", flush=True)


if __name__ == "__main__":
    main()
