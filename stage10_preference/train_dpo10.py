"""train_dpo10.py —— 10.1 第 3 步:DPO 训练(policy 起点 = SFT,ref 冻结 = SFT)

机制与 minimind train_dpo.py 逐行对应:
  · policy 与 ref 同权重初始化(SFT),ref requires_grad_(False)
  · batch 里 chosen 在前半、rejected 在后半,一次前向各算一遍
  · loss = −logsigmoid(β·((π_w−π_l) − (ref_w−ref_l))),β 默认 0.15
  · lr 极小(默认 1e-6,minimind 用 4e-8 防遗忘;我们的规则信号更强,
    留了更大余地,效果不好再降)

内置指标(dev 对,前向即可,不生成):
  · implicit margin = ((π_w−π_l) − (ref_w−ref_l)) 的均值 → DPO 在压什么
  · chosen 上的 policy−ref logp 差 = 单样本 KL 估计 → 偏离参考模型多远
  · loss 起点应 ≈ log 2 ≈ 0.693(π 初态 = ref,logit=0)

行为评估(生成)见 eval_pref.py(harness 20 题 + 规则指标)。

用法(Spark):
  ~/llm_study/.venv/bin/python train_dpo10.py --epochs 1 --lr 1e-6
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
STAGE9 = HERE.parent / "stage9_modern_gpt"
STAGE7 = HERE.parent / "stage7_gpu_scale"
sys.path.insert(0, str(HERE.parent / "stage4_scaling_bpe"))
sys.path.insert(0, str(STAGE9))
from bpe import BPETokenizer, EOS_ID, PAD_ID           # noqa: E402
from dpo import build_batch, dpo_loss, logits_to_log_probs  # noqa: E402
from model_modern import GPT, GPTConfig                # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path)
    cfg = GPTConfig(**ckpt["config"])
    m = GPT(cfg).to(DEVICE)
    m.load_state_dict(ckpt["model"])
    return m, cfg


@torch.no_grad()
def dev_metrics(policy, ref, pairs, tok):
    """在 dev 对上前向,算隐式奖励差与 KL 估计。pairs 是 dict 列表。"""
    flat = []
    for ex in pairs:
        flat.append((ex["prompt"], ex["chosen"]))
        flat.append((ex["prompt"], ex["rejected"]))
    x, y, mask = build_batch(flat, tok, EOS_ID)
    policy_logps = (logits_to_log_probs(policy(x), y) * mask)
    ref_logps = (logits_to_log_probs(ref(x), y) * mask)
    B2 = x.shape[0]
    pi_w = policy_logps[:B2 // 2].sum(1); pi_l = policy_logps[B2 // 2:].sum(1)
    rf_w = ref_logps[:B2 // 2].sum(1); rf_l = ref_logps[B2 // 2:].sum(1)
    margin = ((pi_w - pi_l) - (rf_w - rf_l)).mean().item()
    kl_est = (policy_logps[:B2 // 2] - ref_logps[:B2 // 2]).sum(1).mean().item()
    return margin, kl_est


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-ckpt", default=str(HERE / "ckpt_10_1_sft.pt"))
    ap.add_argument("--data", default=str(HERE / "pref_train.jsonl"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=4, help="每步的对数")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.15)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--n-dev", type=int, default=128, help="留作 dev 的对数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ckpt-out", default=str(HERE / "ckpt_10_1_dpo.pt"))
    args = ap.parse_args()

    global DEVICE
    if args.device != "auto":
        DEVICE = args.device
    torch.manual_seed(args.seed); random.seed(args.seed)

    tok = BPETokenizer.load(STAGE7 / "cache" / "bpe.json")
    raw = [json.loads(ln) for ln in open(args.data, encoding="utf-8")]
    random.shuffle(raw)
    dev, train = raw[:args.n_dev], raw[args.n_dev:]
    print(f"偏好对: 训练 {len(train)} | dev {len(dev)} | β={args.beta}", flush=True)

    policy, cfg = load_model(args.sft_ckpt)
    ref, _ = load_model(args.sft_ckpt)
    ref.eval(); ref.requires_grad_(False)
    print(f"policy/ref: {sum(p.numel() for p in policy.parameters()):,} 参数",
          flush=True)

    # 平铺:chosen 对 前半,rejected 后半(与 minimind 同布局)
    flat = []
    for ex in train:
        flat.append((ex["prompt"], ex["chosen"]))
        flat.append((ex["prompt"], ex["rejected"]))
    n_pairs = len(flat) // 2
    iters_per_epoch = n_pairs // args.batch_size

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    step = 0
    for epoch in range(args.epochs):
        random.shuffle(flat)
        for bi in range(iters_per_epoch):
            chunk = flat[bi * args.batch_size:(bi + 1) * args.batch_size]
            x, y, mask = build_batch(chunk, tok, EOS_ID, max_len=160)
            if x.shape[0] == 0:
                continue
            step += 1
            policy.train()
            with torch.no_grad():
                ref_logps = (logits_to_log_probs(ref(x), y) * mask).detach()
            pi_logps = logits_to_log_probs(policy(x), y) * mask
            loss = dpo_loss(pi_logps, ref_logps, mask, args.beta)
            opt.zero_grad(); loss.backward(); opt.step()

            if step == 1 or step % args.eval_every == 0:
                margin, kl = dev_metrics(policy, ref, dev, tok)
                print(f"  step {step:5d} | dpo {loss.item():.4f} | "
                      f"margin {margin:+.4f} | KL~ {kl:+.4f} | "
                      f"lr {opt.param_groups[0]['lr']:.1e}", flush=True)

    margin, kl = dev_metrics(policy, ref, dev, tok)
    print(f"\n完成 | dev margin {margin:+.4f} | KL~ {kl:+.4f}", flush=True)
    torch.save({"model": policy.state_dict(), "config": vars(cfg),
                "step": step, "beta": args.beta, "dpo_of": args.sft_ckpt},
               args.ckpt_out)
    print(f"已保存 -> {args.ckpt_out}", flush=True)


if __name__ == "__main__":
    main()
