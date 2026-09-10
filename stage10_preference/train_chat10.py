"""train_chat10.py —— 10.3:在 11-mm 上做 chat-template SFT

基座:11-mm(官方预训练语料版,stage11 产物)。数据:render_sft_chat.py 产出
的三段式 chat 样本。机制同 stage8/10.1 的 SFT(user_side 不参与 loss,
answer+tail 参与——tail 含 <|im_end|>,让模型学会"说完就收尾")。

可验证的三个问题:
  A. chat 格式 20 题(与官方 64M 同格式对打,官方 5-6/20)
  B. 语料内 val 是否被 SFT 遗忘(11-mm 原 2.267)
  C. raw 格式是否被遗忘(11-mm 原 ask ≈4/20)

用法(Spark):
  ~/llm_study/.venv/bin/python train_chat10.py --steps 1200
"""

import argparse
import json
import multiprocessing as mp
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
S11 = HERE.parent / "stage11_datascale"
sys.path.insert(0, str(HERE.parent / "stage4_scaling_bpe"))
sys.path.insert(0, str(HERE.parent / "stage9_modern_gpt"))
from bpe import BPETokenizer, EOS_ID, PAD_ID               # noqa: E402
from model_modern import GPT, GPTConfig                    # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IGNORE = -100


_TOK = None


def _init(tok):
    global _TOK
    _TOK = tok


def _enc(t):
    return _TOK.encode(t)


def build(rows, tok, max_len=512, workers=12):
    """三段 → (x, y):full = user_side+answer+tail+[EOS];掩码只盖 answer+tail。"""
    with mp.get_context("fork").Pool(processes=workers, initializer=_init,
                                     initargs=(tok,)) as pool:
        us = pool.map(_enc, [r["user_side"] for r in rows])
        an = pool.map(_enc, [r["answer"] for r in rows])
        tl = pool.map(_enc, [r["tail"] for r in rows])
    out = []
    for u, a, t in zip(us, an, tl):
        full = u + a + t + [EOS_ID]
        if len(full) > max_len or len(full) < 6:
            continue
        x = torch.tensor(full[:-1])
        y = torch.tensor(full[1:])
        start = len(u) - 1            # answer 第一个 token 的预测位置
        y[:start] = IGNORE
        out.append((x, y))
    return out


def collate(batch):
    L = max(len(x) for x, _ in batch)
    xs, ys = [], []
    for x, y in batch:
        p = L - len(x)
        xs.append(torch.cat([x, torch.full((p,), PAD_ID, dtype=torch.long)]))
        ys.append(torch.cat([y, torch.full((p,), IGNORE, dtype=torch.long)]))
    return torch.stack(xs).to(DEVICE), torch.stack(ys).to(DEVICE)


@torch.no_grad()
def eval_loss(model, examples, vocab, bs=32):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(examples), bs):
        x, y = collate(examples[i:i + bs])
        loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1),
                               ignore_index=IGNORE, reduction="sum")
        tot += loss.item()
        n += (y != IGNORE).sum().item()
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", default=str(S11 / "ckpt_11_1_mm.pt"))
    ap.add_argument("--bpe", default=str(S11 / "cache_mm" / "bpe.json"))
    ap.add_argument("--train-data", default=str(HERE / "chat_train.jsonl"))
    ap.add_argument("--test-data", default=str(HERE / "chat_test.jsonl"))
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt-out", default=str(HERE / "ckpt_10_3_chat_think.pt"))
    args = ap.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed)
    tok = BPETokenizer.load(args.bpe)
    tr = build([json.loads(ln) for ln in open(args.train_data, encoding="utf-8")][:40000],
               tok)
    te = build([json.loads(ln) for ln in open(args.test_data, encoding="utf-8")], tok)
    print(f"样本: 训练 {len(tr)} | 测试 {len(te)} | vocab {len(tok)}", flush=True)

    ckpt = torch.load(args.base_ckpt)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    print(f"基座: {sum(p.numel() for p in model.parameters())/1e6:.1f}M "
          f"(11-mm) | SFT 前 test loss: {eval_loss(model, te, len(tok)):.4f}",
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for step in range(1, args.steps + 1):
        model.train()
        batch = random.sample(tr, args.batch_size)
        x, y = collate(batch)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, len(tok)), y.reshape(-1),
                               ignore_index=IGNORE)
        opt.zero_grad(); loss.backward(); opt.step()
        if step == 1 or step % args.eval_every == 0:
            tl = eval_loss(model, te, len(tok))
            print(f"  step {step:5d} | train {loss.item():.4f} | test {tl:.4f}",
                  flush=True)

    torch.save({"model": model.state_dict(), "config": vars(cfg),
                "step": args.steps, "chat_of": args.base_ckpt}, args.ckpt_out)
    print(f"已保存 -> {args.ckpt_out}", flush=True)


if __name__ == "__main__":
    main()
