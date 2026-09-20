"""train_mt.py —— 14.2:多轮 chat SFT(按段拼掩码)

与 stage10 的 train_chat10.py 是同一套机制(只对 assistant 段算 loss、
user_side 不参与、tail 含 <|im_end|> 让模型学会收尾),**唯一的结构差别是
掩码从"一个连续区间"变成"按段多个区间"** —— 因为 14.2 起每个 assistant 轮
都要算 loss(minimind 的做法,信号量约 3×)。

  train_chat10.py: {user_side, answer, tail} → y[:len(user_side)-1] = IGNORE
  train_mt.py    : {segs:[{t,loss}…]}        → 逐段拼 mask,loss=0 的段位置 IGNORE

三段式是段列表的特例 [0,1,1],所以本文件的 build() 向下兼容单轮数据。

窗口 768(不是 stage11 的 512):依据见 stage9 README 9.2 节的更正块 —— SFT 在
≤512 上训过就能把 768 的 ppl 从 22.0 拉到 7.1;而多轮样本中位数 499 / 90 分位
760,512 会砍掉 45%。

用法(Spark):
  ~/llm_study/.venv/bin/python train_mt.py \
      --base-ckpt ../stage11_datascale/ckpt_11_5_final.pt \
      --bpe ../stage11_datascale/cache_mm10g/bpe.json \
      --train-data mt_train_nt.jsonl --test-data mt_test_nt.jsonl \
      --steps 2000

产物放本阶段目录(`ckpt_14_mt.pt`)。ckpt 的归属约定:每个阶段自己留自己的,
不往别的阶段目录里塞 —— 早阶段(stage1/3/5/6)本地就是各存各的,只有 11.x
那条线因为连续演进放在一起。**eval 结果是例外**(见下),统一放 stage13。
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


def build(rows, tok, max_len=768, workers=12):
    """段列表 → (x, y):逐段编码后拼接,loss=0 的段在 y 里抹成 IGNORE。

    超长样本**丢弃**(不是截断)—— 与 stage10/11 一致,保住与单轮那条线的可比性。
    768 下丢样率约 10%(512 时是 45%),所以这个取舍现在代价很小。
    """
    segs_per_row = [r["segs"] for r in rows]
    flat = [g["t"] for segs in segs_per_row for g in segs]
    with mp.get_context("fork").Pool(processes=workers, initializer=_init,
                                     initargs=(tok,)) as pool:
        enc = pool.map(_enc, flat)

    out, k = [], 0
    for segs in segs_per_row:
        ids, mask = [], []
        for g in segs:
            e = enc[k]                     # k 必须无条件前进(跳样时也不能错位)
            k += 1
            ids += e
            mask += [g["loss"]] * len(e)
        ids.append(EOS_ID)
        mask.append(1)
        if len(ids) > max_len or len(ids) < 6:
            continue
        x = torch.tensor(ids[:-1])
        y = torch.tensor(ids[1:])
        m = torch.tensor(mask[1:], dtype=torch.bool)
        y[~m] = IGNORE
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
def eval_loss(model, examples, vocab, bs=16):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(examples), bs):
        x, y = collate(examples[i:i + bs])
        loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1),
                               ignore_index=IGNORE, reduction="sum")
        tot += loss.item()
        n += (y != IGNORE).sum().item()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", default=str(S11 / "ckpt_11_5_final.pt"))
    ap.add_argument("--bpe", default=str(S11 / "cache_mm10g" / "bpe.json"))
    ap.add_argument("--train-data", default=str(HERE / "mt_train_nt.jsonl"))
    ap.add_argument("--test-data", default=str(HERE / "mt_test_nt.jsonl"))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt-out", default=str(HERE / "ckpt_14_mt.pt"))
    args = ap.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed)
    tok = BPETokenizer.load(args.bpe)
    tr = build([json.loads(ln) for ln in open(args.train_data, encoding="utf-8")],
               tok, max_len=args.max_len)
    te = build([json.loads(ln) for ln in open(args.test_data, encoding="utf-8")],
               tok, max_len=args.max_len)
    print(f"样本: 训练 {len(tr)} | 测试 {len(te)} | 窗口 {args.max_len} "
          f"| vocab {len(tok)}", flush=True)

    ckpt = torch.load(args.base_ckpt)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    print(f"基座: {sum(p.numel() for p in model.parameters())/1e6:.1f}M "
          f"← {Path(args.base_ckpt).name} | "
          f"SFT 前 test loss: {eval_loss(model, te, len(tok)):.4f}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for step in range(1, args.steps + 1):
        model.train()
        batch = random.sample(tr, args.batch_size)
        x, y = collate(batch)
        loss = F.cross_entropy(model(x).reshape(-1, len(tok)), y.reshape(-1),
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
