"""train_best.py —— 阶段五 capstone：把所有技术合起来，尽量提升生成质量

组合拳（全是前几个阶段学过的）：
  · BPE 分词（更多合并 → 更短序列、更词级）      ← 阶段四
  · 更大模型（d_model/层数加大）                 ← 阶段四缩放实验
  · 更长上下文（block_size 加大，看到更多前文）
  · 更多训练步数
目标：在 CPU 可达范围内把鲁迅续写推到最好，并诚实评估天花板。

运行（先小步数探耗时，再完整训）：
  PY=python
  $PY -u train_best.py --steps 30 --eval-every 10      # 探每步耗时
  $PY -u train_best.py                                  # 完整训练（后台）
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "stage3_gpt"))        # 复用 GPT
sys.path.insert(0, str(HERE.parent / "stage4_scaling_bpe"))  # 复用 BPE
from bpe import BPETokenizer
from model_gpt import GPT, GPTConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CORPUS = HERE.parent / "stage3_gpt" / "luxun_clean.txt"


def get_batch(data, vocab, block, batch):
    ix = torch.randint(len(data) - block - 1, (batch,))
    x = torch.stack([data[i:i + block] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block] for i in ix])
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merges", type=int, default=1000)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--d-ff", type=int, default=512)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=300)
    ap.add_argument("--device", default="auto", help="cpu/cuda；默认 auto（有 GPU 用 GPU）")
    ap.add_argument("--ckpt", default=str(HERE / "ckpt_best.pt"))
    args = ap.parse_args()

    global DEVICE
    if args.device != "auto":
        DEVICE = args.device
    print(f"设备: {DEVICE}")

    torch.manual_seed(42)
    text = CORPUS.read_text(encoding="utf-8")

    print(f"训练 BPE（{args.merges} 合并）...")
    bpe = BPETokenizer.train(text, num_merges=args.merges, verbose=False)
    bpe.save(HERE / "bpe_best.json")
    print("编码全文（一次性）...")
    t0 = time.time()
    ids = torch.tensor(bpe.encode(text), dtype=torch.long).to(DEVICE)
    print(f"  {len(text):,} 字符 → {len(ids):,} token "
          f"({len(text)/len(ids):.2f} 字符/token)，编码 {time.time()-t0:.0f}s")

    n_val = int(len(ids) * 0.1)
    train_ids, val_ids = ids[:-n_val], ids[-n_val:]

    cfg = GPTConfig(vocab_size=len(bpe), pad_id=0, max_len=max(args.block_size, 256),
                    d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
                    d_ff=args.d_ff)
    model = GPT(cfg).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {n_params:,}  (d_model={args.d_model}, {args.n_layers}层, "
          f"{args.n_heads}头, block={args.block_size}, batch={args.batch_size})")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best = float("inf")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        x, y = get_batch(train_ids, len(bpe), args.block_size, args.batch_size)
        loss = F.cross_entropy(model(x).reshape(-1, len(bpe)), y.reshape(-1),
                               ignore_index=0)
        opt.zero_grad(); loss.backward(); opt.step()
        if step == 1 or step % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                xv, yv = get_batch(val_ids, len(bpe), args.block_size, args.batch_size)
                vl = F.cross_entropy(model(xv).reshape(-1, len(bpe)),
                                     yv.reshape(-1), ignore_index=0)
            best = min(best, vl.item())
            sps = step / (time.time() - t0)
            print(f"  step {step:5d}/{args.steps} | train {loss.item():.3f} "
                  f"| val {vl.item():.3f} | {sps:.1f} step/s")

    torch.save({"model": model.state_dict(), "config": vars(cfg)}, args.ckpt)
    print(f"\n已保存 -> {args.ckpt}（最佳 val {best:.3f}，总耗时 {time.time()-t0:.0f}s）")


if __name__ == "__main__":
    main()
