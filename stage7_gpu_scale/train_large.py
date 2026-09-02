"""train_large.py —— 阶段七：GPU 规模训练（探路 + 全量共用这一个脚本）

和旧脚本（train_best.py）的区别，全是真实 LLM 训练的标准做法：
  · bf16 混合精度（autocast）——GB10 的强项，显存/带宽省一半
  · warmup + cosine 学习率衰减——先小步稳住，再大步学，最后细调
  · BPE 训练/编码一次性完成并缓存到磁盘（纯 Python 编码很慢，只做一次）
  · 按 best val loss 存 checkpoint
  · 训练中定期打印生成样例（含事实性 prompt，探路时直接看"知识注入"）

探路配置（30M 级，~20 分钟）：
  --corpus corpus_wiki.jsonl --d-model 512 --n-layers 8 --n-heads 8 --d-ff 2048
  --steps 2000 --ckpt ckpt_probe.pt

全量配置（100M 级，1 小时量级）：
  --corpus corpus_wiki.jsonl --d-model 768 --n-layers 12 --n-heads 12 --d-ff 3072
  --steps 10000 --ckpt ckpt_large.pt

用法：
  python -u train_large.py ...（参数见上）
"""

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "stage3_gpt"))
sys.path.insert(0, str(HERE.parent / "stage4_scaling_bpe"))
sys.path.insert(0, str(HERE.parent / "stage5_capstone"))
from bpe import BPETokenizer, EOS_ID
from model_gpt import GPT, GPTConfig
from sampling import generate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 多进程编码（纯 Python BPE 对全文逐 merge 扫描，必须并行）----------
_TOK = None


def _init(tok):
    global _TOK
    _TOK = tok


def _encode_article(text):
    return _TOK.encode(text)


def build_tokens(corpus_path, cache_dir, num_merges, workers):
    """BPE 训练（60k 样本）+ 全文多进程编码，缓存到磁盘，返回 (ids, tok)。"""
    bpe_path = cache_dir / "bpe.json"
    ids_path = cache_dir / "tokens.pt"
    if bpe_path.exists() and ids_path.exists():
        tok = BPETokenizer.load(bpe_path)
        ids = torch.load(ids_path)
        print(f"缓存命中: {ids_path}（{len(ids):,} token）")
        return ids, tok

    print(f"训练 BPE（{num_merges} 合并，60k 字符样本）...")
    texts = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    tok = BPETokenizer.train("".join(texts), num_merges=num_merges, verbose=False)
    # train() 内部取前 60000 字符做样本，字符表覆盖全文（生僻字不丢）
    tok.save(bpe_path)
    print(f"词表: {len(tok):,} | 文章: {len(texts):,}，开始多进程编码...")

    ctx = mp.get_context("fork")
    t0 = time.time()
    with ctx.Pool(processes=workers, initializer=_init, initargs=(tok,)) as pool:
        encoded = pool.map(_encode_article, texts, chunksize=50)
    ids_list = []
    for e in encoded:
        ids_list.extend(e)
        ids_list.append(EOS_ID)          # 篇与篇之间插 <eos>，保留文档边界
    ids = torch.tensor(ids_list, dtype=torch.long)
    torch.save(ids, ids_path)
    print(f"编码完成: {len(ids):,} token（{time.time()-t0:.0f}s，"
          f"{len(ids)/max(1, time.time()-t0):,} token/s）→ 已缓存")
    return ids, tok


def get_batch(data, block, batch):
    ix = torch.randint(len(data) - block - 1, (batch,), device=data.device)
    x = torch.stack([data[i:i + block] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block] for i in ix])
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="build_corpus.py 产出的 JSONL")
    ap.add_argument("--cache-dir", default=str(HERE / "cache"))
    ap.add_argument("--merges", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=10, help="编码用进程数")
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--d-ff", type=int, default=2048)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--no-amp", action="store_true", help="关掉 bf16（调试用）")
    ap.add_argument("--device", default="auto", help="cpu/cuda；默认 auto")
    ap.add_argument("--ckpt", default=str(HERE / "ckpt_large.pt"))
    args = ap.parse_args()

    global DEVICE
    if args.device != "auto":
        DEVICE = args.device
    use_amp = DEVICE.startswith("cuda") and not args.no_amp

    torch.manual_seed(42)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    ids, tok = build_tokens(Path(args.corpus), cache_dir, args.merges, args.workers)
    n_val = int(len(ids) * 0.05)
    train_ids = ids[:-n_val].to(DEVICE)
    val_ids = ids[-n_val:].to(DEVICE)

    cfg = GPTConfig(vocab_size=len(tok), pad_id=0,
                    max_len=max(args.block_size, 512),
                    d_model=args.d_model, n_layers=args.n_layers,
                    n_heads=args.n_heads, d_ff=args.d_ff)
    model = GPT(cfg).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"设备: {DEVICE} | AMP: {use_amp} | 参数: {n_params:,}"
          f" | 词表: {len(tok):,} | 训练 token: {len(train_ids):,}"
          f" | block={args.block_size} batch={args.batch_size}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def lr_at(step):
        """warmup 线性爬升 → cosine 衰减到 0（真实 LLM 的标准调度）。"""
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * t))

    def val_loss():
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=use_amp):
            tot, n = 0.0, 0
            for _ in range(4):                      # 4 个 batch 求平均，稳一点
                x, y = get_batch(val_ids, args.block_size, args.batch_size)
                loss = F.cross_entropy(model(x).reshape(-1, len(tok)),
                                       y.reshape(-1), ignore_index=0)
                tot += loss.item() * (x.numel())
                n += x.numel()
        return tot / n

    def show_samples():
        model.eval()
        for seed in ["中国的首都是", "长城位于", "水的化学式是"]:
            out = generate(model, tok, seed, max_new=40, temperature=0.8,
                           top_k=0, top_p=0.9, rep_penalty=1.2)
            print(f"    {seed!r} → {out[:60]!r}")

    best = float("inf")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        opt.param_groups[0]["lr"] = lr_at(step)
        x, y = get_batch(train_ids, args.block_size, args.batch_size)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss = F.cross_entropy(model(x).reshape(-1, len(tok)),
                                   y.reshape(-1), ignore_index=0)
        opt.zero_grad(); loss.backward(); opt.step()

        if step == 1 or step % args.eval_every == 0:
            vl = val_loss()
            sps = step / (time.time() - t0)
            print(f"  step {step:5d}/{args.steps} | train {loss.item():.3f} "
                  f"| val {vl:.3f} | lr {opt.param_groups[0]['lr']:.2e} "
                  f"| {sps:.1f} step/s", flush=True)
            if vl < best:
                best = vl
                torch.save({"model": model.state_dict(), "config": vars(cfg),
                            "step": step}, args.ckpt)
                print(f"    ✓ 新最佳 val {best:.3f} → {args.ckpt}")
            if step > args.warmup:
                show_samples()

    print(f"\n完成: 最佳 val {best:.3f} | 总耗时 {time.time()-t0:.0f}s "
          f"({args.steps/(time.time()-t0):.1f} step/s)")


if __name__ == "__main__":
    main()
