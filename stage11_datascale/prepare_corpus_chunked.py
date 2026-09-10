"""prepare_corpus_chunked.py —— 11.2 编码(流式分片版,修 OOM)

上一版(整库 pool.map)在 10GB 语料上被 OOM killer 杀(SIGKILL 137):
pool.map 一次性收回全部 2.1B token 的 Python 列表 ≈ 60GB+。这版:

  1. 词表:BPE 已训好(bpe.json 存在)就直接复用,否则先训
  2. 编码:按窗口(6 万行 ≈ 4500 万 token)分批 pool.map,每窗立刻
     转 tensor 存分片 shard_%04d.pt,内存峰值被钉在窗口量级
  3. 拼接:预分配 total×int64 大张量,逐分片填入(峰值 ≈ 17GB),
     存 tokens.pt 后删分片

进度:每窗打印累计 token 数与速度;中途被杀可从分片数续跑
(--resume 跳过已存在的分片,重扫到对应行数再续)。

用法(Spark,~6h):
  ~/llm_study/.venv/bin/python prepare_corpus_chunked.py \
      --corpus ~/llm_study/mm_data/pretrain_t2t.jsonl --out cache_mm10g
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "stage4_scaling_bpe"))
from bpe import BPETokenizer, EOS_ID                        # noqa: E402

_TOK = None
WINDOW = 60_000          # 行/窗


def _init(tok):
    global _TOK
    _TOK = tok


def _enc(t):
    return _TOK.encode(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--merges", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bpe_path, ids_path = out / "bpe.json", out / "tokens.pt"
    if ids_path.exists():
        print("tokens.pt 已存在,跳过")
        return

    if bpe_path.exists():
        tok = BPETokenizer.load(bpe_path)
        print(f"复用词表 {len(tok)}({bpe_path})", flush=True)
    else:
        t0 = time.time()
        texts = []
        with open(args.corpus, encoding="utf-8") as f:
            for ln in f:
                texts.append(json.loads(ln)["text"])
        print(f"读入 {len(texts):,} 行({time.time()-t0:.0f}s),训 BPE...", flush=True)
        tok = BPETokenizer.train("\n".join(texts), num_merges=args.merges,
                                 sample_chars=2_000_000, verbose=False)
        tok.save(bpe_path)
        print(f"词表 {len(tok)} → {bpe_path}", flush=True)
        del texts

    # 流式分窗编码
    ctx = mp.get_context("fork")
    shard_dir = out / "shards"
    shard_dir.mkdir(exist_ok=True)
    done_shards = sorted(shard_dir.glob("shard_*.pt"))
    skip_lines = len(done_shards) * WINDOW      # 续跑:跳过已完成的行
    n_shards = len(done_shards)
    total = 0
    t0 = time.time()
    window = []
    with open(args.corpus, encoding="utf-8") as f:
        for i, ln in enumerate(f):
            if i < skip_lines:
                continue
            window.append(json.loads(ln)["text"])
            if len(window) >= WINDOW:
                with ctx.Pool(processes=args.workers, initializer=_init,
                              initargs=(tok,)) as pool:
                    enc = pool.map(_enc, window, chunksize=500)
                ids = []
                for e in enc:
                    ids.extend(e)
                    ids.append(EOS_ID)
                torch.save(torch.tensor(ids), shard_dir / f"shard_{n_shards:04d}.pt")
                total += len(ids)
                n_shards += 1
                window = []
                sps = total / max(time.time() - t0, 1)
                print(f"  分片 {n_shards} | 累计 {total/1e6:.0f}M token | "
                      f"{sps/1e6:.2f}M tok/s | 预计剩 "
                      f"{(3.4e9/1.6 - total) / max(sps,1) / 60:.0f}min",
                      flush=True)
    if window:                                # 尾窗
        with ctx.Pool(processes=args.workers, initializer=_init,
                      initargs=(tok,)) as pool:
            enc = pool.map(_enc, window, chunksize=500)
        ids = []
        for e in enc:
            ids.extend(e)
            ids.append(EOS_ID)
        torch.save(torch.tensor(ids), shard_dir / f"shard_{n_shards:04d}.pt")
        total += len(ids)
        n_shards += 1

    # 拼接(逐分片填入预分配张量,峰值可控)
    print(f"编码完成: {total:,} token,{n_shards} 分片,拼接中...", flush=True)
    big = torch.empty(total, dtype=torch.long)
    off = 0
    for p in sorted(shard_dir.glob("shard_*.pt")):
        t = torch.load(p)
        big[off:off + len(t)] = t
        off += len(t)
        p.unlink()
    torch.save(big, ids_path)
    print(f"→ {ids_path}({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
