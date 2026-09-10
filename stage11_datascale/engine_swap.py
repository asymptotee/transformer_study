"""engine_swap.py —— 只换引擎:我们的 26,566 词表 → Rust(tokenizers)执行

bpe.json = {itos: id→token, merges: [[左,右],新token] 按创建序}。
我们的 BPE 语义:单字符初始单元 → 按 merges 顺序贪心合并 → id 用 itos。
Rust 引擎复现同一语义:
  models.BPE(vocab=token→我们的id, merges=[(左,右),...], unk=<unk>)
  pre_tokenizer = 逐 Unicode 字符切分(Split("(?s).", isolated))
  → 编码出的 id 与 python 版应当逐位一致(合并算法同源)

注意:merges 必须传 tuple 对而不是空格文本(我们的合并不少含空格字符,
如 ['.',' ']——空格分隔格式无法表达);vocab 必须带我们的 id 映射。

--verify:小样(2000 行)python 引擎 vs rust 引擎逐位对比 + 计时
--encode:全量分片编码(与 prepare_corpus_chunked 同构,产物格式一致)

用法(Spark):
  ~/llm_study/.venv/bin/python engine_swap.py --verify
  ~/llm_study/.venv/bin/python engine_swap.py --encode
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "stage4_scaling_bpe"))
from bpe import BPETokenizer, EOS_ID                       # noqa: E402

from tokenizers import Tokenizer, models, pre_tokenizers   # noqa: E402


def build_rust(bpe_path):
    b = json.load(open(bpe_path, encoding="utf-8"))
    itos = b["itos"]
    vocab = {t: i for i, t in enumerate(itos)}            # 我们的 id 映射
    merges = [(m[0][0], m[0][1]) for m in b["merges"]]    # (左,右) 元组
    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges,
                               unk_token="<unk>", fuse_unk=False))
    # 逐字符预切分:(?s) 让 . 也匹配换行;isolated = 每个字符独立成单元
    # (tokenizers>=0.20 的 Split 直接吃正则字符串,Regex 包装类已移除)
    tok.pre_tokenizer = pre_tokenizers.Split(r"(?s).", behavior="isolated")
    return tok


def py_ids(py_tok, text):
    return py_tok.encode(text)


def verify(bpe_path, corpus, n_lines=2000):
    py_tok = BPETokenizer.load(bpe_path)
    rust = build_rust(bpe_path)
    texts = []
    with open(corpus, encoding="utf-8") as f:
        for i, ln in enumerate(f):
            if i >= n_lines:
                break
            texts.append(json.loads(ln)["text"])

    t0 = time.time()
    py_all = [py_tok.encode(t) for t in texts]
    t_py = time.time() - t0
    t0 = time.time()
    rs_all = [rust.encode(t).ids for t in texts]
    t_rs = time.time() - t0

    bad = 0
    first = None
    for i, (a, c) in enumerate(zip(py_all, rs_all)):
        if a != c:
            bad += 1
            if first is None:
                first = (i, len(a), len(c))
    print(f"python: {t_py:.1f}s | rust: {t_rs:.1f}s | 加速 {t_py/max(t_rs,1e-9):.0f}×")
    print(f"{n_lines} 行对比: 不一致 {bad} 行" + (f"(首个: 行{first[0]}, "
          f"len {first[1]} vs {first[2]})" if first else " — 全部一致 ✓"))
    # 解码往返一致性
    r0 = rust.decode(rs_all[0])
    print("decode 往返一致:", r0 == texts[0])


def encode(bpe_path, corpus, out_dir, workers=8, window=200_000):
    rust = build_rust(bpe_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids_path = out / "tokens.pt"
    shard_dir = out / "shards"
    shard_dir.mkdir(exist_ok=True)
    done = sorted(shard_dir.glob("shard_*.pt"))
    skip = len(done) * window
    n_shards = len(done)
    total = 0
    t0 = time.time()
    batch = []
    with open(corpus, encoding="utf-8") as f:
        for i, ln in enumerate(f):
            if i < skip:
                continue
            batch.append(json.loads(ln)["text"])
            if len(batch) >= window:
                encs = rust.encode_batch(batch)
                ids = []
                for e in encs:
                    ids.extend(e.ids)
                    ids.append(EOS_ID)
                torch.save(torch.tensor(ids), shard_dir / f"shard_{n_shards:04d}.pt")
                total += len(ids)
                n_shards += 1
                batch = []
                sps = total / max(time.time() - t0, 1e-9)
                print(f"  分片 {n_shards} | 累计 {total/1e6:.0f}M token | "
                      f"{sps/1e6:.2f}M tok/s | 已用 {(time.time()-t0)/60:.0f}min",
                      flush=True)
    if batch:
        encs = rust.encode_batch(batch)
        ids = []
        for e in encs:
            ids.extend(e.ids)
            ids.append(EOS_ID)
        torch.save(torch.tensor(ids), shard_dir / f"shard_{n_shards:04d}.pt")
        total += len(ids)
        n_shards += 1
    print(f"拼接 {n_shards} 分片({total:,} token)...", flush=True)
    big = torch.empty(total, dtype=torch.long)
    off = 0
    for p in sorted(shard_dir.glob("shard_*.pt")):
        t = torch.load(p)
        big[off:off + len(t)] = t
        off += len(t)
        p.unlink()
    torch.save(big, ids_path)
    print(f"→ {ids_path} | 总耗时 {(time.time()-t0)/60:.0f}min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bpe", default=str(Path.home() / "llm_study" /
                     "transformer_study" / "stage11_datascale" /
                     "cache_mm10g" / "bpe.json"))
    ap.add_argument("--corpus", default=str(Path.home() / "llm_study" / "mm_data" /
                     "pretrain_t2t.jsonl"))
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--encode", action="store_true")
    ap.add_argument("--n-lines", type=int, default=2000)
    args = ap.parse_args()
    if args.verify:
        verify(args.bpe, args.corpus, args.n_lines)
    if args.encode:
        out = str(Path(args.bpe).parent)
        encode(args.bpe, args.corpus, out)


if __name__ == "__main__":
    main()
