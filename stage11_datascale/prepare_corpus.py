"""prepare_corpus.py —— 11.1 数据对齐:在官方语料上重训 BPE + 全量编码缓存

与 stage7 build_tokens 同管线,只是语料换成 minimind 官方 pretrain_t2t_mini:
  1. 在整份语料上训 BPE(内部取前 2M 字符样本做 merge 统计)
  2. 多进程全文编码(jsonl 每行一条,行间插 <eos> 保文档边界)
  3. 缓存 bpe.json + tokens.pt(训练直接复用)

为什么不沿用 wiki 词表:语料换域(指令式文本、代码、英文混排),旧词表
覆盖会差;分词器跟数据走(minimind 亦自训词表)。代价:val loss 与
stage9 链条不可比——这正是数据实验的性质,可比性转到 20 题/固定裁判。

用法(Spark,~1-1.5 小时):
  ~/llm_study/.venv/bin/python prepare_corpus.py \
      --corpus ~/llm_study/mm_data/pretrain_t2t_mini.jsonl \
      --out ~/llm_study/transformer_study/stage11_datascale/cache_mm
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
from bpe import BPETokenizer, EOS_ID

_TOK = None


def _init(tok):
    global _TOK
    _TOK = tok


def _encode(text):
    return _TOK.encode(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True, help="缓存目录(bpe.json + tokens.pt)")
    ap.add_argument("--merges", type=int, default=3000)
    ap.add_argument("--sample-chars", type=int, default=2_000_000)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bpe_path, ids_path = out / "bpe.json", out / "tokens.pt"
    if bpe_path.exists() and ids_path.exists():
        print("缓存已存在,跳过")
        return

    t0 = time.time()
    texts = []
    with open(args.corpus, encoding="utf-8") as f:
        for ln in f:
            texts.append(json.loads(ln)["text"])
    print(f"{len(texts):,} 行,{sum(len(t) for t in texts):,} 字符(读取 {time.time()-t0:.0f}s)",
          flush=True)

    print(f"训练 BPE({args.merges} merges, 样本 {args.sample_chars:,} 字符)...", flush=True)
    tok = BPETokenizer.train("\n".join(texts), num_merges=args.merges,
                             sample_chars=args.sample_chars, verbose=False)
    tok.save(bpe_path)
    print(f"词表 {len(tok):,} → {bpe_path}", flush=True)

    ctx = mp.get_context("fork")
    t0 = time.time()
    with ctx.Pool(processes=args.workers, initializer=_init, initargs=(tok,)) as pool:
        encoded = pool.map(_encode, texts, chunksize=20)
    ids_list = []
    for e in encoded:
        ids_list.extend(e)
        ids_list.append(EOS_ID)
    ids = torch.tensor(ids_list, dtype=torch.long)
    torch.save(ids, ids_path)
    print(f"编码完成: {len(ids):,} token({time.time()-t0:.0f}s) → {ids_path}", flush=True)


if __name__ == "__main__":
    main()
