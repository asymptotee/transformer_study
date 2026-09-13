"""make_freq.py —— 13.1:用 10GB 语料实测每题的答案词频(替代"拍脑袋标高频")

旧 20 题的"高频/低频"是按 wiki 语料凭感觉标的;语料换成官方 10GB 后标签
已过期。本脚本对每题的标准答案(第一个)做**全语料出现次数统计**,再分档:

  high  ≥ 500        语料里高频出现(模型有充分机会记住)
  mid   50 ~ 499     中频
  low   < 50         低频(接近"知道算你厉害")

产出 qa_v2_freq.jsonl:原字段 + freq(int) + bin(high/mid/low) + occurrences。
原始计数一并保留 —— 阈值以后可重分档,不用重扫。

用法(Spark,一次扫描约几分钟):
  ~/llm_study/.venv/bin/python make_freq.py \
      --corpus ~/llm_study/mm_data/pretrain_t2t.jsonl \
      --in qa_v2.jsonl --out qa_v2_freq.jsonl
"""

import argparse
import json
import re
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--in-file", default="qa_v2.jsonl")
    ap.add_argument("--out", default="qa_v2_freq.jsonl")
    ap.add_argument("--high", type=int, default=500)
    ap.add_argument("--mid", type=int, default=50)
    args = ap.parse_args()

    rows = [json.loads(ln) for ln in open(args.in_file, encoding="utf-8")]
    # 每题用第一个答案做频率统计;多题共用同一答案时共享计数
    keys = []
    for r in rows:
        keys.append(r["a"][0])
    uniq = sorted(set(keys), key=len, reverse=True)   # 长串优先,防子串吞并
    pattern = re.compile("|".join(re.escape(k) for k in uniq))
    counts = {k: 0 for k in uniq}

    t0 = time.time()
    n_lines = 0
    with open(args.corpus, encoding="utf-8") as f:
        for ln in f:
            n_lines += 1
            for m in pattern.finditer(ln):
                counts[m.group()] += 1
            if n_lines % 2_000_000 == 0:
                print(f"  {n_lines/1e6:.0f}M 行 | {time.time()-t0:.0f}s", flush=True)

    def bin_of(c):
        return "high" if c >= args.high else ("mid" if c >= args.mid else "low")

    out_rows = []
    for r, k in zip(rows, keys):
        r2 = dict(r)
        r2["freq"] = counts[k]
        r2["bin"] = bin_of(counts[k])
        out_rows.append(r2)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"扫描 {n_lines:,} 行 → {args.out}({time.time()-t0:.0f}s)")
    from collections import Counter
    print("分档统计:", dict(Counter(r["bin"] for r in out_rows)))
    lo = [r for r in out_rows if r["bin"] == "low"][:8]
    print("低频样例:", [(r["a"][0], r["freq"]) for r in lo])


if __name__ == "__main__":
    main()
