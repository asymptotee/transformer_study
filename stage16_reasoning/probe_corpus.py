"""probe_corpus.py —— 量预训练语料里有多少"推理底料"

**要回答的问题(2026-09-22)**:主流框架说"推理能力的根基在预训练,后训练只是唤醒"。
而我们实测到基座对 2 位加法是 **0%** —— 那是"**没有打底**"(语料里就没有),
还是"有底但调用不出来"?

   · 语料里接近 0  → 没有打底 → 后训练只能"从零建" → 于是覆盖别的东西(实测过三次)
   · 语料里有不少却没学会 → 是"调用不出来" → 后训练(第二阶段)就够

这个区分决定整条路线,而且 10 分钟就能拿到。

**方法**:均匀采样(不能只读前缀 —— stage14 踩过:语料按来源分块排,
前 20 万行多轮占 35.5%、含 tool 0 条,而全文件是 22.2% / 76,574 条)。
所以按**分片**采样,并把每片的数单独报出来,**顺带能看出分块结构**。

用法(Spark):
  ~/llm_study/.venv/bin/python probe_corpus.py --n 8000 --shards 20
"""

import argparse
import json
import re
from pathlib import Path


def rate(hits, n):
    return "%5.1f%%" % (100.0 * hits / max(n, 1))


MATH_KW = re.compile(r"方程|函数|求解|证明|计算|公式|数学|几何|代数|积分|导数|矩阵|概率|统计|"
                     r"equation|formula|theorem|proof|algebra|calculus|matrix|probability")
CODE_KW = re.compile(r"\bdef |\bclass |\bimport |\breturn\b|function\s*\(|#include|public static|"
                     r"```|</?(div|span|p|html)>")
REASON_KW = re.compile(r"因为|所以|因此|首先|其次|然后|最后|推导|推理|步骤|由此可得|"
                       r"therefore|because|thus|first.*then|step \d|reasoning")
NUM_DENSE = re.compile(r"[\d.]+\s*[-+*/×÷=]\s*[\d.]+")   # 至少一处算式


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(Path.home() / "llm_study" / "mm_data"
                                         / "pretrain_t2t.jsonl"))
    ap.add_argument("--n", type=int, default=8000, help="总采样条数")
    ap.add_argument("--shards", type=int, default=20, help="分片数(看分块结构)")
    ap.add_argument("--max-chars", type=int, default=4000, help="每条只看前 N 字符")
    args = ap.parse_args()

    # 第一次扫描:数总行数,好做均匀分片(不能只读前缀)
    total = sum(1 for _ in open(args.src, encoding="utf-8"))
    per_shard = max(1, args.n // args.shards)
    step = max(1, total // args.n)
    print(f"语料 {total:,} 行 | 均匀采样 {args.n} 条(每 {step} 行取一条)| "
          f"分 {args.shards} 片", flush=True)

    tot = {"math": 0, "code": 0, "reason": 0, "numexpr": 0}
    chars = {"math": 0, "code": 0, "reason": 0, "numexpr": 0, "all": 0}
    shard_stats = []
    cur = {"n": 0, **{k: 0 for k in tot}}
    n_read = 0

    with open(args.src, encoding="utf-8") as f:
        for i, ln in enumerate(f):
            if i % step:
                continue
            try:
                t = json.loads(ln).get("text") or ""
            except Exception:
                continue
            t = t[:args.max_chars]
            if not t:
                continue
            n_read += 1
            chars["all"] += len(t)
            for key, pat in (("math", MATH_KW), ("code", CODE_KW),
                             ("reason", REASON_KW), ("numexpr", NUM_DENSE)):
                if pat.search(t):
                    tot[key] += 1
                    cur[key] += 1
                    chars[key] += len(t)
            cur["n"] += 1
            if cur["n"] >= per_shard:
                shard_stats.append(cur)
                cur = {"n": 0, **{k: 0 for k in tot}}
            if n_read >= args.n:
                break

    n = n_read
    print(f"\n=== 命中率({n} 条,按条)===")
    print(f"  数学词汇   {rate(tot['math'], n)}   ({tot['math']}/{n})")
    print(f"  代码特征   {rate(tot['code'], n)}   ({tot['code']}/{n})")
    print(f"  推理连接词 {rate(tot['reason'], n)}   ({tot['reason']}/{n})")
    print(f"  含算式     {rate(tot['numexpr'], n)}   ({tot['numexpr']}/{n})")
    print(f"  任一类     {rate(sum(1 for _ in [0]), n) if False else ''}")
    print(f"\n=== 字符占比(命中文本的字符数 / 全部字符)===")
    for k in tot:
        print(f"  {k:9s} {rate(chars[k], chars['all'])}")
    print(f"  总字符     {chars['all']:,}")

    print(f"\n=== 分片(看有没有分块结构)===")
    print("  %-4s %6s %8s %8s %8s %8s" % ("片", "条数", "数学", "代码", "推理", "算式"))
    for i, s in enumerate(shard_stats):
        if s["n"] == 0:
            continue
        print("  %-4d %6d %8s %8s %8s %8s"
              % (i, s["n"], rate(s["math"], s["n"]), rate(s["code"], s["n"]),
                 rate(s["reason"], s["n"]), rate(s["numexpr"], s["n"])))
    lo = [i for i, s in enumerate(shard_stats) if s["n"] and s["math"] / max(s["n"], 1) < 0.02]
    hi = [i for i, s in enumerate(shard_stats) if s["n"] and s["math"] / max(s["n"], 1) > 0.15]
    print(f"\n  数学含量 <2% 的片: {lo[:10]}{' …' if len(lo) > 10 else ''}  (共 {len(lo)})")
    print(f"  数学含量 >15% 的片: {hi[:10]}{' …' if len(hi) > 10 else ''}  (共 {len(hi)})")


if __name__ == "__main__":
    main()
