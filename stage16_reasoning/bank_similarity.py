"""bank_similarity.py —— 迁移靠的是「多样性」还是「有结构近亲」?

**为什么必须做这个分析(2026-09-28)**。supervisor15 的头条数字是
"未见骨架 7.7% → 33.5% → 45.5%"。但看逐条结果,驱动机制有两种可能:

  (a) **多样性红利** —— 骨架见得多了,学会了"处理数字"这件事本身 → 数字该随 N 平滑涨
  (b) **近亲覆盖**   —— 只在"训练里有结构近亲"时才迁移 → 数字由**最像的那个骨架**决定

**两个方向的极端证据**:
  · H05问在前: N=6/15 都是 **0.0%**,N=30 突然 **74.2%**
    —— 而 N=30 的训练库里刚好有 `T23问在前`,两者**只差一个词**("两个加数是" / "已知两个加数是")
    → 这一跳**不像**是"多样性积累",像是**撞上了近亲**
  · H07编号列表: 三次都是 **0.0%**,而训练库里有 `T20多行标签`(同样多行、同样两数一行再跟一个答案标记)
    → 若 (b) 成立,H07 本该也跳起来。**它没有。**

**所以这一页要算的**:把每个未见骨架和**训练库里最像的那个**做相似度,
再看相似度和 N=30 准确率对不对得上。

  · 对得上 → (b):语料得**按目标结构枚举**,"多放点"是撞运气
  · 对不上 → (a):多样性的量本身在起作用,H07 是个例外(需要单独解释)

用法(Spark):
  ~/llm_study/.venv/bin/python bank_similarity.py [--show H07编号列表 H05问在前 T20多行标签]
"""

import argparse
import json
import re
from pathlib import Path

from gen_arith_bank import BANK, HELD, ALL

HERE = Path(__file__).parent


def shape(name):
    """骨架的「形状」:把数字全换成 #、空格压掉 —— 只留格式的骨架"""
    _kind, fn = ALL[name]
    s = fn(11, 22)
    s = re.sub(r"\d", "#", s)
    return re.sub(r"\s+", " ", s).strip()


def bigrams(s):
    s = re.sub(r"\s+", "", s)
    return {s[i:i + 2] for i in range(len(s) - 1)}


def jaccard(a, b):
    A, B = bigrams(a), bigrams(b)
    return len(A & B) / len(A | B) if (A | B) else 0.0


def load(N, name):
    p = HERE / "results" / f"bank_n{N}_{name}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    tot = [0, 0, 0, 0]
    for v in d["by_rung"].values():
        tot = [tot[i] + v[i] for i in range(4)]
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", nargs="*", default=[],
                    help="原样打印这些骨架的模型输出(用于确认 0.0% 不是判据问题)")
    args = ap.parse_args()

    for name in args.show:
        d = json.loads((HERE / "results" /
                        f"bank_n30_{name}.json").read_text(encoding="utf-8"))
        print(f"=== {name} ===  (N=30 模型)")
        for r in d["rows"][:5]:
            print("   题面 %r" % r["q"])
            print("        期望 %-6r 实得 %-14r %s"
                  % (r["want"], r["got"][:14], "✓" if r["exact"] else "✗"))
        print()

    print("=== 未见骨架 vs 训练库里最像的那个(N=30 训练库 = BANK[:30]) ===")
    print("%-12s %5s %-14s %-7s %7s" % ("未见骨架", "kind", "最近的训练骨架",
                                        "相似度", "N=30"))
    rows = []
    for name, kind, _fn in HELD:
        sh = shape(name)
        best, bn = 0.0, "-"
        for tn, _k, _f in BANK:
            j = jaccard(sh, shape(tn))
            if j > best:
                best, bn = j, tn
        x = load(30, name)
        acc = (x[0] / x[3]) if x else None
        rows.append((name, kind, bn, best, acc))
        print("%-12s %5s %-14s %7.3f %6s"
              % (name, kind, bn, best,
                 "%.1f%%" % (100 * acc) if acc is not None else "-"))

    xs = [r[3] for r in rows]
    ys = [r[4] for r in rows if r[4] is not None]
    if len(ys) == len(xs) and len(xs) > 2:
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        vx = sum((a - mx) ** 2 for a in xs) ** 0.5
        vy = sum((b - my) ** 2 for b in ys) ** 0.5
        r = cov / (vx * vy) if vx * vy else 0.0
        print("\n  相关系数 r = %.3f  (n=%d,草图级 —— 10 个点,别当结论用)"
              % (r, len(xs)))
        print("  r 高 → 迁移由「有没有近亲」决定(覆盖);r 低 → 是量的红利(多样性)")


if __name__ == "__main__":
    main()
