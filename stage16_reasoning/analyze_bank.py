"""analyze_bank.py —— 把 supervisor15.sh 的 81 份结果读成一条曲线

**为什么要有这个文件**:81 个 JSON 靠眼睛看会看错重点。这张表要一句话回答:

    训练里放 N 种骨架时,没见过的骨架能到多少?(N = 6 / 15 / 30)

同时打两列**对照**,否则结论不可信:

  · **见过的骨架**(对照) —— 如果 N=30 在这里就垮了,那是**欠训练**(总数据量随 N 涨
    但步数固定),不是"骨架太多学不动"。这两件事必须分开。
  · **layout 那一列** —— 曲线若上翘,要看是 layout 在涨还是只有 word 在涨
    (word 上一轮已证明有用,layout 才是这次真正的问题)

用法(Spark):
  ~/llm_study/.venv/bin/python analyze_bank.py
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent


def kinds():
    p = HERE / "bank_kinds.txt"
    out = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").split("\n"):
            parts = line.split()
            if len(parts) == 2:
                out[parts[0]] = parts[1]
    return out


def load(N, name):
    """→ [正确, 长度对, 崩坏, 总数],文件不存在返回 None"""
    p = HERE / "results" / f"bank_{N}_{name}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    tot = [0, 0, 0, 0]
    for v in d["by_rung"].values():
        tot = [tot[i] + v[i] for i in range(4)]
    return tot


def pct(a, n):
    return "%.1f%%" % (100.0 * a / n) if n else "-"


def parse_runs(spec):
    """'n6:6,n15:15,n30:30' → [('n6',6), ...]  (结果文件名前缀 : 训练骨架数)"""
    out = []
    for part in spec.split(","):
        lab, n = part.split(":")
        out.append((lab, int(n)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="n6:6,n15:15,n30:30",
                    help="逗号分隔的 标签:训练骨架数,决定列顺序")
    args = ap.parse_args()

    KIND = kinds()
    names = (HERE / "bank_eval_manifest.txt").read_text(
        encoding="utf-8").split()
    seen, held = names[:30], names[30:]
    assert len(held) == 10, f"manifest 形状不对({len(names)} 个)"
    RUNS = parse_runs(args.runs)
    NS = [lab for lab, _ in RUNS]

    print("=== 骨架数量 → 未见骨架迁移率 ===")
    print("跑       骨架数  未见10 个        见过的          见过的layout")
    for lab, skel in RUNS:
        sv = [load(lab, t) for t in seen[:skel]]
        hv = [load(lab, t) for t in held]
        if not any(hv):
            print("%-8s %5d   (缺结果)" % (lab, skel))
            continue
        sa = sum(x[0] for x in sv if x)
        sn = sum(x[3] for x in sv if x)
        la = sum(x[0] for x, t in zip(sv, seen[:skel]) if x and KIND.get(t) == "layout")
        ln = sum(x[3] for x, t in zip(sv, seen[:skel]) if x and KIND.get(t) == "layout")
        ha = sum(x[0] for x in hv if x)
        hn = sum(x[3] for x in hv if x)
        print("%-8s %5d %9s %7s %9s %7s %9s %7s"
              % (lab, skel, "%d/%d" % (ha, hn), pct(ha, hn), "%d/%d" % (sa, sn),
                 pct(sa, sn), "%d/%d" % (la, ln), pct(la, ln) if ln else "-"))

    hdr = "%-12s %-8s" % ("骨架", "kind") + "".join("%12s" % lab for lab in NS)

    print("\n--- 逐个未见骨架(★ = 所有训练都没见过)---")
    print(hdr)
    for t in held:
        row = "%-12s %-8s" % (t, KIND.get(t, "?"))
        for lab in NS:
            x = load(lab, t)
            row += "%12s" % (f"{x[0]}/{x[3]} = {pct(x[0], x[3])}" if x else "-")
        print(row)

    # ★ 真正的对照:BANK[0:6] 每次训练**都训过**。拿它跨 run 比,才能回答
    #   "多加骨架/多训是不是只是欠训练" —— 上面那个"见过的"平均值会骗人(骨架难度不同)。
    print("\n--- ★ 共享骨架 T01–T06(所有 run 都训过)= 欠训练对照 ---")
    print(hdr)
    for t in seen[:6]:
        row = "%-12s %-8s" % (t, KIND.get(t, "?"))
        for lab in NS:
            x = load(lab, t)
            row += "%12s" % (f"{x[0]}/{x[3]} = {pct(x[0], x[3])}" if x else "-")
        print(row)
    print("%-12s %-8s" % ("合计", "") + "".join(
        "%12s" % (lambda v: f"{sum(x[0] for x in v if x)}/{sum(x[3] for x in v if x)}"
                  if any(v) else "-")([load(lab, t) for t in seen[:6]])
        for lab in NS))

    print("\n--- 全部见过的骨架 ---")
    print(hdr)
    for t in seen:
        row = "%-12s %-8s" % (t, KIND.get(t, "?"))
        for lab in NS:
            x = load(lab, t)
            row += "%12s" % (f"{x[0]}/{x[3]} = {pct(x[0], x[3])}" if x else "-")
        print(row)


if __name__ == "__main__":
    main()
