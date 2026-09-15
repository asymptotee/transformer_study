"""rejudge.py —— 用当前 judge.py 复算历史评估结果(离线,不需要 GPU)

历史 JSON 里存了每题模型原始输出(out),所以判分规则一改,直接重判即可
拿到新口径的全套数字 —— 这是"修判分"这件事可验证的前提(见 judge.py 文件头)。

同时打印**新旧差异清单**(哪些题从真阳变假阳/反之),方便人工审计改得对不对。

用法:
  python rejudge.py results_v2/11_5_final.json            # 单文件
  python rejudge.py results_v2/*.json                     # 批量对比表
  python rejudge.py results_v2/11_5_final.json --diff      # 打印变化明细
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from judge import judge                                  # noqa: E402


def rejudge_one(path, show_diff=False, field="real"):
    rep = json.load(open(path, encoding="utf-8"))
    out = {}
    for fmt, d in rep["formats"].items():
        rows = d["rows"]
        new = []
        for x in rows:
            raw, real, susp = judge(x["out"], x["a"], x["q"])
            new.append({**x, "raw": raw, "real": real, "suspect": susp})
        chased = [x for x, y in zip(new, rows) if x[field] != y[field]]
        out[fmt] = {
            "old": sum(1 for x in rows if x[field]),
            "new": sum(1 for x in new if x[field]),
            "n": len(rows),
            "chased": chased,
        }
        if show_diff and chased:
            print(f"\n--- [{fmt}] {Path(path).stem} 判定变化的题:")
            for y in chased:
                old = next(x for x in rows if x["q"] == y["q"])
                arrow = "真阳→假阳" if old[field] and not y[field] else "假阳→真阳"
                print(f"  {arrow} | {y['q']} | 期望 {y['a']}")
                print(f"     {y['out'][:100]!r}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--diff", action="store_true", help="打印判定变化的明细")
    ap.add_argument("--field", default="real", choices=["real", "raw"])
    args = ap.parse_args()

    files = []
    for p in args.paths:
        files += sorted(Path().glob(p)) if "*" in p else [Path(p)]
    if len(files) == 1:
        res = rejudge_one(files[0], args.diff, args.field)
        for fmt, d in res.items():
            print(f"[{fmt}] {args.field}: {d['old']} → {d['new']} / {d['n']}"
                  f"  ({d['new']-d['old']:+d})")
        return

    print(f"{'run':<16}{'fmt':<7}{'旧':>5}{'新':>5}{'差':>5}   (口径 {args.field})")
    for f in files:
        for fmt, d in rejudge_one(f, args.diff, args.field).items():
            print(f"{Path(f).stem:<16}{fmt:<7}{d['old']:>5}{d['new']:>5}"
                  f"{d['new']-d['old']:>+5}")


if __name__ == "__main__":
    main()
