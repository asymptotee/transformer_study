"""probe_h07.py —— H07 卡在 0% 是因为「序号本身是数字」吗?

**线索(2026-09-28)**:10 个未见骨架里 9 个都随 N 涨了,H07编号列表 三次都是 0.0%,
而且**不是判据问题** —— 它在算错(15+28 答 343、98+52 答 7110)。

H07 的题面是 `1) 1 5\\n2) 2 8\\n合计：`。**`1)` `2)` 是数字**,而训练库 30 种骨架里,
提示中出现的**每一个数字都是操作数**。所以一个自然的假设:

    模型把序号也当成了操作数 → 它以为有四个数要处理 → 答案形状对、数值错

**这个假设可证伪**:把序号换成字母,其余一字不改。

    H07a  1) X / 2) Y / 合计：    ← 原样(数字序号)      预期 0%
    H07b  a) X / b) Y / 合计：    ← 序号换字母          若假设成立 → 大涨
    H07c  X / Y / 合计：          ← 去掉序号            若 H07b 不涨而这个涨 → 是"多一行前缀"的问题

三组用**同一个 N=30 模型**、**同一批数**。

⚠️ 这只是**探针**,三种格式都没进过训练集,所以 H07b 若涨也说明不了"学会了编号列表",
只说明**挡住它的是那两个数字**,不是那行结构。

用法(Spark):
  ~/llm_study/.venv/bin/python probe_h07.py
"""

import json
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
PY = sys.executable
LEVELS = [2, 3, 4, 5]


def d(n):
    return " ".join(str(x) for x in str(n))


VARIANTS = {
    "H07a数字序号": lambda a, b: "1) %s\n2) %s\n合计：" % (d(a), d(b)),
    "H07b字母序号": lambda a, b: "a) %s\nb) %s\n合计：" % (d(a), d(b)),
    "H07c无序号":   lambda a, b: "%s\n%s\n合计：" % (d(a), d(b)),
}


def main():
    rng = random.Random(77)
    for name, f in VARIANTS.items():
        p = HERE / f"probe_h07_{name}.jsonl"
        with open(p, "w", encoding="utf-8") as fo:
            for k in LEVELS:
                for _ in range(100):
                    a = rng.randint(10 ** (k - 1), 10 ** k - 1)
                    b = rng.randint(10 ** (k - 1), 10 ** k - 1)
                    fo.write(json.dumps({"d": k, "theme": name, "q": f(a, b),
                                         "a": str(a + b)},
                                        ensure_ascii=False) + "\n")
    for name in VARIANTS:
        subprocess.run([PY, "-u", "eval_arith.py",
                        "--ckpt", "ckpt_16_bank_n30.pt",
                        "--eval-file", f"probe_h07_{name}.jsonl",
                        "--label", f"n30_{name}",
                        "--json-out", f"results/probe_h07_{name}.json"],
                       cwd=HERE, check=False)
    print("\n=== 汇总(同一个 N=30 模型)===")
    for name in VARIANTS:
        p = HERE / "results" / f"probe_h07_{name}.json"
        if not p.exists():
            print("  %-14s 缺结果" % name)
            continue
        dd = json.loads(p.read_text(encoding="utf-8"))
        tot = [0, 0, 0, 0]
        for v in dd["by_rung"].values():
            tot = [tot[i] + v[i] for i in range(4)]
        print("  %-14s %d/%d = %.1f%%" % (name, tot[0], tot[3],
                                          100.0 * tot[0] / tot[3]))
    print("\n样例(第一题):")
    for name in VARIANTS:
        p = HERE / "results" / f"probe_h07_{name}.json"
        if p.exists():
            r = json.loads(p.read_text(encoding="utf-8"))["rows"][0]
            print("  %-14s %r → 期望 %s 实得 %r"
                  % (name, r["q"], r["want"], r["got"][:14]))


if __name__ == "__main__":
    main()
