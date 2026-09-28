"""gen_arith_variant.py —— stage16 算术的泛化测试

**为什么(2026-09-23)**:stage17 的泛化测试显示,单一模板训出来的能力
**泛化半径只有几个词宽**(关系推理换措辞 23.5%~93.5%、状态追踪 2.5%)。

而 stage16 那个 **83.5%(5 位加法)也是单模板训的** —— 所以那个数很可能同样
受模板约束。这个测试决定它到底是"学会了加法"还是"学会了那个格式"。

**五种措辞,答案内容不变(还是那个和)**:

  A 训练同款    `1 2 3 4 + 5 6 7 8 = 6 9 1 2`      ← 对照,应该 100%
  B 汉字算符    `1 2 3 4 加 5 6 7 8 = …`
  C 问句        `计算 1 2 3 4 加 5 6 7 8 等于多少？…`
  D 自然写法    `1234 + 5678 = 6912`               ← **关键**:数字不逐位加空格
  E 三个加数    `1 2 + 3 4 + 5 6 = …`              ← 结构变化:训练时只有两个加数

**D 最值得看**:如果它还答得对,说明学的是"加法"而不是"那个空格格式";
如果崩了,说明"逐位对齐"是它依赖的拐杖 —— 而真实文本里数字并不逐位分开。

用法(Spark):
  ~/llm_study/.venv/bin/python gen_arith_variant.py
"""

import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).parent
THEMES = ["A同款", "B汉字算符", "C问句", "D自然写法", "E三加数"]


def d(n):
    return " ".join(str(x) for x in str(n))


def make(theme, k, rng):
    """返回 (题面, 答案字符串)。答案都是求和结果的数字串。"""
    if theme == "E三加数":
        nums = [rng.randint(10 ** (k - 1), 10 ** k - 1) for _ in range(3)]
        ans = sum(nums)
        q = " + ".join(d(x) for x in nums) + " ="
    else:
        a = rng.randint(10 ** (k - 1), 10 ** k - 1)
        b = rng.randint(10 ** (k - 1), 10 ** k - 1)
        ans = a + b
        if theme == "A同款":
            q = "%s + %s =" % (d(a), d(b))
        elif theme == "B汉字算符":
            q = "%s 加 %s =" % (d(a), d(b))
        elif theme == "C问句":
            q = "计算 %s 加 %s 等于多少？" % (d(a), d(b))
        elif theme == "D自然写法":
            q = "%d + %d =" % (a, b)
        else:
            raise ValueError(theme)
    return q, str(ans)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="每档每主题的题数")
    ap.add_argument("--levels", default="2,3,4,5")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    levels = [int(x) for x in args.levels.split(",")]
    rng = random.Random(args.seed)

    for th in THEMES:
        out = HERE / f"arith_variant_{th}.jsonl"
        n = 0
        with open(out, "w", encoding="utf-8") as f:
            for lv in levels:
                for _ in range(args.n):
                    q, a = make(th, lv, rng)
                    f.write(json.dumps({"d": lv, "theme": th, "q": q, "a": a},
                                       ensure_ascii=False) + "\n")
                    n += 1
        print("[%s] %d 题 → %s" % (th, n, out.name))
        for lv in levels[:2]:
            print("    %d 位: %s" % (lv, make(th, lv, random.Random(0))[0]))


if __name__ == "__main__":
    main()
