"""gen_arith_struct.py —— 结构多样性:换骨架能不能泛化?

**要回答的问题(2026-09-23)**:上一轮格式混合的结果是 —— **换词能过(U1 84%),
换结构过不去(U2 3.6%、U3 0%)**。但那一轮训练里 5 种格式的**骨架其实相同**
(全都是 `X <算子> Y =`)。

**所以这一轮问**:如果训练时就把"骨架完全不同"的格式混进去,
**没见过的骨架**能不能迁移?

  · 能迁移 → 语料里放十几种结构就够,不用穷举 → 从零方案的语料设计简单
  · 不能   → 每种结构各学各的 → **得枚举目标场景的说法**(劳动密集,但方案完全不同)

**两种结果的方案差别极大 —— 这 25 分钟省的是"语料走错方向"的 30 小时。**

训练(6 种骨架:纯符号 / 无等号 / 求…的和 / 疑问 / 自然写法 / 应用题):

    1 2 3 4 + 5 6 7 8 = 6912
    计算：1 2 3 4 + 5 6 7 8，答案是 6912
    求 1 2 3 4 与 5 6 7 8 的和，结果是 6912
    1 2 3 4 加上 5 6 7 8 等于几？6912
    1234 加 5678 = 6912
    小明有 1234 元，又得到 5678 元，一共多少元？6912

评测(**训练时没见过的 4 种骨架**):

    甲乙两地相距 1234 米，又延长 5678 米，现在多长？6912      ← 换叙事的应用题
    1 2 3 4 与 5 6 7 8 之和是？6912                        ← 倒装疑问
    请把 1 2 3 4 和 5 6 7 8 相加。6912                      ← 祈使
    (1 2 3 4) + (5 6 7 8) = 6912                          ← 带括号

⚠️ 答案**不含空格** —— eval_arith.py 只归一化模型输出那一侧,两边不对称会
把全对判成全错(踩过一次,见 README 的坑 3)。

用法(Spark):
  ~/llm_study/.venv/bin/python gen_arith_struct.py
"""

import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).parent
LEVELS = [2, 3, 4, 5]


def d(n):
    """逐位空格(输入侧用;token 边界与位边界对齐)"""
    return " ".join(str(x) for x in str(n))


# (名称, 生成 prefix 的函数) —— 训练用;答案直接跟在 prefix 后
SEEN = {
    "T1纯符号":   lambda a, b: "%s + %s = " % (d(a), d(b)),
    "T2无等号":   lambda a, b: "计算：%s + %s，答案是 " % (d(a), d(b)),
    "T3求和":     lambda a, b: "求 %s 与 %s 的和，结果是 " % (d(a), d(b)),
    "T4疑问":     lambda a, b: "%s 加上 %s 等于几？" % (d(a), d(b)),
    "T5自然写法": lambda a, b: "%d 加 %d = " % (a, b),
    "T6应用题":   lambda a, b: "小明有 %d 元，又得到 %d 元，一共多少元？" % (a, b),
}
# 训练时**没见过**的 4 种骨架 ★
UNSEEN = {
    "V1新叙事":   lambda a, b: "甲乙两地相距 %d 米，又延长 %d 米，现在多长？" % (a, b),
    "V2倒装":     lambda a, b: "%s 与 %s 之和是？" % (d(a), d(b)),
    "V3祈使":     lambda a, b: "请把 %s 和 %s 相加。" % (d(a), d(b)),
    "V4括号":     lambda a, b: "(%s) + (%s) = " % (d(a), d(b)),
}
ALL = {**SEEN, **UNSEEN}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train-per-fmt", type=int, default=10000)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--seed", type=int, default=31)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    erng = random.Random(args.seed + 1)

    def pair(k, r):
        return (r.randint(10 ** (k - 1), 10 ** k - 1),
                r.randint(10 ** (k - 1), 10 ** k - 1))

    n_tr = 0
    with open(HERE / "struct_train_nt.jsonl", "w", encoding="utf-8") as ftr, \
         open(HERE / "struct_test_nt.jsonl", "w", encoding="utf-8") as fte:
        for name, f in SEEN.items():
            for i in range(args.n_train_per_fmt):
                a, b = pair(rng.choice(LEVELS), rng)
                line = f(a, b) + str(a + b)
                tgt = ftr if i % 20 else fte
                tgt.write(json.dumps({"segs": [{"t": line + "\n", "loss": 1}]},
                                     ensure_ascii=False) + "\n")
                if i % 20 == 0:
                    n_tr += 1
    for name, f in ALL.items():
        with open(HERE / f"struct_eval_{name}.jsonl", "w", encoding="utf-8") as fev:
            for k in LEVELS:
                for _ in range(args.n_eval):
                    a, b = pair(k, erng)
                    fev.write(json.dumps({"d": k, "theme": name,
                                          "q": f(a, b), "a": str(a + b)},
                                         ensure_ascii=False) + "\n")
    print(f"训练 6 种骨架 × {args.n_train_per_fmt}(其中 1/20 进测试集)")
    print("  见过:", " ".join(SEEN))
    print("  ★没见过★:", " ".join(UNSEEN))
    print("\n样例:")
    r2 = random.Random(0)
    for name, f in ALL.items():
        a, b = 1234, 5678
        print("  %-12s %s%s" % (name, f(a, b), a + b))


if __name__ == "__main__":
    main()
