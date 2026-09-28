"""gen_arith_mixed.py —— 格式混合训练 + 跨格式迁移测试

**要回答的问题(2026-09-23)**:上一步测出 stage16 的 92.6% **完全是格式绑定的** ——
训练时只见过 `1 2 3 4 + 5 6 7 8 =`,换成 `加` 就 0/800,不逐位空格就 0.6%。

那**把多种格式混着训**,能力会不会跨格式迁移?
  · 会迁移    → 格式多样性是可设计的抓手 → 从零方案有了明确的语料设计原则
  · 不迁移    → 泛化有上限,得用更激进的手段(更多格式 / 更大模型)

**设计**:
  训练集 = **5 种格式等量混合**(每种 12,000 条,共 60,000)
  评测集 = 训练见过的 5 种 + **4 种没见过的**(关键数据)

**输出格式恒定**(逐位空格的数字),这样测的是**输入格式**的泛化,
不是"输出格式适应"。答案都是求和结果,eval_arith.py 会去掉空格再比对。

用法(Spark):
  ~/llm_study/.venv/bin/python gen_arith_mixed.py
"""

import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).parent

# 训练时见过的 5 种格式
SEEN = {
    "S1同款":      lambda a, b, x: "%s + %s =" % (x(a), x(b)),
    "S2汉字算符":   lambda a, b, x: "%s 加 %s =" % (x(a), x(b)),
    "S3问句":      lambda a, b, x: "计算 %s 加 %s 等于多少？" % (x(a), x(b)),
    "S4自然写法":   lambda a, b, x: "%d + %d =" % (a, b),
    "S5三加数":     None,          # 特殊处理:三个加数
}
# 训练时**没见过**的 4 种 —— 跨格式迁移测试的关键
UNSEEN = {
    "U1加上":      lambda a, b, x: "%s 加上 %s 等于几？" % (x(a), x(b)),
    "U2求和":      lambda a, b, x: "求 %s 与 %s 的和。" % (x(a), x(b)),
    "U3冒号无等号": lambda a, b, x: "计算：%s + %s" % (x(a), x(b)),
    "U4自然加汉字": lambda a, b, x: "%d 加 %d =" % (a, b),
}
LEVELS = [2, 3, 4, 5]


def d(n):
    return " ".join(str(n_) for n_ in str(n))


def sample(k, rng):
    a = rng.randint(10 ** (k - 1), 10 ** k - 1)
    b = rng.randint(10 ** (k - 1), 10 ** k - 1)
    return a, b


def make(theme, f, k, rng):
    """返回 (题面, 答案)。

    ⚠️ **答案必须不含空格** —— eval_arith.py 的判据是
        re.sub(r"\s+","",模型输出) == r["a"]
    它只归一化模型输出那一侧。第一版我把答案写成逐位空格的 `d(a+b)`,
    于是 59+84=143 这种**全对的答案**被记成 0/800 —— 五个训练格式全"归零",
    看起来像"混合训练把能力毁了",实际是判据两侧不对称。"""
    if theme == "S5三加数":
        nums = [rng.randint(10 ** (k - 1), 10 ** k - 1) for _ in range(3)]
        return " + ".join(d(n) for n in nums) + " =", str(sum(nums))
    a, b = sample(k, rng)
    return f(a, b, d), str(a + b)


ALL = {**SEEN, **UNSEEN}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train-per-fmt", type=int, default=12000)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    erng = random.Random(args.seed + 1)

    # 训练集:5 种格式等量混合
    n_tr = 0
    with open(HERE / "mixed_train_nt.jsonl", "w", encoding="utf-8") as ftr:
        for th, f in SEEN.items():
            for _ in range(args.n_train_per_fmt):
                k = rng.choice(LEVELS)
                q, a = make(th, f, k, rng)
                ftr.write(json.dumps({"segs": [{"t": "%s %s\n" % (q, a), "loss": 1}]},
                                     ensure_ascii=False) + "\n")
                n_tr += 1
    # 测试集(给 train_mt.py 算 test loss)
    with open(HERE / "mixed_test_nt.jsonl", "w", encoding="utf-8") as fte:
        for th, f in SEEN.items():
            for _ in range(args.n_train_per_fmt // 20):
                k = rng.choice(LEVELS)
                q, a = make(th, f, k, rng)
                fte.write(json.dumps({"segs": [{"t": "%s %s\n" % (q, a), "loss": 1}]},
                                     ensure_ascii=False) + "\n")
    # 评测集:见过的 5 + 没见过的 4,各 800 题
    for th, f in ALL.items():
        with open(HERE / f"mixed_eval_{th}.jsonl", "w", encoding="utf-8") as fev:
            for k in LEVELS:
                for _ in range(args.n_eval):
                    q, a = make(th, f, k, erng)
                    fev.write(json.dumps({"d": k, "theme": th, "q": q, "a": a},
                                         ensure_ascii=False) + "\n")
    print(f"训练 {n_tr} 条(5 种格式等量混合)")
    print(f"评测集:{len(ALL)} 种 × 800 题")
    print("  见过:", " ".join(SEEN))
    print("  没见过:", " ".join(UNSEEN))
    print("\n样例:")
    r2 = random.Random(0)
    for th, f in ALL.items():
        print("  %-12s %s" % (th, make(th, f, 3, r2)[0]))


if __name__ == "__main__":
    main()
