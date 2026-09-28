"""gen_arith.py —— 合成算术数据生成器(难度阶梯 + 逐位对齐格式)

**要回答的问题(2026-09-21)**:我们的模型只会"检索",不会"处理输入" ——
`12 × 8 = 40` 不是算错,是在检索一个看起来合理的数。要逼它学会*处理*,
训练分布必须满足一条:

    语料里**查得到答案** → 教检索;  **查不到** → 逼它计算。

自然文本的答案都能查到,所以要用**程序化生成**的算术:每条输入的组合都是新的。

**格式为什么逐位加空格**:我们的 BPE 把连续数字切成乱块,和"位"不对齐 ——

    1234  → ['12','3','4']        67890 → ['6','7','8','90']
    1 2 3 4 → 每数字 1 token ✅    单个 0-9 本来就各占 1 token

而算术要求**从右往左按位对齐做进位**。token 边界和位边界不重合,等于人为加难度,
可能正是现有模型"有外壳没内核"的原因之一。加空格后边界对齐,且**不用重训分词器**。

产物:
  arith_train_nt.jsonl   训练集(段列表格式,整行算 loss = 继续预训练)
  arith_test_nt.jsonl    测试集(同上格式,给 train_mt.py 算 test loss)
  arith_eval.jsonl       逐档评测集(题面 + 答案,给 eval_arith.py 判分用)

用法(Spark):
  ~/llm_study/.venv/bin/python gen_arith.py
"""

import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).parent


def fmt_digits(n):
    """把数字拆成逐位空格分隔。123 → '1 2 3'"""
    return " ".join(str(n))


def make(d, rng):
    """生成一个 d 位 + d 位的加法题。返回 (题面含答案的整行, 题面, 答案)。"""
    lo, hi = 10 ** (d - 1), 10 ** d - 1
    a, b = rng.randint(lo, hi), rng.randint(lo, hi)
    s = a + b
    q = "%s + %s =" % (fmt_digits(a), fmt_digits(b))
    return "%s %s" % (q, fmt_digits(s)), q, str(s)


def make_unaligned(d, rng):
    """对照组:同样的题,但**不加空格**(用现在的乱切格式)。"""
    lo, hi = 10 ** (d - 1), 10 ** d - 1
    a, b = rng.randint(lo, hi), rng.randint(lo, hi)
    return "%d + %d = %d" % (a, b, a + b), "%d + %d =" % (a, b), str(a + b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", default="1,2,3,4,5", help="难度档:加数的位数")
    ap.add_argument("--n-per-rung", type=int, default=60000, help="每档训练样本数")
    ap.add_argument("--n-eval", type=int, default=200, help="每档评测样本数")
    ap.add_argument("--unaligned", action="store_true",
                    help="对照组:不加逐位空格")
    ap.add_argument("--out-prefix", default="arith")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rungs = [int(x) for x in args.rungs.split(",")]
    gen = make_unaligned if args.unaligned else make
    tag = "_un" if args.unaligned else ""
    rng = random.Random(args.seed)
    erng = random.Random(args.seed + 1)          # 评测用独立随机流,避免重叠

    n_tr = n_te = 0
    with open(HERE / f"{args.out_prefix}_train_nt{tag}.jsonl", "w", encoding="utf-8") as ftr, \
         open(HERE / f"{args.out_prefix}_test_nt{tag}.jsonl", "w", encoding="utf-8") as fte, \
         open(HERE / f"{args.out_prefix}_eval{tag}.jsonl", "w", encoding="utf-8") as fev:
        for d in rungs:
            for _ in range(args.n_per_rung):
                line, _, _ = gen(d, rng)
                ftr.write(json.dumps({"segs": [{"t": line + "\n", "loss": 1}]},
                                     ensure_ascii=False) + "\n")
                n_tr += 1
            for _ in range(args.n_per_rung // 20):        # 测试集 5%
                line, _, _ = gen(d, rng)
                fte.write(json.dumps({"segs": [{"t": line + "\n", "loss": 1}]},
                                     ensure_ascii=False) + "\n")
                n_te += 1
            for _ in range(args.n_eval):                  # 逐档评测集(独立流)
                _, q, a = gen(d, erng)
                fev.write(json.dumps({"d": d, "q": q, "a": a},
                                     ensure_ascii=False) + "\n")
    print("档位 %s | 训练 %d | 测试 %d | 评测 %d×%d"
          % (rungs, n_tr, n_te, args.n_eval, len(rungs)))
    print("→ %s_train_nt%s.jsonl / _test_nt%s.jsonl / _eval%s.jsonl"
          % (args.out_prefix, tag, tag, tag))
    # 打几行样例肉眼确认
    r2 = random.Random(0)
    print("\n样例:")
    for d in rungs:
        print("  %d 位: %s" % (d, gen(d, r2)[0]))


if __name__ == "__main__":
    main()
