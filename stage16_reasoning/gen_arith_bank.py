"""gen_arith_bank.py —— 骨架数量 → 未见骨架的迁移率(曲线)

**要回答的问题(2026-09-28)**:从零方案的语料制备要花多少钱,取决于一个问题:

    语料里得放多少种「骨架」,没见过的骨架才能迁移?

  · 曲线上翘、K 小(≈20)  → **手写骨架就够了**,不用起 LLM 批量生成
  · 曲线一直平            → 骨架绑定是硬约束,得换方案(枚举目标场景 / LLM 海量生成)

**上一轮(gen_arith_struct.py)**:6 种骨架 → 未见骨架 1–7%(V3 例外,见 README)。
但那只是曲线上的**一个点**。这一点不能回答"要多少",只能回答"6 不够"。

## 设计:骨架库做成**嵌套前缀**

    BANK[0:6]   → N=6   (★ 就是原来的 T1–T6,与上一轮**逐字相同**)
    BANK[0:15]  → N=15
    BANK[0:30]  → N=30
    HELD[0:10]  → 永远不训

三次训练是**前缀关系**,所以唯一变化的量就是骨架数量。而且 N=6 那一点
若能复现上一轮的数字,说明整套流程是稳的(**内部对照**)。

**控制变量**:每个骨架固定 1 万条(所以总数据量 6万/15万/30万),步数固定 5000。
总数据量随 N 涨,所以必须看**见过的骨架**的准确率当对照 —— 如果 N=30 在见过的
骨架上就垮了,那是欠训练,不是"骨架太多学不动"。

## 骨架的两种多样性(要分开看)

给每个骨架打一个标签,因为它们的含义不同:

    word   —— 只换词/标点,句子骨架不变       (上一轮已证明:这种多样性有用,U1 84%)
    layout —— 换句子骨架/信息布局/数字表示     (上一轮已证明:这种**没**迁移)

如果只看"总数到 30 就通了",却不说其中多少是 layout,结论会误导。

## 答案格式

**答案不含空格** —— eval_arith.py 的判据只归一化模型输出那一侧,两边不对称会把
全对判成全错(踩过一次,见 README 坑 3)。

用法(Spark):
  ~/llm_study/.venv/bin/python gen_arith_bank.py --n-skel 6
"""

import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).parent
LEVELS = [2, 3, 4, 5]


def d(n):
    """逐位空格(数字表示之一;token 边界与位边界对齐)"""
    return " ".join(str(x) for x in str(n))


# ══════════════════════════════════════════════════════════════════════════
# 训练骨架库(30 条,**有序**:N=6/15/30 取前缀)
# 每项 = (名字, 种类, prefix 函数)   kind ∈ {"word","layout"}
# ══════════════════════════════════════════════════════════════════════════
BANK = [
    # ---- 第 1 组(idx 0–5):与 gen_arith_struct.py 的 SEEN **逐字相同** ----
    ("T01纯符号",   "word",   lambda a, b: "%s + %s = " % (d(a), d(b))),
    ("T02无等号",   "word",   lambda a, b: "计算：%s + %s，答案是 " % (d(a), d(b))),
    ("T03求和",     "word",   lambda a, b: "求 %s 与 %s 的和，结果是 " % (d(a), d(b))),
    ("T04疑问",     "word",   lambda a, b: "%s 加上 %s 等于几？" % (d(a), d(b))),
    ("T05自然写法", "word",   lambda a, b: "%d 加 %d = " % (a, b)),
    ("T06应用题",   "word",   lambda a, b: "小明有 %d 元，又得到 %d 元，一共多少元？" % (a, b)),

    # ---- 第 2 组(idx 6–14):换词/换标点(与第 1 组同骨架) ----
    ("T07相加得",   "word",   lambda a, b: "%s 与 %s 相加得 " % (d(a), d(b))),
    ("T08计算结果", "word",   lambda a, b: "%s + %s 的计算结果是 " % (d(a), d(b))),
    ("T09把加起来", "word",   lambda a, b: "把 %s 和 %s 加起来，得到 " % (d(a), d(b))),
    ("T10半句和",   "word",   lambda a, b: "%s 加上 %s，和是 " % (d(a), d(b))),
    ("T11请计算",   "word",   lambda a, b: "请计算 %s + %s = " % (d(a), d(b))),
    ("T12等于多少", "word",   lambda a, b: "%s 加 %s 等于多少？" % (d(a), d(b))),
    ("T13求和冒号", "word",   lambda a, b: "求和：%s + %s = " % (d(a), d(b))),
    ("T14增加",     "word",   lambda a, b: "%s 增加 %s 后是多少？" % (d(a), d(b))),
    ("T15总和",     "word",   lambda a, b: "%s 和 %s 的总和是 " % (d(a), d(b))),

    # ---- 第 3 组(idx 15–29):**换布局**为主(11 layout / 4 word) ----
    ("T16符号紧凑", "layout", lambda a, b: "%s+%s=" % (d(a), d(b))),
    ("T17全连续",   "layout", lambda a, b: "%d+%d=" % (a, b)),
    ("T18带标签",   "layout", lambda a, b: "总数 = %s + %s = " % (d(a), d(b))),
    ("T19竖式",     "layout", lambda a, b: "%s\n+ %s\n= " % (d(a), d(b))),
    ("T20多行标签", "layout", lambda a, b: "第一个数：%s\n第二个数：%s\n和：" % (d(a), d(b))),
    ("T21逆序操作", "layout", lambda a, b: "%s + %s = " % (d(b), d(a))),
    ("T22提示前缀", "layout", lambda a, b: "[加法] %s + %s = " % (d(a), d(b))),
    ("T23问在前",   "layout", lambda a, b: "和是多少？两个加数是 %s 和 %s。" % (d(a), d(b))),
    ("T24分号字段", "layout", lambda a, b: "加数 %s；加数 %s；和 " % (d(a), d(b))),
    ("T25函数式",   "layout", lambda a, b: "求和(%s, %s) = " % (d(a), d(b))),
    ("T26合起来",   "word",   lambda a, b: "%s 与 %s 合起来是 " % (d(a), d(b))),
    ("T27和为多少", "word",   lambda a, b: "%s 与 %s 的和为多少？" % (d(a), d(b))),
    ("T28结果为",   "word",   lambda a, b: "%s 加 %s 的结果为 " % (d(a), d(b))),
    ("T29总计",     "word",   lambda a, b: "%s 和 %s 总计是 " % (d(a), d(b))),
    ("T30位移",     "layout", lambda a, b: "  %s\n+ %s\n= " % (d(a), d(b))),
]

# ══════════════════════════════════════════════════════════════════════════
# ★ 永远不训练的 10 种(V1–V4 与上一轮**逐字相同**,所以 N=6 那一点可复现)
# ══════════════════════════════════════════════════════════════════════════
HELD = [
    ("H01新叙事",   "layout", lambda a, b: "甲乙两地相距 %d 米，又延长 %d 米，现在多长？" % (a, b)),
    ("H02倒装",     "layout", lambda a, b: "%s 与 %s 之和是？" % (d(a), d(b))),
    ("H03祈使",     "layout", lambda a, b: "请把 %s 和 %s 相加。" % (d(a), d(b))),
    ("H04括号",     "layout", lambda a, b: "(%s) + (%s) = " % (d(a), d(b))),
    ("H05问在前",   "layout", lambda a, b: "和是多少？已知两个加数是 %s 和 %s。" % (d(a), d(b))),
    ("H06填空",     "layout", lambda a, b: "%s + %s = ____" % (d(a), d(b))),
    ("H07编号列表", "layout", lambda a, b: "1) %s\n2) %s\n合计：" % (d(a), d(b))),
    ("H08箭头",     "layout", lambda a, b: "%s 与 %s → " % (d(a), d(b))),
    # H09 是**换一种描述方式**(从 X 往后数 Y 个)而不是换格式 —— 语义难度略高,单列
    ("H09往后数",   "resem",  lambda a, b: "从 %s 开始往后数 %s 个，得到 " % (d(a), d(b))),
    ("H10绳子",     "layout", lambda a, b: "%s 米长的绳子接上 %s 米，全长多少米？" % (d(a), d(b))),
]

# 名字 → (kind, fn),供评测脚本按名字取
ALL = {n: (k, f) for n, k, f in (BANK + HELD)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-skel", type=int, default=30,
                    help="训练用 BANK 的前 N 个骨架(6/15/30)")
    ap.add_argument("--per-skel", type=int, default=10000)
    ap.add_argument("--n-eval", type=int, default=100, help="每档题数(×4 档)")
    ap.add_argument("--seed", type=int, default=41)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    erng = random.Random(args.seed + 1)

    train = BANK[:args.n_skel]
    assert len(train) == args.n_skel, "BANK 不够长"

    def pair(k, r):
        return (r.randint(10 ** (k - 1), 10 ** k - 1),
                r.randint(10 ** (k - 1), 10 ** k - 1))

    # ---- 训练/测试语料(1/20 进测试集) ----
    n_tr = n_te = 0
    tr_name = HERE / f"bank_train_n{args.n_skel}.jsonl"
    te_name = HERE / f"bank_test_n{args.n_skel}.jsonl"
    with open(tr_name, "w", encoding="utf-8") as ftr, \
         open(te_name, "w", encoding="utf-8") as fte:
        for name, _kind, f in train:
            for i in range(args.per_skel):
                a, b = pair(rng.choice(LEVELS), rng)
                line = f(a, b) + str(a + b)      # ★ 答案不含空格
                tgt = fte if i % 20 == 0 else ftr
                n_te += (i % 20 == 0)
                n_tr += (i % 20 != 0)
                tgt.write(json.dumps({"segs": [{"t": line + "\n", "loss": 1}]},
                                     ensure_ascii=False) + "\n")

    # ---- 评测集:40 个骨架全出,与训练用的 N 无关(同一份文件三次复用)----
    names = []
    for name, _kind, f in BANK + HELD:
        names.append(name)
        with open(HERE / f"bank_eval_{name}.jsonl", "w", encoding="utf-8") as fev:
            for k in LEVELS:
                for _ in range(args.n_eval):
                    a, b = pair(k, erng)
                    fev.write(json.dumps({"d": k, "theme": name,
                                          "q": f(a, b), "a": str(a + b)},
                                         ensure_ascii=False) + "\n")

    (HERE / "bank_eval_manifest.txt").write_text("\n".join(names) + "\n",
                                                 encoding="utf-8")
    # 给 analyze_bank.py 用(否则它得把 40 个 lambda 再抄一遍,抄错就静默出错)
    (HERE / "bank_kinds.txt").write_text(
        "\n".join("%s %s" % (n, k) for n, k, _ in BANK + HELD) + "\n",
        encoding="utf-8")

    kind_ct = {}
    for _n, k, _f in train:
        kind_ct[k] = kind_ct.get(k, 0) + 1
    print(f"训练 N={args.n_skel} 骨架 × {args.per_skel}  → {n_tr} 条训练 / {n_te} 条测试")
    print(f"  其中 layout {kind_ct.get('layout', 0)} / word {kind_ct.get('word', 0)}")
    print(f"  {tr_name.name}  /  {te_name.name}")
    print(f"评测集 {len(names)} 个 × {args.n_eval*len(LEVELS)} 题 → bank_eval_*.jsonl")
    print(f"  ★ 未见:{' '.join(n for n,_,_ in HELD)}")
    print("\n样例:")
    for name, _kind, f in train[:3] + HELD[:3]:
        a, b = 1234, 5678
        print("  %-12s %r" % (name, f(a, b) + str(a + b)))


if __name__ == "__main__":
    main()
