"""gen_reason.py —— stage17 能力边界扫描的数据生成器(三个维度)

**要回答的问题(2026-09-22)**:stage16 证明了 126M 能用 14 分钟学会 5 位加法。
但**加法只是一个窄算法**。"逻辑思维"能不能也这样教出来?

**Holmes 的实测是个负面信号** —— 0.5B、4 倍参数、专为推理设计、100 倍数据,
在关系推理 0/3、指令变换 2/4、形式逻辑 2/6。

所以先花 ~1 小时扫三个维度。**每个都用 stage16 验证过的三件套**:
程序生成 + 答案可验证 + 干净格式。

**三个任务**(难度阶梯):

  ① 关系推理    `A 比 C 高，C 比 B 高。谁最矮？B`          阶梯:n = 3→6 个元素
  ② 指令变换    `把 3 7 1 5 倒过来。5 1 7 3`              阶梯:长度 3→6
                `删掉 4 2 9 6 里的第 2 个。4 9 6`
  ③ 状态追踪    `盒子里有 8 个球。放进 3 个，拿走 5 个。现在有几个？6`
                                                          阶梯:操作数 2→5

**防作弊设计(很重要)**:
  · 关系推理:两两关系**打乱顺序**陈述,而且"问最大/问最小"**随机** ——
    否则模型只要输出最后提到的那个就对了,根本不用读关系
  · 指令变换:**位置随机**、数字随机
  · 状态追踪:每步是加还是减随机,保证中间结果不为负
  · 三个任务的答案**都不是位置** —— 必须真的算

**数字逐位加空格**(同 stage16):我们的 BPE 把多位数切成乱块,
而逐位对齐后 token 边界 = 位边界。

用法(Spark):
  ~/llm_study/.venv/bin/python gen_reason.py --task relation --n 60000
  ~/llm_study/.venv/bin/python gen_reason.py --task transform --n 60000
  ~/llm_study/.venv/bin/python gen_reason.py --task state --n 60000
"""

import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).parent
NAMES = "ABCDEFGH"          # 关系推理用字母,避免中文姓名的歧义


def d(n):
    """数字逐位空格分隔(对齐 token 与位)"""
    return " ".join(str(x) for x in str(n))


# ---------------- ① 关系推理 ----------------
def gen_relation(n, rng):
    """n 个元素排成一条严格顺序;打乱顺序陈述两两相邻关系;随机问 max 或 min。"""
    k = rng.randint(3, n)                      # 本轮用几个元素
    perm = rng.sample(NAMES[:k], k)            # perm[0] 最高 … perm[-1] 最矮
    pairs = list(zip(perm, perm[1:]))          # 相邻两两
    rng.shuffle(pairs)                         # ★ 打乱陈述顺序(防位置猜)
    ask_max = rng.random() < 0.5               # ★ 随机问最大还是最小
    stmt = "，".join("%s 比 %s 高" % (a, b) for a, b in pairs)
    q = "%s。谁最%s？" % (stmt, "高" if ask_max else "矮")
    ans = perm[0] if ask_max else perm[-1]
    return "%s%s" % (q, ans), q, ans


# ---------------- ② 指令变换 ----------------
def gen_transform(n, rng):
    """两种变换:整串倒序 / 删掉第 k 个。位置与数字都随机。"""
    k = rng.randint(3, n)                      # 串长
    s = [rng.randint(0, 9) for _ in range(k)]
    txt = " ".join(str(x) for x in s)
    if rng.random() < 0.5:
        q = "把 %s 倒过来。" % txt
        ans = " ".join(str(x) for x in reversed(s))
    else:
        pos = rng.randint(1, k)                # ★ 随机位置
        q = "删掉 %s 里的第 %s 个。" % (txt, pos)
        ans = " ".join(str(x) for x in s[:pos - 1] + s[pos:])
    return "%s%s" % (q, ans), q, ans


# ---------------- ③ 状态追踪 ----------------
def gen_state(n, rng):
    """初始数量 + n 步加减,保证中间不为负。问最终数量。"""
    cur = rng.randint(3, 20)
    start = cur
    ops, parts = [], []
    for _ in range(rng.randint(2, n)):
        if cur <= 2 or rng.random() < 0.5:
            v = rng.randint(1, 9)
            cur += v
            parts.append("放进 %s 个" % d(v))
        else:
            v = rng.randint(1, min(9, cur))
            cur -= v
            parts.append("拿走 %s 个" % d(v))
    q = "盒子里有 %s 个球。%s。现在有几个？" % (d(start), "，".join(parts))
    ans = d(cur)
    return "%s%s" % (q, ans), q, ans


TASKS = {"relation": gen_relation, "transform": gen_transform, "state": gen_state}
# 难度阶梯:每个任务的上限(元素数 / 串长 / 操作数)
LADDER = {"relation": (3, 6), "transform": (3, 6), "state": (2, 5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    gen = TASKS[args.task]
    lo, hi = LADDER[args.task]
    rng = random.Random(args.seed)
    erng = random.Random(args.seed + 1)          # 评测独立随机流,避免重叠

    n_tr = n_te = 0
    with open(HERE / f"{args.task}_train_nt.jsonl", "w", encoding="utf-8") as ftr, \
         open(HERE / f"{args.task}_test_nt.jsonl", "w", encoding="utf-8") as fte:
        for _ in range(args.n):
            line, _, _ = gen(hi, rng)
            ftr.write(json.dumps({"segs": [{"t": line + "\n", "loss": 1}]},
                                 ensure_ascii=False) + "\n")
            n_tr += 1
        for _ in range(args.n // 20):
            line, _, _ = gen(hi, rng)
            fte.write(json.dumps({"segs": [{"t": line + "\n", "loss": 1}]},
                                 ensure_ascii=False) + "\n")
            n_te += 1
    # 评测集:每个难度档都要(这样能画出曲线,而不是只有一个点)
    with open(HERE / f"{args.task}_eval.jsonl", "w", encoding="utf-8") as fev:
        for lv in range(lo, hi + 1):
            for _ in range(args.n_eval):
                _, q, a = gen(lv, erng)
                fev.write(json.dumps({"task": args.task, "level": lv,
                                      "q": q, "a": a},
                                     ensure_ascii=False) + "\n")

    print(f"[{args.task}] 训练 {n_tr} | 测试 {n_te} | 评测 {args.n_eval}×"
          f"{hi-lo+1} 档({lo}-{hi})")
    print("样例:")
    r2 = random.Random(0)
    for lv in range(lo, hi + 1):
        print("  %d 档: %s" % (lv, gen(lv, r2)[0]))


if __name__ == "__main__":
    main()
