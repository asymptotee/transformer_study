"""gen_variant.py —— 泛化测试:换措辞,答案格式不变

**为什么必须做这个(2026-09-23)**:stage17 扫出关系推理和指令变换都到 **100%** ——
但那 100% 是**同一个生成器、同一套措辞**测出来的。所以严格说只证明了:

    "学会了这个模板"        而不是        "学会了推理"

**这个区分决定结论的方向:**
  · 换措辞仍然会 → 真的学会了处理(推理是可训练的)→ 30 小时方案有依据
  · 换措辞就崩   → 那 100% 是模板匹配 → 上面的乐观判断要推翻

**设计原则**:只换**措辞**,答案的形态**完全不变**(关系推理还是单个字母、
指令变换还是逐位空格、状态追踪还是数字)。这样测的才是"措辞泛化",
不是"输出格式适应"。

每个任务给 3 个 theme:
  · 第一个是**训练时见过的措辞**(对照,应该 100%)
  · 后两个是**没见过的措辞** ← 关键数据

用法(Spark):
  ~/llm_study/.venv/bin/python gen_variant.py
"""

import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).parent
NAMES = "ABCDEFGH"


def d(n):
    return " ".join(str(x) for x in str(n))


# ---------------- 关系推理:4 种措辞 ----------------
def _rel(perm, pairs, rng, stmt_tpl, ask_hi, ask_lo):
    rng.shuffle(pairs)
    stmt = "，".join(stmt_tpl.format(a=a, b=b) for a, b in pairs)
    hi = rng.random() < 0.5
    q = "%s。%s？" % (stmt, ask_hi if hi else ask_lo)
    return q, (perm[0] if hi else perm[-1])


REL_THEMES = {
    # (主题名, 陈述模板, 问"高"的说法, 问"矮"的说法, 是否会)
    "身高(训练同款)": ("{a} 比 {b} 高", "谁最高", "谁最矮", True),
    "年龄(新)":       ("{a} 比 {b} 年长", "谁年龄最大", "谁年龄最小", False),
    "分数(新)":       ("{a} 的分数比 {b} 高", "谁分数最高", "谁分数最低", False),
    "排队(新)":       ("{a} 排在 {b} 前面", "谁排在最前面", "谁排在最后面", False),
}


def gen_relation(theme, lv, rng):
    tpl, hi, lo, _ = REL_THEMES[theme]
    k = rng.randint(3, max(3, lv))
    perm = rng.sample(NAMES[:k], k)
    pairs = list(zip(perm, perm[1:]))
    return _rel(perm, pairs, rng, tpl, hi, lo)


# ---------------- 指令变换:6 种措辞 ----------------
def _rev(s, rng, tpl):
    txt = " ".join(str(x) for x in s)
    return tpl.format(x=txt), " ".join(str(x) for x in reversed(s))


def _del(s, rng, tpl):
    txt = " ".join(str(x) for x in s)
    pos = rng.randint(1, len(s))
    return (tpl.format(x=txt, k=pos),
            " ".join(str(x) for x in s[:pos - 1] + s[pos:]))


TR_THEMES = {
    "倒序(训练同款)": ("rev", "把 {x} 倒过来。"),
    "逆序(新)":       ("rev", "把 {x} 逆序排列。"),
    "反过来写(新)":   ("rev", "{x} 反过来写是什么？"),
    "删第k个(训练同款)": ("del", "删掉 {x} 里的第 {k} 个。"),
    "去掉(新)":       ("del", "去掉 {x} 的第 {k} 个。"),
    "删去第k项(新)":  ("del", "从 {x} 里删去第 {k} 项。"),
}


def gen_transform(theme, lv, rng):
    kind, tpl = TR_THEMES[theme]
    k = rng.randint(3, max(3, lv))
    s = [rng.randint(0, 9) for _ in range(k)]
    return _rev(s, rng, tpl) if kind == "rev" else _del(s, rng, tpl)


# ---------------- 状态追踪:3 种措辞 ----------------
def _state(rng, n, tpl, add_w, sub_w):
    cur = rng.randint(3, 20)
    start = cur
    parts = []
    for _ in range(rng.randint(2, n)):
        if cur <= 2 or rng.random() < 0.5:
            v = rng.randint(1, 9); cur += v
            parts.append(add_w.format(v=d(v)))
        else:
            v = rng.randint(1, min(9, cur)); cur -= v
            parts.append(sub_w.format(v=d(v)))
    return tpl.format(n=d(start), ops="，".join(parts)), d(cur)


ST_THEMES = {
    "球盒(训练同款)": ("盒子里有 {n} 个球。{ops}。现在有几个？", "放进 {v} 个", "拿走 {v} 个"),
    "苹果篮(新)":     ("篮子里有 {n} 个苹果。{ops}。现在有多少个？", "增加 {v} 个", "减少 {v} 个"),
    "极简(新)":       ("初始 {n} 个。{ops}。结果？", "加 {v}", "减 {v}"),
}

TASKS = {
    "relation": (REL_THEMES, gen_relation),
    "transform": (TR_THEMES, gen_transform),
    "state": (ST_THEMES, gen_state := (lambda th, lv, rng: _state(rng, lv, *ST_THEMES[th]))),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="每个 theme 的题数")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    for task, (themes, gen) in TASKS.items():
        out = HERE / f"{task}_variant_eval.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for th in themes:
                for _ in range(args.n):
                    q, a = gen(th, 6 if task != "state" else 5, rng)
                    f.write(json.dumps({"task": task, "level": 0, "theme": th,
                                        "q": q, "a": a},
                                       ensure_ascii=False) + "\n")
        print(f"[{task}] {len(themes)} 个 theme × {args.n} = {len(themes)*args.n} 题 → {out.name}")
        for th in themes:
            q, a = gen(th, 6 if task != "state" else 5, random.Random(0))
            print("   %-20s %s" % (th, q[:64]))
        print()


if __name__ == "__main__":
    main()
