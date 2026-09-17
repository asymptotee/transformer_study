"""render_mt_chat.py —— 14.1:多轮对话渲染(阶段14 的数据准备)

背景:stage11 的 SFT 数据(chat_train_nt.jsonl)来自 stage10 的
render_sft_chat.py,而那个渲染器第一行过滤就是

    if len(convs) != 2 or ...:  return None      # 只留单轮对

于是官方 sft_t2t_mini.jsonl 里 **201,294 条多轮对话(22.2%)一条没用过**
(其中含 tool 的 76,574 条留给 stage15,本阶段只用**纯多轮 124,720 条**)。
11.3 的实测结论因此一直成立:多轮对本模型是外推,表现是主题粘连、
角色混乱、自问自答。

本文件把那个过滤放开,并按**最后一个 assistant 轮**切:

    user_side = 整段历史(含前面所有轮) —— 全部掩码,只当上下文
    answer    = 最后一个 assistant 的内容   —— 算 loss
    tail      = <|im_end|> 及之后           —— 算 loss

**关键设计:输出沿用 stage10 的三段式格式,所以 train_chat10.py 一行都不用改**
(它的 build() 只关心 user_side 有多长,不关心里面有几轮)。渲染器 + 训练脚本
这条分工与 stage11 复用 stage10 的约定一致。

两个踩过的坑(都写进代码里了):
  1. **不能只读前缀**:sft_t2t_mini.jsonl 是**按来源分块排的** —— 前 20 万行
     多轮占 35.5%、含 tool 的 0 条;全文件多轮只占 22.2%、含 tool 76,574 条。
     只从前缀取样本 = 从一个偏斜的连续段取样。故用**两遍扫描**:先廉价地
     (只 json.loads + 角色检查)收集全部合格候选行号,全局 shuffle 后再渲染。
  2. **think 要剥整段,不能只剥答案**:历史轮里留 `<think></think>` 而目标轮
     不带,等于教模型"别人想、我别想"。stage11 剥 think 的理由正是"不然会学到
     先空想再绕圈"。

用法(Spark,要调官方 tokenizer → 有 minimind 仓库 + transformers 的机器):
  ~/llm_study/.venv/bin/python render_mt_chat.py --n 30000 --strip-think \
      --out mt_train_nt.jsonl --out-test mt_test_nt.jsonl
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

MM = Path.home() / "llm_study" / "minimind"
sys.path.insert(0, str(MM))
from transformers import AutoTokenizer                    # noqa: E402

MARKER = "<|im_start|>assistant\n"
THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def qualifies(convs):
    """廉价预筛(不渲染)。返回 (合格?, 原因码) 便于统计。"""
    if not convs or convs[-1]["role"] != "assistant":
        return False, "末条非 assistant"
    if sum(1 for m in convs if m["role"] == "assistant") < 2:
        return False, "assistant<2"
    if convs[0]["role"] != "user":
        return False, "首条非 user"
    if any(m["role"] == "tool" or m.get("tool_calls") for m in convs):
        return False, "含 tool"
    return True, "ok"


def split_sample(tok, convs, strip_think=False):
    """多轮:历史当上下文,最后一个 assistant 轮当目标。

    strip_think 作用于**整段渲染文本**(历史轮也剥),不只是答案 —— 见文件头坑 2。
    """
    text = tok.apply_chat_template(convs, tokenize=False, add_generation_prompt=False)
    if strip_think:
        text = THINK_RE.sub("", text)
    i = text.rfind(MARKER)                 # 最后一个 assistant 开头
    if i < 0:
        return None
    s = i + len(MARKER)
    e = text.find("<|im_end|>", s)
    if e < 0:
        return None
    answer = text[s:e].strip()
    if not answer:
        return None
    return {"user_side": text[:s], "answer": answer, "tail": text[e:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(Path.home() / "llm_study" / "mm_data" /
                                         "sft_t2t_mini.jsonl"))
    ap.add_argument("--n", type=int, default=30000,
                    help="训练样本目标数(对齐 stage11 单轮那次的 32057)")
    ap.add_argument("--max-answer-chars", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strip-think", action="store_true",
                    help="剥掉整段渲染文本里的 think 块(与 stage11 的 nt 版一致)")
    ap.add_argument("--out", default="mt_train_nt.jsonl")
    ap.add_argument("--out-test", default="mt_test_nt.jsonl")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MM / "model")

    # ---- 第一遍:只做廉价预筛,收集全部合格候选的行号 ----
    cand, reasons = [], {}
    for i, ln in enumerate(open(args.src, encoding="utf-8")):
        ok, why = qualifies(json.loads(ln)["conversations"])
        if ok:
            cand.append(i)
        else:
            reasons[why] = reasons.get(why, 0) + 1
    print(f"全文件合格多轮候选: {len(cand)} 条 | 各级过滤 {reasons}", flush=True)

    # ---- 全局 shuffle 后取样(训练 n 条 + 测试 500,互不相交)----
    random.seed(args.seed)
    random.shuffle(cand)
    need = args.n + 500
    sel = cand[:min(len(cand), int(need * 1.5))]      # 多取 50% 做渲染丢样缓冲
    sel_set = set(sel)
    order = {i: k for k, i in enumerate(sel)}
    lines = {}
    for i, ln in enumerate(open(args.src, encoding="utf-8")):
        if i in sel_set:
            lines[i] = ln

    out_tr, out_te = [], []
    for i in sel:                                     # 按 shuffle 后的顺序渲染
        s = split_sample(tok, json.loads(lines[i])["conversations"],
                         strip_think=args.strip_think)
        if s is None or len(s["answer"]) > args.max_answer_chars:
            continue
        (out_te if len(out_tr) >= args.n else out_tr).append(s)
        if len(out_tr) >= args.n and len(out_te) >= 500:
            break

    out_te = out_te[:500]
    for path, rows in ((args.out, out_tr), (args.out_test, out_te)):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"训练 {len(out_tr)} | 测试 {len(out_te)}", flush=True)
    for k in range(min(2, len(out_tr))):
        ex = out_tr[k]
        print(f"\n--- 样例 {k} ---")
        print("user_side 尾部:", ex["user_side"][-170:].replace("\n", "⏎"))
        print("answer       :", ex["answer"][:60].replace("\n", "⏎"))
        print("tail         :", ex["tail"].replace("\n", "⏎"))


if __name__ == "__main__":
    main()
