"""render_mt_chat.py —— 14.1:多轮对话渲染(阶段14 的数据准备)

背景:stage11 的 SFT 数据(chat_train_nt.jsonl)来自 stage10 的
render_sft_chat.py,而那个渲染器第一行过滤就是

    if len(convs) != 2 or ...:  return None      # 只留单轮对

于是官方 sft_t2t_mini.jsonl 里 **201,294 条多轮对话(22.2%)一条没用过**
(其中含 tool 的 76,574 条留给 stage15,本阶段只用**纯多轮 124,720 条**)。
11.3 的实测结论因此一直成立:多轮对本模型是外推,表现是主题粘连、
角色混乱、自问自答。

本文件把那个过滤放开,并按**段列表**输出(14.2 起):

    {"segs": [{"t": "<|im_start|>user\\nQ1<|im_end|>\\n",        "loss": 0},
              {"t": "<|im_start|>assistant\\nA1<|im_end|>\\n",    "loss": 1},
              {"t": "<|im_start|>user\\nQ2<|im_end|>\\n",        "loss": 0},
              {"t": "<|im_start|>assistant\\nA2<|im_end|>\\n",    "loss": 1}]}

**每个 assistant 轮都算 loss,不是只算最后一轮** —— 这是 minimind 的做法
(`generate_labels` 扫全流,每段 `<|assistant|>…<|/assistant|>` 都标),也比
"只训最后一轮"信号量高约 3×。配套的 `train_mt.py` 按段拼掩码。

> 14.1 曾沿用 stage10 的三段式(`user_side`/`answer`/`tail`),那样只能支持
> "一个连续的目标区间",即只训最后一轮。改成段列表后三段式成为它的特例
> (`[0,1,1]`),而 `train_mt.py` 也必须按段拼掩码 —— 故两个文件一起改。

两个踩过的坑(都写进代码里了):
  1. **不能只读前缀**:sft_t2t_mini.jsonl 是**按来源分块排的** —— 前 20 万行
     多轮占 35.5%、含 tool 的 0 条;全文件多轮只占 22.2%、含 tool 76,574 条。
     只从前缀取样本 = 从一个偏斜的连续段取样。故用**两遍扫描**:先廉价地
     (只 json.loads + 角色检查)收集全部合格候选行号,全局 shuffle 后再渲染。
  2. **think 要剥整段,不能只剥答案**:历史轮里留 `<think></think>` 而目标轮
     不带,等于教模型"别人想、我别想"。stage11 剥 think 的理由正是"不然会学到
     先空想再绕圈"。

用法(Spark,要调官方 tokenizer → 有 minimind 仓库 + transformers 的机器):
  ~/llm_study/.venv/bin/python render_mt_chat.py --n 40000 --strip-think \
      --out mt_train_nt.jsonl --out-test mt_test_nt.jsonl

窗口 768(不是 stage11 的 512)的依据见 stage9 README 9.2 节的更正块:
SFT 在 ≤512 上训过就能把 768 的 ppl 从 22.0 拉到 7.1,而多轮样本的中位数
499 / 90 分位 760 —— 512 会砍掉 45%。
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


def qualifies(convs, mix=False):
    """廉价预筛(不渲染)。返回 (合格?, 原因码) 便于统计。

    mix=False(默认,多轮专用):要求 assistant ≥ 2
    mix=True(混训模式):放宽到 assistant ≥ 1,即**不按对话形态筛选** ——
        2 条消息的单轮对也收,渲染出来是 [ctx, target] 两段,和多轮的
        [ctx,target,ctx,target,…] 是同一种结构。
    """
    if not convs or convs[-1]["role"] != "assistant":
        return False, "末条非 assistant"
    n_asst = sum(1 for m in convs if m["role"] == "assistant")
    if n_asst < (1 if mix else 2):
        return False, "assistant<%d" % (1 if mix else 2)
    if convs[0]["role"] != "user":
        return False, "首条非 user"
    if any(m["role"] == "tool" or m.get("tool_calls") for m in convs):
        return False, "含 tool"
    return True, "ok"


def split_sample(tok, convs, strip_think=False):
    """渲染后按 `<|im_start|>assistant\\n…<|im_end|>` 切成段列表。

    每段的 loss 标记决定它是否参与训练:assistant 轮的全部 token(含开头的
    marker 之后的内容与收尾的 <|im_end|>)算 loss,user 轮与衔接文本不算。
    最后一个目标段吃掉尾部剩余(与 stage10 的 tail 语义一致)。

    strip_think 作用于**整段渲染文本**(历史轮也剥),不只是答案 —— 见文件头坑 2。
    """
    text = tok.apply_chat_template(convs, tokenize=False, add_generation_prompt=False)
    if strip_think:
        text = THINK_RE.sub("", text)

    bounds = []                            # 每个 assistant 目标的 [起, 止)
    pos = 0
    while True:
        i = text.find(MARKER, pos)
        if i < 0:
            break
        s = i + len(MARKER)
        e = text.find("<|im_end|>", s)
        if e < 0:                          # 末尾 assistant 没闭合 → 整条丢弃
            return None
        bounds.append([s, e + len("<|im_end|>")])
        pos = e + len("<|im_end|>")
    if not bounds:
        return None
    bounds[-1][1] = len(text)              # 最后一段吃掉尾部换行

    segs, prev = [], 0
    for s, e in bounds:
        if not text[s:e].strip():          # 空回答(think 剥完为空)→ 丢弃
            return None
        if s > prev:
            segs.append({"t": text[prev:s], "loss": 0})
        segs.append({"t": text[s:e], "loss": 1})
        prev = e
    return {"segs": segs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(Path.home() / "llm_study" / "mm_data" /
                                         "sft_t2t_mini.jsonl"))
    ap.add_argument("--n", type=int, default=40000,
                    help="训练样本目标数(对齐 stage11 单轮那次的 32057;"
                         "768 窗口渲染后还会再砍一刀,故多要一些)")
    ap.add_argument("--mix", action="store_true",
                    help="混训模式:单轮+多轮一起收,按 --ratio-single 控制占比")
    ap.add_argument("--ratio-single", type=float, default=0.5,
                    help="混训时单轮样本占比(0.5=50:50)。依据见 14.5 节:"
                         "自然比例 22%% 多轮只能把污染率从 63.9%% 降到 53.8%%,"
                         "远不如纯多轮的 8.1%%,故要明显加重多轮")
    ap.add_argument("--max-answer-chars", type=int, default=800,
                    help="单个 assistant 目标段的字符数上限")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strip-think", action="store_true",
                    help="剥掉整段渲染文本里的 think 块(与 stage11 的 nt 版一致)")
    ap.add_argument("--out", default="mt_train_nt.jsonl")
    ap.add_argument("--out-test", default="mt_test_nt.jsonl")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MM / "model")

    # ---- 第一遍:只做廉价预筛,收集全部合格候选的行号 ----
    # 混训模式下按对话形态分两池,再按 --ratio-single 取样
    c_single, c_multi, reasons = [], [], {}
    for i, ln in enumerate(open(args.src, encoding="utf-8")):
        convs = json.loads(ln)["conversations"]
        ok, why = qualifies(convs, mix=args.mix)
        if not ok:
            reasons[why] = reasons.get(why, 0) + 1
        elif args.mix and sum(1 for m in convs if m["role"] == "assistant") < 2:
            c_single.append(i)
        else:
            c_multi.append(i)
    if args.mix:
        print(f"候选池:单轮 {len(c_single)} | 多轮 {len(c_multi)} | 过滤 {reasons}",
              flush=True)
    else:
        print(f"全文件合格多轮候选: {len(c_multi)} 条 | 各级过滤 {reasons}", flush=True)

    # ---- 全局 shuffle 后取样(训练 n 条 + 测试 500,互不相交)----
    random.seed(args.seed)
    random.shuffle(c_single)
    random.shuffle(c_multi)
    if args.mix:
        n_s = min(len(c_single), int(args.n * args.ratio_single))
        n_m = min(len(c_multi), args.n - n_s)
        n_s = min(len(c_single), args.n - n_m)        # 某池不够就由另一池补足
        cand = c_single[:n_s] + c_multi[:n_m]
        random.shuffle(cand)
        print(f"取样:单轮 {n_s} + 多轮 {n_m} = {len(cand)}(目标比例 "
              f"单轮 {args.ratio_single:.0%})", flush=True)
    else:
        cand = c_multi
    need = args.n + 500
    sel = cand[:min(len(cand), int(need * 1.5))]      # 多取 50% 做渲染丢样缓冲
    # 测试集**从前面预留**,不靠"训练填满后的溢出" —— 候选池不够大时那种写法
    # 会得到空的测试集(踩过一次:9 万候选渲染后只剩 8.2 万,永远填不满)
    test_sel, train_sel = sel[:500], sel[500:]
    sel_set = set(sel)
    lines = {}
    for i, ln in enumerate(open(args.src, encoding="utf-8")):
        if i in sel_set:
            lines[i] = ln

    def render_rows(indices, cap):
        """按给定行号渲染,凑够 cap 条就停(丢弃的样本不计入)。"""
        out = []
        for i in indices:
            if len(out) >= cap:
                break
            s = split_sample(tok, json.loads(lines[i])["conversations"],
                             strip_think=args.strip_think)
            if s is None:
                continue
            if max(len(g["t"]) for g in s["segs"] if g["loss"]) > args.max_answer_chars:
                continue
            out.append(s)
        return out

    # 两个池**分别**按各自的来源列表渲染 —— 不能用"训练集长度"当条件,
    # 那个条件在候选池不够大时永远不成立,测试集会是空的(踩过两次)
    out_tr = render_rows(train_sel, args.n)
    out_te = render_rows(test_sel, 500)
    n_turns = sum(sum(1 for g in r["segs"] if g["loss"]) for r in out_tr + out_te)
    for path, rows in ((args.out, out_tr), (args.out_test, out_te)):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"训练 {len(out_tr)} | 测试 {len(out_te)} | "
          f"平均目标轮数 {n_turns/max(1,len(out_tr)+len(out_te)):.2f}", flush=True)
    for k in range(min(2, len(out_tr))):
        print(f"\n--- 样例 {k}({len(out_tr[k]['segs'])} 段)---")
        for g in out_tr[k]["segs"]:
            t = g["t"].replace("\n", "⏎")
            print(f"  [loss={g['loss']}] {t[:70]}{'…' if len(t) > 70 else ''} "
                  f"…{t[-40:] if len(t) > 110 else ''}")


if __name__ == "__main__":
    main()
