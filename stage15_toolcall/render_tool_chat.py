"""render_tool_chat.py —— 工具调用数据渲染器

**背景(侦察结论,2026-09-20)**:`sft_t2t_mini.jsonl` 里 76,574 条含 tool 的对话,
结构和 stage14 README 当初的设想**不一样**:

  · **100% 是单轮用户请求**(1 个 user → 1 轮工具 → 1 个答复),没有多轮
    → 所以"拆 A/B 切片(单轮 / +多轮)分离难度"那个设计前提不成立
  · 81% 的对话**只有 1 个可用工具**,每次调用 1 个工具(92%)
  · 只有 **11 种人造工具**(random_number / generate_image / get_exchange_rate…),
    不是真实 API —— 玩具级数据集
  · 难点因此是两件新事:① 生成结构化 `<tool_call>` JSON ② 读 `<tool_response>` 作答

对话形态(语料格式):
  [0] system  content="" + tools="[函数定义 JSON]"
  [1] user    请求
  [2] assistant  content="简述" + tool_calls="[{function:{name,arguments}}]"
  [3] tool    结果 JSON
  [4] assistant 最终答复

渲染成(他们的 chat_template,注意 **tools 要当顶层 kwarg 传**,塞进 system
消息里是渲染不出来的 —— 这个坑在侦察时踩过一次,会静默产出空的 system):
  <|im_start|>system\n# Tools\n…<tools>[定义]</tools>…<|im_end|>
  <|im_start|>user\n请求<|im_end|>
  <|im_start|>assistant\n\n\n<tool_call>\n{name, arguments}\n</tool_call><|im_end|>
  <|im_start|>user\n<tool_response>\n结果\n</tool_response><|im_end|>
  <|im_start|>assistant\n最终答复<|im_end|>

**两个 assistant 轮都算 loss**(段列表格式,同 stage14)—— 工具调用和最终答复
都是要学的行为。

**窗口 1024,不是 768**:渲染后中位 **761** token,768 只活 53.7%、**512 存活 0.0%**
(主因是 system 里的工具定义)。1024 存活 95.9%。

用法(Spark):
  ~/llm_study/.venv/bin/python render_tool_chat.py \
      --n 40000 --strip-think --out tool_train_nt.jsonl --out-test tool_test_nt.jsonl
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "stage4_scaling_bpe"))
sys.path.insert(0, str(Path.home() / "llm_study" / "minimind"))
from bpe import BPETokenizer                          # noqa: E402
from transformers import AutoTokenizer                # noqa: E402

MARKER = "<|im_start|>assistant\n"
THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def to_hf(convs):
    """语料格式 → (HF 消息列表, tools)。tool_calls 从 JSON 字符串解出来。"""
    msgs, tools = [], None
    for m in convs:
        if m["role"] == "system" and m.get("tools"):
            tools = json.loads(m["tools"])
            continue
        d = {"role": m["role"], "content": m.get("content", "")}
        if m.get("tool_calls"):
            d["tool_calls"] = [{"type": "function", "function": x["function"]}
                               for x in json.loads(m["tool_calls"])]
        msgs.append(d)
    return msgs, tools


def split_sample(tok, convs, strip_think=False):
    """渲染后按 `<|im_start|>assistant\\n…<|im_end|>` 切成段列表(同 stage14)。"""
    msgs, tools = to_hf(convs)
    if tools is None:                     # 含 tool 却没有定义 → 丢弃
        return None
    text = tok.apply_chat_template(msgs, tools=tools, tokenize=False,
                                   add_generation_prompt=False)
    if strip_think:
        text = THINK_RE.sub("", text)

    bounds, pos = [], 0
    while True:
        i = text.find(MARKER, pos)
        if i < 0:
            break
        s = i + len(MARKER)
        e = text.find("<|im_end|>", s)
        if e < 0:
            return None
        bounds.append([s, e + len("<|im_end|>")])
        pos = e + len("<|im_end|>")
    if not bounds:
        return None
    bounds[-1][1] = len(text)

    segs, prev = [], 0
    for s, e in bounds:
        if not text[s:e].strip():         # 空回答 → 丢弃
            return None
        if s > prev:
            segs.append({"t": text[prev:s], "loss": 0})
        segs.append({"t": text[s:e], "loss": 1})
        prev = e
    return {"segs": segs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(Path.home() / "llm_study" / "mm_data"
                                         / "sft_t2t_mini.jsonl"))
    ap.add_argument("--out", default="tool_train_nt.jsonl")
    ap.add_argument("--out-test", default="tool_test_nt.jsonl")
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--max-len", type=int, default=1024,
                    help="口径说明,不在这里过滤:**逐 token 长度检查交给 train_mt.py 的 "
                         "build()(12 进程并行)**。渲染器里单进程 bpe.encode 每条约 0.07 秒,"
                         "76,000 条要 92 分钟 —— 踩过,渲染器卡死 12 分钟无输出")
    ap.add_argument("--max-answer-chars", type=int, default=2000)
    ap.add_argument("--strip-think", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 两个 tokenizer,别混:**mm** 负责 chat 模板(它才有 apply_chat_template),
    # **bpe** 负责编码与长度(我们的词表)。踩过:把 bpe 传进 split_sample,
    # 76,520 条候选静默丢掉,被 except 吞了,只看到"训练 0 测试 0"
    mm = AutoTokenizer.from_pretrained(Path.home() / "llm_study" / "minimind" / "model")
    bpe = BPETokenizer.load(str(REPO / "stage11_datascale/cache_mm10g/bpe.json"))

    # ---- 第一遍:廉价预筛,收集全部候选行号 ----
    # (stage14 踩过:语料按来源分块排,前缀扫描会取到密集连续段而不是语料)
    cand, reasons = [], {}
    for i, ln in enumerate(open(args.src, encoding="utf-8")):
        c = json.loads(ln)["conversations"]
        if not any(m.get("role") == "tool" or m.get("tool_calls") for m in c):
            continue
        if c[-1]["role"] != "assistant":
            reasons["末条非 assistant"] = reasons.get("末条非 assistant", 0) + 1
            continue
        if not any(m["role"] == "system" and m.get("tools") for m in c):
            reasons["无工具定义"] = reasons.get("无工具定义", 0) + 1
            continue
        cand.append(i)
    print(f"合格 tool 候选: {len(cand)} 条 | 过滤 {reasons}", flush=True)

    random.seed(args.seed)
    random.shuffle(cand)
    test_sel, train_sel = cand[:500], cand[500:]
    sel_set = set(cand)
    lines = {}
    for i, ln in enumerate(open(args.src, encoding="utf-8")):
        if i in sel_set:
            lines[i] = ln

    drops = {}

    def bump(k):
        drops[k] = drops.get(k, 0) + 1

    def render_rows(indices, cap):
        out = []
        for i in indices:
            if len(out) >= cap:
                break
            s = split_sample(mm, json.loads(lines[i])["conversations"],
                             strip_think=args.strip_think)
            if s is None:
                bump("渲染失败(无目标段/未闭合/空回答)")
                continue
            if max(len(g["t"]) for g in s["segs"] if g["loss"]) > args.max_answer_chars:
                bump("单段答案过长")
                continue
            out.append({"segs": s["segs"]})
        return out

    out_tr = render_rows(train_sel, args.n)
    out_te = render_rows(test_sel, 500)
    nturns = sum(sum(1 for g in r["segs"] if g["loss"]) for r in out_tr + out_te)
    print(f"训练 {len(out_tr)} | 测试 {len(out_te)} | 平均目标轮数 "
          f"{nturns/max(len(out_tr)+len(out_te),1):.2f}", flush=True)
    if drops:
        print(f"丢弃统计: {drops}", flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out_tr:
            f.write(json.dumps({"segs": r["segs"]}, ensure_ascii=False) + "\n")
    with open(args.out_test, "w", encoding="utf-8") as f:
        for r in out_te:
            f.write(json.dumps({"segs": r["segs"]}, ensure_ascii=False) + "\n")
    print(f"→ {args.out} / {args.out_test}", flush=True)

    # 抽一条打印,肉眼确认渲染格式(侦察时踩过 system 空掉的坑)
    if out_tr:
        print("\n--- 样例 ---", flush=True)
        for g in out_tr[0]["segs"]:
            t = g["t"].replace("\n", "⏎")
            print("  [loss=%d] %s" % (g["loss"], t[:200] + ("…" if len(t) > 200 else "")))


if __name__ == "__main__":
    main()
