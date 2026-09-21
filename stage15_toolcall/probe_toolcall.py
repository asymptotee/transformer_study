"""probe_toolcall.py —— 零样本裸测:现有模型见到 tool 格式会怎样

**动机(stage15 侦察后)**:76,574 条 tool 样本的结构和 README 当初的设想不一样 ——
100% 是单轮用户请求(1 个 user → 1 轮工具 → 1 个答复),**没有多轮**;81% 的对话
只有 1 个可用工具。所以难点不是"多轮历史",也不在"从多个工具里挑",而是:

  ① 生成结构化 `<tool_call>` JSON(全新输出格式)
  ② 读 `<tool_response>` 后给最终答复
  ③ **窗口 1024**(渲染中位 761 token;768 只活 53.7%,512 存活 0.0%)

训练之前先量**零样本 gap**:把带工具定义的 prompt 直接喂给现有模型(它们从没见过
tool 数据),看输出是什么。这一刀切出三种可能,决定后面怎么做:

  · 已经会发 `<tool_call>` 且 JSON 合法   → 只需少量微调对齐细节
  · 只发普通文本(把工具调用当成"回答用户")  → 要从零教格式
  · 输出崩坏/复读                        → 窗口或能力不够,先解决前置问题

**顺带记录 prompt 长度** —— 如果 prompt 本身就超过模型训练时的窗口,那个结果不算数
(是"没见过的长度"而不是"没见过的格式")。

用法(Spark):
  ~/llm_study/.venv/bin/python probe_toolcall.py \
      --ckpt ../stage14_multiturn/ckpt_14_mt.pt --label 14_mt --n 40
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "stage4_scaling_bpe"))
sys.path.insert(0, str(REPO / "stage9_modern_gpt"))
sys.path.insert(0, str(Path.home() / "llm_study" / "minimind"))
from bpe import BPETokenizer, EOS_ID                # noqa: E402
from model_modern import GPT, GPTConfig             # noqa: E402
from transformers import AutoTokenizer              # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IM_END = "<|im_end|>"
THINK = re.compile(r"<think>.*?</think>", re.S)
TOOLCALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
SRC = Path.home() / "llm_study" / "mm_data" / "sft_t2t_mini.jsonl"


def convert(c):
    """语料格式 → (HF 消息列表, tools)。tool_calls 从 JSON 字符串解出来。"""
    msgs, tools = [], None
    for m in c:
        if m["role"] == "system" and m.get("tools"):
            tools = json.loads(m["tools"])
            continue
        d = {"role": m["role"], "content": m.get("content", "")}
        if m.get("tool_calls"):
            d["tool_calls"] = [{"type": "function", "function": x["function"]}
                               for x in json.loads(m["tool_calls"])]
        msgs.append(d)
    return msgs, tools


def gold_call(msgs):
    """金标:第一条带 tool_calls 的 assistant 消息(名字 + 参数)。"""
    for m in msgs:
        if m.get("tool_calls"):
            f = m["tool_calls"][0]["function"]
            return f["name"], f.get("arguments")
    return None, None


def load_items(n, seed=42):
    convs = []
    with open(SRC, encoding="utf-8") as f:
        for ln in f:
            c = json.loads(ln).get("conversations") or []
            if any(m.get("role") == "tool" or m.get("tool_calls") for m in c):
                convs.append(c)
    random.seed(seed)
    random.shuffle(convs)
    return convs[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bpe", default=str(REPO / "stage11_datascale/cache_mm10g/bpe.json"))
    ap.add_argument("--label", default="")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--mode", choices=["full", "notools", "shorttools"],
                    default="full",
                    help="full=原样;notools=只给用户问题(验证模型本身正常);"
                         "shorttools=工具定义砍短(把 prompt 压进 512,"
                         "用来分开'长度'和'格式'两个原因)")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    tok = BPETokenizer.load(args.bpe)
    ckpt = torch.load(args.ckpt)
    model = GPT(GPTConfig(**ckpt["config"])).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    mm = AutoTokenizer.from_pretrained(Path.home() / "llm_study" / "minimind" / "model")
    label = args.label or Path(args.ckpt).stem

    items = load_items(args.n)
    print(f"=== {label} | {len(items)} 条 tool 请求 | max_new {args.max_new}", flush=True)

    n_fmt = n_json = n_name = n_arg = n_plain = 0
    rows = []
    with torch.no_grad():
        for c in items:
            msgs, tools = convert(c)
            # prompt = system(tools) + user 请求,让模型自由生成
            prefix = msgs[:1]   # 见 eval_toolcall.py 的说明:msgs[1] 是金标调用
            if args.mode == "notools":
                prefix, tools = msgs[1:2], None       # 只留用户问题
            elif args.mode == "shorttools":
                # 工具定义砍到只剩名字 + 极短描述,把 prompt 压进 512
                tools = [{"type": "function", "function": {
                    "name": t["function"]["name"],
                    "description": (t["function"].get("description") or "")[:12],
                    "parameters": {"type": "object", "properties": {}}}}
                    for t in tools]
            prompt = THINK.sub("", mm.apply_chat_template(
                prefix, tools=tools, tokenize=False, add_generation_prompt=True))
            plen = len(tok.encode(prompt))
            ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=DEVICE)
            logits, past = model.forward_cached(ids, None)
            gen = []
            for _ in range(args.max_new):
                nxt = int(logits[0, -1].argmax().item())
                if nxt == EOS_ID or IM_END in tok.decode(gen):
                    break
                gen.append(nxt)
                logits, past = model.forward_cached(
                    torch.tensor([[nxt]], dtype=torch.long, device=DEVICE), past)
            out = tok.decode(gen).split(IM_END)[0].strip()

            gname, garg = gold_call(msgs)
            has_fmt = bool(TOOLCALL.search(out))
            ok_json = ok_name = ok_arg = False
            if has_fmt:
                try:
                    obj = json.loads(TOOLCALL.search(out).group(1))
                    ok_json = True
                    ok_name = obj.get("name") == gname
                    if ok_name and garg:
                        try:
                            a1 = obj.get("arguments")
                            a1 = json.loads(a1) if isinstance(a1, str) else a1
                            ok_arg = a1 == json.loads(garg)
                        except Exception:
                            pass
                except Exception:
                    pass
            plain = not has_fmt
            n_fmt += has_fmt; n_json += ok_json; n_name += ok_name
            n_arg += ok_arg; n_plain += plain
            rows.append({"q": msgs[1]["content"][:80], "prompt_tok": plen,
                         "gold": gname, "out": out[:300], "has_tool_call": has_fmt,
                         "json_ok": ok_json, "name_ok": ok_name, "arg_ok": ok_arg})

    n = len(rows)
    print(f"\n--- [{label}] n={n}")
    print(f"    发出 <tool_call> 标签 : {n_fmt:3d}/{n} = {n_fmt/n:5.1%}")
    print(f"    标签内 JSON 可解析    : {n_json:3d}/{n} = {n_json/n:5.1%}")
    print(f"    工具名正确            : {n_name:3d}/{n} = {n_name/n:5.1%}")
    print(f"    参数完全正确          : {n_arg:3d}/{n} = {n_arg/n:5.1%}")
    print(f"    完全没发标签(当普通问句答): {n_plain:3d}/{n} = {n_plain/n:5.1%}")
    pl = sorted(x["prompt_tok"] for x in rows)
    print(f"    prompt token: 中位 {pl[n//2]} | 最大 {pl[-1]}"
          f"   ← 超过 512 的有 {sum(1 for x in pl if x > 512)}/{n}")
    print("\n    样例输出:")
    for x in rows[:4]:
        print("      [gold=%s] %r" % (x["gold"], x["out"][:110]))

    if args.json_out:
        p = Path(args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"label": label, "n": n, "fmt": n_fmt, "json_ok": n_json,
                       "name_ok": n_name, "arg_ok": n_arg, "plain": n_plain,
                       "rows": rows}, f, ensure_ascii=False, indent=1)
        print(f"JSON → {p}")


if __name__ == "__main__":
    main()
