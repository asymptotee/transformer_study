"""eval_toolcall.py —— 工具调用评测(两阶段)

工具链的核心是两件事,分开测:

  **阶段 1「调用」** —— 给 system(工具定义) + user 请求,看它会不会/对不对地发
    `<tool_call>`。指标:发标签率 / JSON 可解析 / 工具名 / 参数完全正确。

  **阶段 2「收尾」** —— 给 system + user + **金标**调用 + **金标**工具结果,
    看它拿到结果后能不能给出正确的最终答复。这一阶段**才是工具链的核心**
    ("拿到结果再作答"),而且它把阶段 1 的错误排除掉了:即使模型自己调不对,
    只要读得懂给它的结果,就能在这里拿分。

    **怎么自动判对错**:答复里必须出现**最后一轮工具结果**里的关键值。
    工具结果是 JSON,把其中的标量抽出来(数字转字符串、字符串取长度≥3 的),
    答复命中其一即算读到。这是规则可判的,不需要人看。

    **同时检测崩坏**:如果输出是单 token 复读(同一个 token 占比 > 80%),
    单独记为 degenerate —— pass@k 那次教训:汇总数字会把"噪声里蹭到关键词"
    算成命中,所以要把崩坏和"答对"分开统计。

⚠️ 生成循环的写法照抄 `stage13_eval/eval_v2.py` 的 `gen()`。
   **不要自己改** —— stage15 的探针第一版把 `logits, past = ...` 写成了
   `lg, past = ...`,于是每一步都用 prefill 的同一个 argmax,**所有输出都是
   同一个 token 无限重复**,看起来像"模型完全崩了",实际是笔误。

用法(Spark):
  ~/llm_study/.venv/bin/python eval_toolcall.py \
      --ckpt ../stage14_multiturn/ckpt_14_mt.pt --label 14_mt --n 200
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
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


@torch.no_grad()
def gen(model, tok, prompt, max_new):
    """贪心生成 —— 结构与 eval_v2.gen() 一致,别改(见文件头警告)。"""
    ids = tok.encode(prompt)
    ctx = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    logits, past = model.forward_cached(ctx, None)      # prefill
    gen_ids = []
    for _ in range(max_new):
        nxt = int(logits[0, -1].argmax().item())
        gen_ids.append(nxt)
        if nxt == EOS_ID or IM_END in tok.decode(gen_ids):
            break
        t = torch.tensor([[nxt]], dtype=torch.long, device=DEVICE)
        logits, past = model.forward_cached(t, past)    # 增量步
    return tok.decode(gen_ids).split(IM_END)[0].strip()


def to_hf(convs):
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


def scalars(obj, out=None):
    """递归收集 JSON 里的标量(值,不是键)。"""
    out = [] if out is None else out
    if isinstance(obj, dict):
        for v in obj.values():
            scalars(v, out)
    elif isinstance(obj, list):
        for v in obj:
            scalars(v, out)
    else:
        out.append(obj)
    return out


def parse_tool_content(txt):
    """工具结果的 content **不一定是单个 JSON** —— 可以是多个换行分隔的 JSON
    (一条对话调了两个工具时: `{"result": 69}\\n{"rate": 7.21, ...}`)。
    逐行解,失败再整体解。踩过一次:直接 json.loads 报 "Extra data"。"""
    objs = []
    for line in txt.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            objs.append(json.loads(line))
        except Exception:
            pass
    if not objs:
        try:
            objs = [json.loads(txt)]
        except Exception:
            objs = []
    return objs


def key_values(tool_content, prior_text=""):
    """关键值 = 工具结果里的标量,**但要排除前文(用户请求 / 调用参数)里已经出现过的**。

    否则会高估:汇率结果里 `"from_currency": "USD"` 的 "USD" 同时也写在调用参数里,
    不读结果也会答出来 —— 那不算"读懂了工具结果"。只有"只可能来自结果"的值才算数。
    """
    vals = []
    for o in parse_tool_content(tool_content):
        for s in scalars(o):
            t = ("%g" % s) if isinstance(s, float) else str(s).strip()
            if not (len(t) >= 3 or (t.isdigit() and len(t) >= 2 and t != "0")):
                continue
            if prior_text and t in prior_text:
                continue
            vals.append(t)
    return vals


def degenerate(txt):
    """单 token 复读检测:最高频 token 占比 > 80%(字符级近似)。"""
    if len(txt) < 20:
        return False
    ch, n = Counter(txt).most_common(1)[0]
    return n / len(txt) > 0.8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bpe", default=str(REPO / "stage11_datascale/cache_mm10g/bpe.json"))
    ap.add_argument("--src", default=str(Path.home() / "llm_study" / "mm_data"
                                         / "sft_t2t_mini.jsonl"))
    ap.add_argument("--label", default="")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    tok = BPETokenizer.load(args.bpe)
    ckpt = torch.load(args.ckpt)
    model = GPT(GPTConfig(**ckpt["config"])).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    mm = AutoTokenizer.from_pretrained(Path.home() / "llm_study" / "minimind" / "model")
    label = args.label or Path(args.ckpt).stem

    convs = []
    with open(args.src, encoding="utf-8") as f:
        for ln in f:
            c = json.loads(ln)["conversations"]
            if any(m.get("role") == "tool" or m.get("tool_calls") for m in c):
                if any(m["role"] == "system" and m.get("tools") for m in c):
                    convs.append(c)
    random.seed(args.seed)
    random.shuffle(convs)
    items = convs[:args.n]
    print(f"=== {label} | n={len(items)} | max_new {args.max_new}", flush=True)

    r1_fmt = r1_json = r1_name = r1_arg = 0
    r2_hit = r2_deg = 0
    r2_den = 0
    rows = []
    for k, c in enumerate(items):
        msgs, tools = to_hf(c)
        ai = [i for i, m in enumerate(msgs) if m.get("tool_calls")]
        ti = [i for i, m in enumerate(msgs) if m["role"] == "tool"]
        if not ai or not ti:
            continue
        first = ai[0]
        # ---- 阶段 1:给 tools + user,自由生成 ----
        # ⚠️ 必须 msgs[:1] 而不是 msgs[:2]:to_hf() 跳过了 system 消息,所以
        # msgs[0]=user、msgs[1]=assistant(带**金标 tool_call**)。写 msgs[:2]
        # 等于把正确答案放进 prompt,让模型"在金标调用之后接着写"——
        # 那个数衡量的是别的东西。踩过:test loss 0.124 却只 9% 参数正确,
        # 这个矛盾就是线索(几乎背下格式的模型不可能发不对调用)。
        p1 = THINK.sub("", mm.apply_chat_template(
            msgs[:1], tools=tools, tokenize=False, add_generation_prompt=True))
        o1 = gen(model, tok, p1, args.max_new)
        m1 = TOOLCALL.search(o1)
        ok_json = ok_name = ok_arg = False
        if m1:
            r1_fmt += 1
            try:
                obj = json.loads(m1.group(1))
                ok_json = True
                g = msgs[first]["tool_calls"][0]["function"]
                ok_name = obj.get("name") == g["name"]
                if ok_name:
                    a1 = obj.get("arguments")
                    a1 = json.loads(a1) if isinstance(a1, str) else a1
                    ok_arg = a1 == json.loads(g["arguments"])
            except Exception:
                pass
        r1_json += ok_json; r1_name += ok_name; r1_arg += ok_arg
        # ---- 阶段 2:给到**最后一个工具结果**为止,看收尾 ----
        last_tool = ti[-1]
        p2 = THINK.sub("", mm.apply_chat_template(
            msgs[:last_tool + 1], tools=tools, tokenize=False,
            add_generation_prompt=True))
        o2 = gen(model, tok, p2, args.max_new)
        prior = " ".join(m.get("content", "") + json.dumps(
            m.get("tool_calls", ""), ensure_ascii=False) for m in msgs[:last_tool])
        kv = key_values(msgs[last_tool]["content"], prior)
        deg = degenerate(o2)
        hit = bool(kv) and any(v in o2 for v in kv)
        r2_den += 1; r2_hit += hit and not deg; r2_deg += deg
        rows.append({"q": msgs[1]["content"][:80], "gold": msgs[first]["tool_calls"][0]["function"]["name"],
                     "stage1": o1[:200], "stage1_fmt": bool(m1), "stage1_json": ok_json,
                     "stage1_name": ok_name, "stage1_arg": ok_arg,
                     "key_vals": kv, "stage2": o2[:300], "stage2_hit": hit,
                     "stage2_degenerate": deg})
        if (k + 1) % 50 == 0:
            print(f"  …{k+1}/{len(items)}", flush=True)

    n = max(r2_den, 1)
    print(f"\n--- [{label}] 阶段1(调用)")
    print(f"    发出 <tool_call> : {r1_fmt:3d}/{len(items)} = {r1_fmt/len(items):5.1%}")
    print(f"    JSON 可解析      : {r1_json:3d}/{len(items)} = {r1_json/len(items):5.1%}")
    print(f"    工具名正确       : {r1_name:3d}/{len(items)} = {r1_name/len(items):5.1%}")
    print(f"    参数完全正确     : {r1_arg:3d}/{len(items)} = {r1_arg/len(items):5.1%}")
    print(f"--- [{label}] 阶段2(收尾,给金标调用+结果)")
    print(f"    答复含结果关键值 : {r2_hit:3d}/{n} = {r2_hit/n:5.1%}")
    print(f"    输出崩坏(复读) : {r2_deg:3d}/{n} = {r2_deg/n:5.1%}")

    if args.json_out:
        p = Path(args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"label": label, "n": len(items), "s1_fmt": r1_fmt,
                       "s1_json": r1_json, "s1_name": r1_name, "s1_arg": r1_arg,
                       "s2_hit": r2_hit, "s2_deg": r2_deg, "s2_den": n,
                       "rows": rows}, f, ensure_ascii=False, indent=1)
        print(f"JSON → {p}")


if __name__ == "__main__":
    main()
