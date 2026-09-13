"""eval_v2.py —— 13.1:扩容评测(131 题,两类格式,修正后的判定)

相对旧 20 题 harness 的两处判定升级(针对历史踩坑):
  1. **归一化匹配**:去掉空白/半全角逗号/顿号后再匹配 —— 修 "8,848米"
     这类"内容对、关键词没匹配上"的漏判
  2. **复读假阳检测**:若某次命中只出现在"题目回声"里,记为 suspect 不算
     真阳 —— 修"秦始皇统一六国后…是秦始皇统一六国后"里 "秦" 命中
     "秦始皇" 这类假阳(规则:命中片段周边的上下文若也出现在题干里,
     视为题目回声)
输出:每题的 raw/real/suspect 命中 + 汇总(总体/按格式/按分类/按频率档)。

用法(Spark):
  ~/llm_study/.venv/bin/python eval_v2.py --ckpt ../stage11_datascale/ckpt_11_4_chat.pt \
      --bpe ../stage11_datascale/cache_mm10g/bpe.json --label 11_4_chat \
      --json-out results_v2/11_4_chat.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "stage4_scaling_bpe"))
sys.path.insert(0, str(REPO / "stage9_modern_gpt"))
from bpe import BPETokenizer, EOS_ID                      # noqa: E402
from model_modern import GPT, GPTConfig                   # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IM_END = "<|im_end|>"
NORM_RE = re.compile(r"[\s,，、·．。．]+")


def norm(s):
    return NORM_RE.sub("", s).lower()


def judge(out, answers, question):
    """返回 (raw_hit, real_hit, suspect_hit)。"""
    o, q = norm(out), norm(question)
    raw = real = suspect = False
    for a in answers:
        a_n = norm(a)
        if not a_n:
            continue
        start = 0
        while True:
            i = o.find(a_n, start)
            if i < 0:
                break
            raw = True
            # 命中片段±2 字的上下文若整体出现在题干里 → 题目回声,算 suspect
            ctx = o[max(0, i - 2): i + len(a_n) + 2]
            if a_n in q and ctx in q:
                suspect = True
            else:
                real = True
            start = i + 1
    return raw, real, suspect


@torch.no_grad()
def gen(model, tok, prompt, max_new=80):
    """贪心生成(KV cache 增量解码,快 ~10×;全量重算版见 eval_chat10)。"""
    ids = tok.encode(prompt)
    ctx = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    logits, past = model.forward_cached(ctx, None)        # prefill
    gen_ids = []
    for _ in range(max_new):
        nxt = int(logits[0, -1].argmax().item())
        gen_ids.append(nxt)
        if nxt == EOS_ID or IM_END in tok.decode(gen_ids):
            break
        t = torch.tensor([[nxt]], dtype=torch.long, device=DEVICE)
        logits, past = model.forward_cached(t, past)      # 增量步
    return tok.decode(gen_ids).split(IM_END)[0].strip()


def prompt_of(fmt, q):
    if fmt == "chat":
        return f"<|im_start|>user\n{q}{IM_END}\n<|im_start|>assistant\n"
    return f"问：{q}\n答："


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bpe", required=True)
    ap.add_argument("--qa", default=str(HERE / "qa_v2_freq.jsonl"))
    ap.add_argument("--label", default="")
    ap.add_argument("--formats", default="raw,chat")
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    tok = BPETokenizer.load(args.bpe)
    ckpt = torch.load(args.ckpt)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    label = args.label or Path(args.ckpt).stem
    print(f"=== {label} | {sum(p.numel() for p in model.parameters())/1e6:.1f}M "
          f"| vocab {len(tok)}", flush=True)

    rows = [json.loads(ln) for ln in open(args.qa, encoding="utf-8")]
    report = {"label": label, "ckpt": str(args.ckpt), "n": len(rows),
              "formats": {}}
    for fmt in args.formats.split(","):
        out_rows = []
        for r in rows:
            out = gen(model, tok, prompt_of(fmt, r["q"]), args.max_new)
            raw, real, susp = judge(out, r["a"], r["q"])
            out_rows.append({"q": r["q"], "a": r["a"], "cat": r.get("cat"),
                             "bin": r.get("bin"), "freq": r.get("freq"),
                             "out": out[:120], "raw": raw, "real": real,
                             "suspect": susp})
        n = len(out_rows)
        n_real = sum(x["real"] for x in out_rows)
        n_raw = sum(x["raw"] for x in out_rows)
        n_susp = sum(x["suspect"] for x in out_rows)
        print(f"\n--- [{fmt}] 真阳 {n_real}/{n}(原始命中 {n_raw},嫌疑 {n_susp})")
        by_cat, by_bin = {}, {}
        for x in out_rows:
            by_cat.setdefault(x["cat"], [0, 0])
            by_cat[x["cat"]][1] += 1
            by_cat[x["cat"]][0] += x["real"]
            by_bin.setdefault(x["bin"], [0, 0])
            by_bin[x["bin"]][1] += 1
            by_bin[x["bin"]][0] += x["real"]
        print("   分类:", {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_cat.items())})
        print("   频率档:", {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_bin.items())})
        report["formats"][fmt] = {"real": n_real, "raw": n_raw, "suspect": n_susp,
                                  "by_cat": by_cat, "by_bin": by_bin,
                                  "rows": out_rows}

    if args.json_out:
        p = Path(args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
        print(f"JSON → {p}")


if __name__ == "__main__":
    main()
