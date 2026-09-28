"""eval_arith.py —— 逐档算术准确率(难度曲线)

**判什么**:给 `3 7 + 8 4 =` 这样的题面,贪心生成,和正确答案**逐位比较**。
分开记三个数,因为它们的含义完全不同:

  · **完全正确**   —— 真算对了
  · **长度对但数错** —— 至少知道答案是几位数(说明它不是在瞎发 token)
  · **崩坏(复读)**  —— 输出了退化文本,单独统计

这个三分法是 pass@k 那次的教训:**汇总数字会把"噪声里蹭到"算成"会了"**,
所以要把"会"和"看起来像会"分开。

⚠️ 生成循环的写法照抄 `stage13_eval/eval_v2.py` 的 `gen()`。
   别自己改 —— 本项目已经栽过一次(`logits, past = ...` 写成 `lg, past = ...`,
   导致每一步都用 prefill 的同一个 argmax,全部输出变成同一个 token 无限重复)。

用法(Spark):
  ~/llm_study/.venv/bin/python eval_arith.py --ckpt ckpt_16_arith.pt --label 16_arith
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch

HERE = Path(__file__).parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "stage4_scaling_bpe"))
sys.path.insert(0, str(REPO / "stage9_modern_gpt"))
from bpe import BPETokenizer, EOS_ID                # noqa: E402
from model_modern import GPT, GPTConfig             # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IM_END = "<|im_end|>"


@torch.no_grad()
def gen(model, tok, prompt, max_new):
    """贪心生成 —— 结构与 eval_v2.gen() 一致,别改(见文件头警告)。"""
    ids = tok.encode(prompt)
    ctx = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    logits, past = model.forward_cached(ctx, None)      # prefill
    out = []
    for _ in range(max_new):
        nxt = int(logits[0, -1].argmax().item())
        out.append(nxt)
        if nxt == EOS_ID or IM_END in tok.decode(out) or "\n" in tok.decode(out):
            break
        logits, past = model.forward_cached(
            torch.tensor([[nxt]], dtype=torch.long, device=DEVICE), past)
    return tok.decode(out).split(IM_END)[0].split("\n")[0].strip()


def degenerate(txt):
    if len(txt) < 12:
        return False
    ch, n = Counter(txt).most_common(1)[0]
    return n / len(txt) > 0.8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bpe", default=str(REPO / "stage11_datascale/cache_mm10g/bpe.json"))
    ap.add_argument("--eval-file", default=str(HERE / "arith_eval.jsonl"))
    ap.add_argument("--label", default="")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    tok = BPETokenizer.load(args.bpe)
    ck = torch.load(args.ckpt)
    model = GPT(GPTConfig(**ck["config"])).to(DEVICE)
    model.load_state_dict(ck["model"])
    model.eval()
    label = args.label or Path(args.ckpt).stem

    rows = [json.loads(l) for l in open(args.eval_file, encoding="utf-8")]
    by = {}
    detail = []
    for r in rows:
        out = gen(model, tok, r["q"] + " ", args.max_new)
        got = re.sub(r"\s+", "", out)
        want = r["a"]
        exact = got == want
        samelen = len(got) == len(want)
        by.setdefault(r["d"], [0, 0, 0, 0])          # 正确/长度对/崩坏/总数
        by[r["d"]][3] += 1
        by[r["d"]][0] += exact
        by[r["d"]][1] += samelen
        by[r["d"]][2] += degenerate(out)
        if len(detail) < 3 or not exact:
            pass
        detail.append({"d": r["d"], "q": r["q"], "want": want, "got": got,
                       "exact": exact, "samelen": samelen})

    print(f"=== {label} | 逐档准确率(贪心,逐位比较)===")
    print("  %-6s %8s %10s %10s" % ("位数", "完全正确", "长度对", "崩坏"))
    tot = [0, 0, 0, 0]
    for d in sorted(by):
        a, b, c, n = by[d]
        tot = [tot[0] + a, tot[1] + b, tot[2] + c, tot[3] + n]
        print("  %-6d %5d/%-3d %9d/%-3d %9d/%-3d"
              % (d, a, n, b, n, c, n))
    print("  %-6s %5d/%-3d %9d/%-3d %9d/%-3d"
          % ("合计", tot[0], tot[3], tot[1], tot[3], tot[2], tot[3]))

    print("\n  样例:")
    for x in detail[:2] + [y for y in detail if not y["exact"]][:3]:
        print("    %-6s %s → 期望 %s | 实得 %r" % ("%d位" % x["d"], x["q"], x["want"], x["got"][:20]))

    if args.json_out:
        p = Path(args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"label": label, "by_rung": {str(k): v for k, v in by.items()},
                       "rows": detail}, f, ensure_ascii=False, indent=1)
        print(f"JSON → {p}")


if __name__ == "__main__":
    main()
