"""eval_reason.py —— stage17 逐档评测(三个任务同口径)

**判据**:给题面(到答案之前),贪心生成,和唯一答案比对。
答案都是程序生成的唯一字符串(字母 / 数字序列),**不需要 judge** ——
这是本任务最大的优势(stage16 那条线已经吃过五次代理指标骗人的亏)。

三个数分开记:
  · 完全正确   —— 按第一个答案比对(模型可能会复读,所以取第一次出现)
  · 崩坏(复读) —— 输出退化
  · 空输出     —— 一个 token 就收尾

**为什么取"第一个答案"**:stage16 测 Holmes 时踩过 —— 它会无限复读,
如果按"输出里有没有出现正确答案"判,一旦碰巧复读到就误判成会做。

⚠️ 生成循环照抄 `stage13_eval/eval_v2.py` 的 `gen()`。本项目在此栽过一次
   (`logits, past = ...` 写成 `lg, past = ...`,导致每步都用同一个 argmax,
   全部输出变成单 token 复读,看起来像"模型完全崩了")。

用法(Spark):
  ~/llm_study/.venv/bin/python eval_reason.py --ckpt ckpt_17_relation.pt \
      --task relation --label 17_relation --json-out results/eval_17_relation.json
"""

import argparse
import json
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
    """贪心生成 —— 结构照抄 eval_v2.gen(),别改(见文件头警告)。"""
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
    c, n = Counter(txt).most_common(1)[0]
    return n / len(txt) > 0.7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", required=True, choices=["relation", "transform", "state"])
    ap.add_argument("--bpe", default=str(REPO / "stage11_datascale/cache_mm10g/bpe.json"))
    ap.add_argument("--eval-file", default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    tok = BPETokenizer.load(args.bpe)
    ck = torch.load(args.ckpt)
    model = GPT(GPTConfig(**ck["config"])).to(DEVICE)
    model.load_state_dict(ck["model"])
    model.eval()
    label = args.label or Path(args.ckpt).stem
    ev = Path(args.eval_file) if args.eval_file else HERE / f"{args.task}_eval.jsonl"
    rows = [json.loads(l) for l in open(ev, encoding="utf-8")]

    # 有 theme 字段(泛化测试)就按 theme 分组,否则按难度档
    key = "theme" if rows and "theme" in rows[0] else "level"
    by, detail = {}, []
    for r in rows:
        out = gen(model, tok, r["q"], args.max_new)
        # 取**第一个**答案 —— 防复读蒙对
        exact = out.strip() == r["a"]
        k = r.get(key, 0)
        by.setdefault(k, [0, 0, 0, 0])          # 对/空/崩坏/总
        by[k][3] += 1
        by[k][0] += exact
        by[k][1] += (len(out) == 0)
        by[k][2] += degenerate(out)
        detail.append({key: k, "q": r["q"], "want": r["a"],
                       "got": out, "exact": exact})

    print(f"=== {label} | {args.task} | 按{'措辞' if key=='theme' else '难度档'}"
          f"（取第一个答案)===")
    print("  %-22s %10s %10s %10s" % ("措辞/档位", "答对", "空输出", "复读"))
    tot = [0, 0, 0, 0]
    for k in sorted(by, key=lambda x: (isinstance(x, str), x)):
        a, e, d, n = by[k]
        tot = [tot[0] + a, tot[1] + e, tot[2] + d, tot[3] + n]
        print("  %-22s %6d/%-3d %7d/%-3d %7d/%-3d" % (k, a, n, e, n, d, n))
    print("  %-22s %6d/%-3d %7d/%-3d %7d/%-3d"
          % ("合计", tot[0], tot[3], tot[1], tot[3], tot[2], tot[3]))
    print("\n  样例:")
    for x in detail[:1] + [y for y in detail if not y["exact"]][:3]:
        print("    %s | %s → 期望 %r | 实得 %r"
              % (x[key], x["q"][:48], x["want"], x["got"][:26]))

    if args.json_out:
        p = Path(args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"label": label, "task": args.task,
                       "by_level": {str(k): v for k, v in by.items()},
                       "rows": detail}, f, ensure_ascii=False, indent=1)
        print(f"JSON → {p}")


if __name__ == "__main__":
    main()
