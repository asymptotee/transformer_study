"""eval_usability.py —— 13.2:可用性指标(自然收尾率 + 生成长度)

**为什么需要它**:131 题那套指标只看"第一句里有没有关键答案"(判分规则见
judge.py —— 匹配区间截断到第一个自问自答标记之前),**完全不管后面写成什么样**。

于是出现了一个反直觉的结果:

| 模型 | 131 题 chat | 实际体验 |
|---|---|---|
| 11_5_chat_2000 | **83**(最高) | 长、拖尾、**停不住** |
| 14_mt | 74 | 短、完整、能用 |

**chat 分最高的模型可用性最差** —— 两个指标在这批模型上是反着的。根因是数据:
单轮 SFT 语料的答案中位 580 字符(82% 超 300),多轮语料的目标轮中位只有
110 字符(16% 超 300);模型学到"要写长",而 126M 维持不了长文连贯 → 滑进
复读 → **复读时永远不发收尾标记**。

所以本脚本补一个正交的维度:**给定充足预算,它能不能自己停下来?**

  · 自然收尾率 = 生成了 <|im_end|>/EOS/<|im_start|> 而主动停止的比例
  · 长度:token 数中位数(给定预算内)

口径与 eval_v2 一致(贪心、同题库、同 chat 模板),只是预算给到 400 —— 必须
大于训练中位长度,否则测到的是"预算不够"而不是"停不住"。

用法(Spark):
  ~/llm_study/.venv/bin/python eval_usability.py \
      --ckpt ../stage11_datascale/ckpt_11_5_chat_2000.pt --label 11_5_chat_2000 \
      --bpe ../stage11_datascale/cache_mm10g/bpe.json --json-out results_u/xxx.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "stage4_scaling_bpe"))
sys.path.insert(0, str(REPO / "stage9_modern_gpt"))
from bpe import BPETokenizer, EOS_ID                       # noqa: E402
from model_modern import GPT, GPTConfig                    # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IM_END = "<|im_end|>"
IM_ANY = "<|im"          # 任一模板标记都算"它想收尾"(模型偶尔吐 <|im_start|>)


@torch.no_grad()
def gen_tracked(model, tok, prompt, max_new):
    """贪心生成,返回 (文本, 是否自然收尾, 生成 token 数)。"""
    ids = tok.encode(prompt)
    ctx = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    logits, past = model.forward_cached(ctx, None)
    gen, stopped = [], False
    for _ in range(max_new):
        nxt = int(logits[0, -1].argmax().item())
        gen.append(nxt)
        txt = tok.decode(gen)
        if IM_ANY in txt or nxt == EOS_ID:
            stopped = True
            break
        tk = torch.tensor([[nxt]], dtype=torch.long, device=DEVICE)
        logits, past = model.forward_cached(tk, past)
    return tok.decode(gen).split(IM_ANY)[0].strip(), stopped, len(gen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bpe", required=True)
    ap.add_argument("--qa", default=str(HERE / "qa_v2_freq.jsonl"))
    ap.add_argument("--label", default="")
    ap.add_argument("--format", choices=["chat", "raw"], default="chat")
    ap.add_argument("--max-new", type=int, default=400,
                    help="必须大于训练中位长度,否则测到的是预算不够")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    tok = BPETokenizer.load(args.bpe)
    ckpt = torch.load(args.ckpt)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    label = args.label or Path(args.ckpt).stem

    rows = [json.loads(ln) for ln in open(args.qa, encoding="utf-8")]
    out_rows = []
    for i, r in enumerate(rows):
        if args.format == "chat":
            p = f"<|im_start|>user\n{r['q']}<|im_end|>\n<|im_start|>assistant\n"
        else:
            p = f"问：{r['q']}\n答："
        txt, stopped, n = gen_tracked(model, tok, p, args.max_new)
        out_rows.append({"q": r["q"], "stopped": stopped, "ntok": n,
                         "chars": len(txt)})
        if (i + 1) % 40 == 0:
            print(f"  …{i+1}/{len(rows)}", flush=True)

    n = len(out_rows)
    ns = sum(x["stopped"] for x in out_rows)
    toks = sorted(x["ntok"] for x in out_rows)
    chs = sorted(x["chars"] for x in out_rows)
    print(f"\n=== {label} [{args.format}] 预算 {args.max_new} ===")
    print(f"  自然收尾率: {ns}/{n} = {ns/n:.1%}")
    print(f"  token 中位 {toks[n//2]} | 90% {toks[int(n*0.9)]} | 最大 {toks[-1]}")
    print(f"  字符中位 {chs[n//2]} | 90% {chs[int(n*0.9)]}")
    rep = {"label": label, "ckpt": str(args.ckpt), "format": args.format,
           "max_new": args.max_new, "n": n, "stopped": ns,
           "stopped_rate": ns / n, "ntok_median": toks[n // 2],
           "chars_median": chs[n // 2], "rows": out_rows}
    if args.json_out:
        p = Path(args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=1)
        print(f"JSON → {p}")


if __name__ == "__main__":
    main()
