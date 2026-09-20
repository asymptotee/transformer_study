"""eval_knowledge_minimind.py —— 把 **minimind 官方权重**放进我们的 131 题尺子

用途(2026-09-20):我们一直在跟自己比,从没跟官方模型比过。而"要不要把 SFT
数据量对齐到他们(905,718 条 × 2 epochs = 113,000 步,是我们的 56 倍)"这个
问题,可以直接测出来:

  · 他们没比我们强 → SFT 体积不是杠杆,省下 ~47 小时
  · 他们明显更强   → 长期低 lr 的 SFT 确实有别的收益,值得研究

**口径必须与 eval_v2.py 完全一致**,否则比的不是模型:
  · 同 131 题(qa_v2_freq.jsonl)、同 chat 模板、同**贪心**、同 **max_new=80**
  · 判分复用 `judge.py`(纯文本、无 tokenizer 依赖,所以能跨词表用)
  · 他们的 BPE 是 6,400,我们 26,566 —— 但判分在**文本层**做,不受影响

用法(Spark):
  ~/llm_study/.venv/bin/python eval_knowledge_minimind.py \
      --weight ~/llm_study/mm_weights/minimind-3-pytorch/full_sft_768.pth \
      --label mm_full_sft --json-out results_u/know_mm_full_sft.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
MM = Path.home() / "llm_study" / "minimind"
sys.path.insert(0, str(MM))
sys.path.insert(0, str(HERE))
from transformers import AutoTokenizer                       # noqa: E402
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM   # noqa: E402
from judge import judge                                      # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", required=True)
    ap.add_argument("--qa", default=str(HERE / "qa_v2_freq.jsonl"))
    ap.add_argument("--label", default="")
    ap.add_argument("--hidden-size", type=int, default=768)
    ap.add_argument("--num-hidden-layers", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=80,
                    help="必须与 eval_v2.py 一致(默认 80),否则换尺子了")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MM / "model")
    model = MiniMindForCausalLM(MiniMindConfig(
        hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
        use_moe=False))
    model.load_state_dict(torch.load(args.weight, map_location=DEVICE), strict=True)
    model = model.half().eval().to(DEVICE)
    label = args.label or Path(args.weight).stem
    print(f"=== {label} | {sum(p.numel() for p in model.parameters())/1e6:.1f}M "
          f"| vocab {len(tok)} | max_new {args.max_new}(同 eval_v2)", flush=True)

    rows = [json.loads(ln) for ln in open(args.qa, encoding="utf-8")]
    out_rows = []
    with torch.no_grad():
        for i, r in enumerate(rows):
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": r["q"]}],
                tokenize=False, add_generation_prompt=True)
            ids = tok(prompt, return_tensors="pt").to(DEVICE)
            gen = model.generate(
                inputs=ids["input_ids"], attention_mask=ids["attention_mask"],
                max_new_tokens=args.max_new, do_sample=False, repetition_penalty=1.0,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            txt = tok.decode(gen[0][ids["input_ids"].shape[1]:],
                             skip_special_tokens=True).strip()
            raw, real, susp = judge(txt, r["a"], r["q"])
            out_rows.append({"q": r["q"], "a": r["a"], "cat": r.get("cat"),
                             "bin": r.get("bin"), "out": txt[:300],
                             "raw": raw, "real": real, "suspect": susp})
            if (i + 1) % 40 == 0:
                print(f"  …{i+1}/{len(rows)}", flush=True)

    n = len(out_rows)
    nreal = sum(x["real"] for x in out_rows)
    nraw = sum(x["raw"] for x in out_rows)
    by_cat, by_bin = {}, {}
    for x in out_rows:
        by_cat.setdefault(x["cat"], [0, 0]); by_cat[x["cat"]][1] += 1
        by_cat[x["cat"]][0] += x["real"]
        by_bin.setdefault(x["bin"], [0, 0]); by_bin[x["bin"]][1] += 1
        by_bin[x["bin"]][0] += x["real"]
    print(f"\n--- [{label}] 真阳 {nreal}/{n}(原始命中 {nraw})")
    print("   分类:", {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_cat.items())})
    print("   频率档:", {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_bin.items())})
    if args.json_out:
        p = Path(args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"label": label, "weight": args.weight, "n": n,
                       "real": nreal, "raw": nraw, "by_cat": by_cat,
                       "by_bin": by_bin, "rows": out_rows}, f,
                      ensure_ascii=False, indent=1)
        print(f"JSON → {p}")


if __name__ == "__main__":
    main()
