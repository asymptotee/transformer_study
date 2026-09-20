"""eval_usability_minimind.py —— 用同一口径测 **minimind 官方权重**的可用性

为什么需要它(2026-09-20):我们一直在纠结"单轮数据把模型训成列举机器、
收尾率只有 45%",而 minimind 官方模型从没表现出这个问题 —— 查代码发现
**他们的 full_sft 直接用整个 sft_t2t_mini.jsonl(单轮+多轮混在一起),
从来没做过单轮过滤**;而我们的 chat_train_nt 是从"2 条消息的对话"里筛出来的
纯单孔子集(41% 含列举、中位 538 字符)。

所以问题变成:**混合训练到底能不能避免列举循环?** 他们的官方权重是现成的
参照物 —— 用同一个探针(同 131 题、同 chat 模板、同贪心、同 400 token 预算、
同重复惩罚 1.1),测他们,就能在我们自己花 40 分钟跑混训之前先知道答案。

  · 他们官方模型也停不住 → 问题在这份数据/这个规模,混合不解决
  · 他们能停住       → 混合是正解,我们的纯单轮筛选是自找的

注意规模差异:他们是 64M(764M? 见下),我们是 126.6M —— 所以绝对数字不能
直接比,**看的是"能不能停"这个定性**。

用法(Spark):
  ~/llm_study/.venv/bin/python eval_usability_minimind.py \
      --weight ~/llm_study/mm_weights/minimind-3-pytorch/full_sft_768.pth \
      --label mm_full_sft --json-out results_u/mm_full_sft.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).parent
MM = Path.home() / "llm_study" / "minimind"
sys.path.insert(0, str(MM))
from transformers import AutoTokenizer                       # noqa: E402
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM   # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", required=True, help=".pth 权重路径")
    ap.add_argument("--qa", default=str(HERE / "qa_v2_freq.jsonl"))
    ap.add_argument("--label", default="")
    ap.add_argument("--hidden-size", type=int, default=768)
    ap.add_argument("--num-hidden-layers", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--rep-penalty", type=float, default=1.1,
                    help="与我们自己的探针一致;他们的默认是 1.0")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MM / "model")
    model = MiniMindForCausalLM(MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=False))
    model.load_state_dict(torch.load(args.weight, map_location=DEVICE), strict=True)
    model = model.half().eval().to(DEVICE)
    label = args.label or Path(args.weight).stem
    n_par = sum(p.numel() for p in model.parameters())
    print(f"=== {label} | {n_par/1e6:.1f}M | vocab {len(tok)} | "
          f"预算 {args.max_new} | rep {args.rep_penalty}", flush=True)

    rows = [json.loads(ln) for ln in open(args.qa, encoding="utf-8")]
    out_rows = []
    t0 = time.time()
    with torch.no_grad():
        for i, r in enumerate(rows):
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": r["q"]}],
                tokenize=False, add_generation_prompt=True)
            ids = tok(prompt, return_tensors="pt").to(DEVICE)
            gen = model.generate(
                inputs=ids["input_ids"], attention_mask=ids["attention_mask"],
                max_new_tokens=args.max_new, do_sample=False,
                repetition_penalty=args.rep_penalty,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            n_gen = gen.shape[1] - ids["input_ids"].shape[1]
            txt = tok.decode(gen[0][ids["input_ids"].shape[1]:],
                             skip_special_tokens=True)
            out_rows.append({"q": r["q"], "stopped": n_gen < args.max_new,
                             "ntok": int(n_gen), "chars": len(txt)})
            if (i + 1) % 40 == 0:
                print(f"  …{i+1}/{len(rows)}", flush=True)

    n = len(out_rows)
    ns = sum(x["stopped"] for x in out_rows)
    toks = sorted(x["ntok"] for x in out_rows)
    chs = sorted(x["chars"] for x in out_rows)
    print(f"\n=== {label} 自然收尾率 {ns}/{n} = {ns/n:.1%} | "
          f"token 中位 {toks[n//2]} | 字符中位 {chs[n//2]} | "
          f"用时 {time.time()-t0:.0f}s ===")
    if args.json_out:
        p = Path(args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"label": label, "weight": args.weight, "n": n,
                       "stopped": ns, "stopped_rate": ns / n,
                       "ntok_median": toks[n // 2], "chars_median": chs[n // 2],
                       "rows": out_rows}, f, ensure_ascii=False, indent=1)
        print(f"JSON → {p}")


if __name__ == "__main__":
    main()
