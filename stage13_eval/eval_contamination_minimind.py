"""eval_contamination_minimind.py —— 在 minimind 官方权重上跑多轮测试

**Step 0 的目的(2026-09-20)**:我们准备做混训版,配方是"自然比例"
(77.8% 单轮 / 22.2% 多轮,照他们的来)。但有个前提没验:

    **自然比例这个剂量,够不够保住多轮能力?**

他们的 full_sft 就是自然比例训出来的 —— 直接测它就知道。如果他们多轮很好,
说明 22% 的多轮占比够用,我们的混训可以做得便宜;如果他们也差,说明得多加多轮。

与我们自己的 eval_contamination.py 的差别:
  · 模型是他们的(MiniMindForCausalLM),生成走 HF 的 model.generate
  · **单轮基线在本脚本内现算**,不依赖外部 json —— 保证两次生成参数完全一致
  · 口径与 eval_v2 一致:贪心、max_new=80、同 131 题
  · think 要剥:他们的 chat_template 会给每个 assistant 轮插 `<think></think>`,
    而训练时 80% 概率剥掉(post_processing_chat)—— 剥掉是多数情形

用法(Spark):
  ~/llm_study/.venv/bin/python eval_contamination_minimind.py \
      --weight ~/llm_study/mm_weights/minimind-3-pytorch/full_sft_768.pth \
      --mode contam --label mm_full_sft
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
MM = Path.home() / "llm_study" / "minimind"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(MM))
from transformers import AutoTokenizer                        # noqa: E402
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM    # noqa: E402
from judge import judge                                       # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THINK_RE = re.compile(r"<think>.*?</think>", re.S)
RECALL_Q = "我上一个问题的答案是什么？请直接回答。"


def make_partners(n, seed=42):
    rng = random.Random(seed)
    idx = list(range(n))
    while True:
        rng.shuffle(idx)
        if all(idx[i] != i for i in range(n)):
            return idx


def render(tok, msgs, strip_think=True):
    t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return THINK_RE.sub("", t) if strip_think else t


def gen(model, tok, prompt, max_new, rep):
    ids = tok(prompt, return_tensors="pt").to(DEVICE)
    out = model.generate(inputs=ids["input_ids"],
                         attention_mask=ids["attention_mask"],
                         max_new_tokens=max_new, do_sample=False,
                         repetition_penalty=rep,
                         pad_token_id=tok.pad_token_id,
                         eos_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", required=True)
    ap.add_argument("--qa", default=str(HERE / "qa_v2_freq.jsonl"))
    ap.add_argument("--label", default="")
    ap.add_argument("--mode", choices=["contam", "recall"], default="contam")
    ap.add_argument("--hidden-size", type=int, default=768)
    ap.add_argument("--num-hidden-layers", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rep-penalty", type=float, default=1.0,
                    help="1.0=纯贪心,与我们 eval_contamination.py 严格可比")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MM / "model")
    model = MiniMindForCausalLM(MiniMindConfig(
        hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
        use_moe=False))
    model.load_state_dict(torch.load(args.weight, map_location=DEVICE), strict=True)
    model = model.half().eval().to(DEVICE)
    label = args.label or Path(args.weight).stem

    rows = [json.loads(ln) for ln in open(args.qa, encoding="utf-8")]
    partner = make_partners(len(rows), args.seed)
    print(f"=== {label} | mode={args.mode} | 131 题 | max_new {args.max_new} "
          f"| rep {args.rep_penalty} "
          f"| 单轮基线在本脚本内现算", flush=True)

    out_rows = []
    with torch.no_grad():
        for i, r in enumerate(rows):
            # ① 单轮基线
            p1 = render(tok, [{"role": "user", "content": r["q"]}])
            t1 = gen(model, tok, p1, args.max_new, args.rep_penalty)
            _, single_real, _ = judge(t1, r["a"], r["q"])
            # ② 多轮:带一段历史
            j = partner[i]
            h = rows[j]
            q2 = RECALL_Q if args.mode == "recall" else r["q"]
            tgt_ans = h["a"] if args.mode == "recall" else r["a"]
            tgt_q = RECALL_Q if args.mode == "recall" else r["q"]
            p2 = render(tok, [{"role": "user", "content": h["q"]},
                              {"role": "assistant", "content": h["a"][0]},
                              {"role": "user", "content": q2}])
            t2 = gen(model, tok, p2, args.max_new, args.rep_penalty)
            _, multi_real, _ = judge(t2, tgt_ans, tgt_q)
            out_rows.append({"q": r["q"], "a": r["a"], "bin": r.get("bin"),
                             "hist_q": h["q"],
                             "single_real": single_real, "multi_real": multi_real,
                             "single_out": t1[:200], "multi_out": t2[:200]})
            if (i + 1) % 40 == 0:
                print(f"  …{i+1}/{len(rows)}", flush=True)

    n = len(out_rows)
    sg = sum(x["single_real"] for x in out_rows)
    mt = sum(x["multi_real"] for x in out_rows)
    if args.mode == "recall":
        print(f"\n--- 历史回忆 {mt}/{n} = {mt/n:.1%} | 同题单轮 {sg}/{n}")
    else:
        kept = [x for x in out_rows if x["single_real"]]
        lost = [x for x in kept if not x["multi_real"]]
        print(f"\n--- 单轮 {sg}/{n} | **多轮 {mt}/{n}** | "
              f"污染率 {len(lost)}/{len(kept)} = {len(lost)/max(len(kept),1):.1%}")

    if args.json_out:
        p = Path(args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"label": label, "mode": args.mode, "n": n,
                       "single_ok": sg, "multi_ok": mt, "rows": out_rows},
                      f, ensure_ascii=False, indent=1)
        print(f"JSON → {p}")


if __name__ == "__main__":
    main()
