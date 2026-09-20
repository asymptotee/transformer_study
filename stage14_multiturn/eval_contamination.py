"""eval_contamination.py —— 14.3:多轮污染测试(复用 131 题,不另写题库)

问题:多轮 SFT 之后,模型在**带着一段历史**作答时,还答不答得对?

设计(关键是不用手写探针题,直接拿现成的 131 题构造):
  对每道题 B,拼一个两轮对话

      <|im_start|>user\\n{Q_A}<|im_end|>
      <|im_start|>assistant\\n{金标答案A}<|im_end|>
      <|im_start|>user\\n{Q_B}<|im_end|>
      <|im_start|>assistant\\n            ← 看这里生成什么

  然后判 B 答对没。**对照就是现成的**:同一道 B,同一个模型,单轮问它时的
  结果已经在 results_v2/*.json 里。于是得到

      污染率 = (单轮对 且 多轮错) / 单轮对

  判分直接复用 judge.py(含回声截断那两条修正),生成参数与 eval_v2 一致
  (贪心),所以两个数字严格可比。

为什么不用随机历史:配对用固定 seed 的随机环(第 i 题拿第 j 题当历史,
j 是 n 个题目的一个置换、无自环),保证可复现;而"同类别配对"是更狠的
变体(主题粘连压力最大),留给需要时再跑。

用法(Spark):
  ~/llm_study/.venv/bin/python eval_contamination.py \
      --ckpt ../stage11_datascale/ckpt_14_mt.pt \
      --bpe ../stage11_datascale/cache_mm10g/bpe.json \
      --baseline ../stage13_eval/results_v2/14_mt.json --label 14_mt \
      --json-out results/contam_14_mt.json
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
MM = Path.home() / "llm_study" / "minimind"
sys.path.insert(0, str(MM))
from transformers import AutoTokenizer                     # noqa: E402
sys.path.insert(0, str(REPO / "stage13_eval"))
sys.path.insert(0, str(REPO / "stage4_scaling_bpe"))
sys.path.insert(0, str(REPO / "stage9_modern_gpt"))
from eval_v2 import gen, DEVICE                            # noqa: E402
from judge import judge                                    # noqa: E402
from bpe import BPETokenizer                               # noqa: E402
from model_modern import GPT, GPTConfig                    # noqa: E402

IM_END = "<|im_end|>"


RECALL_Q = "我上一个问题的答案是什么？请直接回答。"
# ⚠️ 必须整段剥 think:minimind 的 chat_template 会给**每个** assistant 轮自动插入
# `<think>\n\n</think>\n\n`,而训练数据是 --strip-think 整段剥掉的。不剥就是分布外
# 输入 —— 第一版漏了这一步,测出来的数字全部作废(模型在猜没见过的格式)。
THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def make_mt_prompt(tok_mm, q_hist, a_hist, q_now):
    """两轮提示词:**用 minimind 的 chat_template 渲染 + 整段剥 think**,
    与训练数据(render_mt_chat.py --strip-think)走完全相同的变换。

    不手写模板:渲染出来的文本里 assistant 段前后有额外换行(目标段以 `\\n\\n`
    开头),手写差一个换行同样是分布外。
    """
    msgs = [{"role": "user", "content": q_hist},
            {"role": "assistant", "content": a_hist},
            {"role": "user", "content": q_now}]
    text = tok_mm.apply_chat_template(msgs, tokenize=False,
                                      add_generation_prompt=True)
    return THINK_RE.sub("", text)


def recall_questions(rows):
    """历史回忆测试用的"第二问"——固定问上一轮的答案。

    为什么需要它(见 14.3 报告):污染测试里 14_mt 的多轮分 70 ≈ 它的单轮分 74,
    所以"读懂历史"与"无视历史、照答最后一问"在这个指标上**完全同分**,分不开。
    本测试的第二问答案**逐字写在历史里**,不读历史就必然答不出 —— 它把两者分开。
    """
    return RECALL_Q


def make_partners(n, seed=42):
    """n 个题目的无自环随机置换:第 i 题用 partner[i] 题当历史。"""
    rng = random.Random(seed)
    idx = list(range(n))
    while True:
        rng.shuffle(idx)
        if all(idx[i] != i for i in range(n)):
            return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bpe", required=True)
    ap.add_argument("--qa", default=str(REPO / "stage13_eval" / "qa_v2_freq.jsonl"))
    ap.add_argument("--baseline", required=True,
                    help="同模型的单轮结果 json(取 chat 通道做对照)")
    ap.add_argument("--label", default="")
    ap.add_argument("--mode", choices=["contam", "recall"], default="contam",
                    help="contam=答第二问(测污染);recall=问上一轮答案(测是否读历史)")
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    tok = BPETokenizer.load(args.bpe)
    ckpt = torch.load(args.ckpt)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    label = args.label or Path(args.ckpt).stem
    print(f"=== {label} | 多轮污染测试 | 单轮基线 {Path(args.baseline).name}",
          flush=True)

    rows = [json.loads(ln) for ln in open(args.qa, encoding="utf-8")]
    base = json.load(open(args.baseline, encoding="utf-8"))["formats"]["chat"]["rows"]
    assert [r["q"] for r in rows] == [r["q"] for r in base], "题库顺序不一致"
    partner = make_partners(len(rows), args.seed)
    tok_mm = AutoTokenizer.from_pretrained(MM / "model")     # 只为渲染模板

    out_rows = []
    for i, r in enumerate(rows):
        j = partner[i]
        h = rows[j]
        if args.mode == "recall":
            # 第二问固定为"上一轮答案是什么";答案逐字在历史里
            prompt = make_mt_prompt(tok_mm, h["q"], h["a"][0], RECALL_Q)
            raw, real, susp = judge(out := gen(model, tok, prompt, args.max_new),
                                    h["a"], RECALL_Q)
            out_rows.append({"q": h["q"], "a": h["a"], "bin": h.get("bin"),
                             "cat": h.get("cat"), "hist_a": h["a"][0],
                             "out": out[:300], "raw": raw, "real": real,
                             "suspect": susp, "single_real": base[j]["real"]})
        else:
            prompt = make_mt_prompt(tok_mm, h["q"], h["a"][0], r["q"])
            raw, real, susp = judge(out := gen(model, tok, prompt, args.max_new),
                                    r["a"], r["q"])
            out_rows.append({"q": r["q"], "a": r["a"], "cat": r.get("cat"),
                             "bin": r.get("bin"), "hist_q": h["q"],
                             "out": out[:300], "raw": raw, "real": real,
                             "suspect": susp, "single_real": base[i]["real"]})
        if (i + 1) % 40 == 0:
            print(f"  …{i+1}/{len(rows)}", flush=True)

    n = len(out_rows)
    mt_ok = sum(x["real"] for x in out_rows)
    sg_ok = sum(x["single_real"] for x in out_rows)

    if args.mode == "recall":
        # 历史里的答案单轮问原题时本来就答得对 → 直接给"不读历史"的下限参照
        print(f"\n--- 历史回忆 {mt_ok}/{n}(={mt_ok/n:.0%})")
        print(f"--- 同题的单轮命中 {sg_ok}/{n}(读不到历史时,这就是天花板)")
        by_bin = {}
        for x in out_rows:
            by_bin.setdefault(x["bin"], [0, 0])
            by_bin[x["bin"]][1] += 1
            by_bin[x["bin"]][0] += x["real"]
        print("--- 分档回忆率:", {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_bin.items())})
        rep = {"label": label, "ckpt": str(args.ckpt), "mode": "recall", "n": n,
               "recall_ok": mt_ok, "single_ok": sg_ok, "by_bin": by_bin,
               "rows": out_rows}
    else:
        kept = [x for x in out_rows if x["single_real"]]      # 单轮就会的题
        lost = [x for x in kept if not x["real"]]
        gained = [x for x in out_rows if not x["single_real"] and x["real"]]
        contam = len(lost) / max(len(kept), 1)
        print(f"\n--- 单轮 {sg_ok}/{n} | 多轮 {mt_ok}/{n}")
        print(f"--- 污染率 {contam:.1%}(单轮会的 {len(kept)} 题里,多轮答错 {len(lost)} 题)")
        print(f"--- 反向:单轮不会但多轮答对 {len(gained)} 题")
        by_bin = {}
        for x in kept:
            b = x["bin"]
            by_bin.setdefault(b, [0, 0])
            by_bin[b][1] += 1
            by_bin[b][0] += (not x["real"])
        print("--- 分档污染率:", {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_bin.items())})
        rep = {"label": label, "ckpt": str(args.ckpt), "mode": "contam", "n": n,
               "single_ok": sg_ok, "multi_ok": mt_ok,
               "contamination": contam, "gained": len(gained),
               "by_bin": by_bin, "rows": out_rows}
    if args.json_out:
        p = Path(args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=1)
        print(f"JSON → {p}")


if __name__ == "__main__":
    main()
