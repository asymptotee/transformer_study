"""eval_pref10.py —— 10.1 行为评估:SFT vs DPO 在 held-out 规则指标上的差异

qa_test(1000 条)从未进过偏好数据(make_pref 只用 qa_train)→ 是干净的
held-out。对每个 ckpt(SFT / DPO)各采样 200 题 × 1 回答(温度 0.9),
用与造数据相同的三条规则打分,比三件事:
  · 平均规则分(0~3):DPO 应把分布推向"实质、不绕、不重复"
  · 平均长度:偏好方向包含"≥8 token",DPO 后空答应减少
  · 空答率(规则分 0 的占比)
生成耗时提示:200×(prompt+~30 token)单序列 ≈ 1.5 分钟/模型。

用法(Spark):
  ~/llm_study/.venv/bin/python eval_pref10.py
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
STAGE8 = HERE.parent / "stage8_knowledge_sft"
STAGE9 = HERE.parent / "stage9_modern_gpt"
STAGE7 = HERE.parent / "stage7_gpu_scale"
sys.path.insert(0, str(HERE.parent / "stage4_scaling_bpe"))
sys.path.insert(0, str(STAGE9))
from bpe import BPETokenizer, EOS_ID                     # noqa: E402
from generate_cached import generate_cached_ids          # noqa: E402
from make_pref import rule_score                         # noqa: E402
from model_modern import GPT, GPTConfig                  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load(ckpt_path):
    ckpt = torch.load(ckpt_path)
    cfg = GPTConfig(**ckpt["config"])
    m = GPT(cfg).to(DEVICE)
    m.load_state_dict(ckpt["model"])
    m.eval()
    return m, ckpt_path


def evaluate(model, tok, prompts):
    lens, scores, zeros = [], [], []
    for p in prompts:
        ids = generate_cached_ids(model, tok, p, 64, temperature=0.9,
                                  top_k=0, top_p=0.9, rep_penalty=1.1,
                                  eos_id=EOS_ID)
        text = tok.decode(ids).strip()
        lens.append(len(ids))
        s = rule_score(tok, p, text)
        scores.append(s)
        zeros.append(1 if s == 0 else 0)
    n = len(prompts)
    return (sum(scores) / n, sum(lens) / n, sum(zeros) / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpts", nargs="+",
                    default=[str(HERE / "ckpt_10_1_sft.pt"),
                             str(HERE / "ckpt_10_1_dpo.pt")])
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    global DEVICE
    if args.device != "auto":
        DEVICE = args.device
    random.seed(args.seed); torch.manual_seed(args.seed)

    tok = BPETokenizer.load(STAGE7 / "cache" / "bpe.json")
    prompts = []
    with open(STAGE8 / "qa_test.jsonl", encoding="utf-8") as f:
        for ln in f:
            prompts.append(json.loads(ln)["prompt"])
    random.shuffle(prompts)
    prompts = prompts[:args.n]

    for ckpt in args.ckpts:
        model, _ = load(ckpt)
        mean_score, mean_len, zero_rate = evaluate(model, tok, prompts)
        print(f"{Path(ckpt).name}: 平均规则分 {mean_score:.2f}/3 | "
              f"平均长度 {mean_len:.1f} token | 空答率 {zero_rate:.0%}")


if __name__ == "__main__":
    main()
