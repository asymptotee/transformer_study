"""eval_grpo10.py —— 10.2 最终对比:SFT vs DPO(同规则) vs GRPO 在 held-out 上

用 qa_test(从没进过任何训练/造对数据)的 prompt,每模型采 K 个回答,
按 rule_reward 打分(与 GRPO 在线奖励同函数)——"同一个裁判"测三个
优化器谁把分布推得最靠近奖励方向;附平均长度/重复率,防"刷分翻车"
(比如奖励没升但回答崩成复读——那正是 RL 的经典失败模式)。

用法(Spark):
  ~/llm_study/.venv/bin/python eval_grpo10.py --n 120 --k 3
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
from bpe import BPETokenizer, EOS_ID                        # noqa: E402
from generate_cached import generate_cached_ids             # noqa: E402
from grpo import rule_reward                                # noqa: E402
from model_modern import GPT, GPTConfig                     # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load(ckpt_path):
    ckpt = torch.load(ckpt_path)
    cfg = GPTConfig(**ckpt["config"])
    m = GPT(cfg).to(DEVICE)
    m.load_state_dict(ckpt["model"])
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120, help="held-out prompt 数")
    ap.add_argument("--k", type=int, default=3, help="每模型每 prompt 采样数")
    ap.add_argument("--ckpts", nargs="+",
                    default=[str(HERE / f) for f in
                             ("ckpt_10_sft.pt", "ckpt_10_dpo.pt",
                              "ckpt_10_grpo.pt", "ckpt_10_grpo_strong.pt",
                              "ckpt_10_dpo_rule.pt")])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    global DEVICE
    if args.device != "auto":
        DEVICE = args.device
    random.seed(args.seed); torch.manual_seed(args.seed)

    tok = BPETokenizer.load(STAGE7 / "cache" / "bpe.json")
    prompts = [json.loads(ln)["prompt"]
               for ln in open(STAGE8 / "qa_test.jsonl", encoding="utf-8")]
    random.shuffle(prompts)
    prompts = prompts[:args.n]

    for ckpt in args.ckpts:
        if not Path(ckpt).exists():
            print(f"{Path(ckpt).name}: (不存在,跳过)")
            continue
        model = load(ckpt)
        rs, lens, dups = [], [], []
        for p in prompts:
            pid = tok.encode(p)
            for _ in range(args.k):
                ids = generate_cached_ids(model, tok, p, 64, temperature=0.9,
                                          top_k=0, top_p=0.9, rep_penalty=1.1,
                                          eos_id=EOS_ID)
                stopped = len(ids) < 64
                rs.append(rule_reward(pid, ids, stopped))
                lens.append(len(ids))
                dups.append(0 if not ids else 1 - len(set(ids)) / len(ids))
        n = len(rs)
        print(f"{Path(ckpt).name}: 规则分 {sum(rs)/n:+.3f} | "
              f"长度 {sum(lens)/n:.1f} | 重复率 {sum(dups)/n:.3f}", flush=True)


if __name__ == "__main__":
    main()
