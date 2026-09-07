"""make_pref_rule.py —— 10.2 对照臂:用 GRPO 的同一规则做离线 DPO 对

10.1 的教训 + 10.2 的设计:同一份规则奖励,两个优化器各走一遍再比。
本脚本把 rule_reward(GRPO 在线的奖励)固化成离线偏好对:
每个 prompt 采 K 个回答,chosen = 最高分,rejected = 最低分(分差 > 0 才留)。
产物喂给 train_dpo10.py,得到 ckpt_10_dpo_rule —— 与 ckpt_10_grpo(_strong)
在 held-out 规则分上三向对比(SFT 基线 / DPO / GRPO)。

用法(Spark):
  ~/llm_study/.venv/bin/python make_pref_rule.py --n-prompts 500 --k 6
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-ckpt", default=str(HERE / "ckpt_10_sft.pt"))
    ap.add_argument("--n-prompts", type=int, default=500)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--out", default=str(HERE / "pref_rule.jsonl"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    global DEVICE
    if args.device != "auto":
        DEVICE = args.device
    torch.manual_seed(args.seed); random.seed(args.seed)

    tok = BPETokenizer.load(STAGE7 / "cache" / "bpe.json")
    ckpt = torch.load(args.sft_ckpt)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    prompts = [json.loads(ln)["prompt"]
               for ln in open(STAGE8 / "qa_train.jsonl", encoding="utf-8")]
    random.shuffle(prompts)
    prompts = prompts[:args.n_prompts]

    made, skipped = 0, 0
    with open(args.out, "w", encoding="utf-8") as out:
        for i, p in enumerate(prompts):
            pid = tok.encode(p)
            scored = []
            for _ in range(args.k):
                ids = generate_cached_ids(model, tok, p, 64, temperature=0.9,
                                          top_k=0, top_p=0.9, rep_penalty=1.1,
                                          eos_id=EOS_ID)
                stopped = len(ids) < 64
                r = rule_reward(pid, ids, stopped)
                scored.append((r, tok.decode(ids).strip()))
            scored.sort(key=lambda x: x[0])
            if scored[-1][0] - scored[0][0] > 0.05:       # 分差太小不分对
                out.write(json.dumps({"prompt": p, "chosen": scored[-1][1],
                                      "rejected": scored[0][1]},
                                     ensure_ascii=False) + "\n")
                made += 1
            else:
                skipped += 1
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(prompts)} | 成对 {made} | 弃 {skipped}",
                      flush=True)
    print(f"偏好对: {made} | 弃: {skipped}", flush=True)


if __name__ == "__main__":
    main()
