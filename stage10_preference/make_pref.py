"""make_pref.py —— 10.1 第 2 步:自造偏好对(chosen / rejected)

机制学习的自造数据(ROADMAP:数据质量不是重点,机制是重点):
  1. 对每条 prompt(qa_train 的补全式问答),用 SFT policy 采样 K 个回答
     (温度 0.9 多样性)
  2. 规则裁判打分——三条可解释规则,每条回答给 0/1:
       R1 实质长度:回答 ≥8 个 BPE token(排除空答/只吐一个字的)
       R2 不糊弄:回答不几乎原样复读 prompt 尾部(新 token 占比 > 50%)
       R3 不重复:唯一 token 占比 > 0.6(排除复读机)
     总分 = R1+R2+R3(0~3)
  3. chosen = 最高分样本,rejected = 最低分样本(分差必须 >0,否则丢弃)
     → 规则的偏好方向是"答得实质、不绕圈、不重复"——和真实 RLHF 里
       "有用、不糊弄"的奖励方向同构,只是更可解释

输出 pref_train.jsonl:{"prompt", "chosen", "rejected"}(纯文本,DPO 时再编码)
+ 抽样几条供人工看。

用法(Spark,SFT 完成后):
  ~/llm_study/.venv/bin/python make_pref.py --n-prompts 2000 --k 6
耗时提示:单序列 KV cache 解码 ~180 tok/s;2000×6 ≈ 30 万新 token ≈ 25 分钟。
机制演示不需要海量对(DPO 数据质量 > 数量),1000 prompt 已够看信号。
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
sys.path.insert(0, str(HERE.parent / "stage5_capstone"))
sys.path.insert(0, str(STAGE9))
from bpe import BPETokenizer, EOS_ID                     # noqa: E402
from generate_cached import generate_cached_ids          # noqa: E402
from model_modern import GPT, GPTConfig                  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def rule_score(tok, prompt, text):
    """三条规则,返回 0~3。text 是模型生成的新 token 的解码。"""
    s = 0
    if len(tok.encode(text)) >= 8:                       # R1 实质长度
        s += 1
    p_tail = tok.encode(prompt)[-6:]
    if len(text) and len(set(tok.encode(text)) & set(p_tail)) / max(1, len(p_tail)) < 0.5:
        s += 1                                           # R2 不复读 prompt 尾
    ids = tok.encode(text)
    if ids and len(set(ids)) / len(ids) > 0.6:           # R3 不原地复读
        s += 1
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-ckpt", default=str(HERE / "ckpt_10_sft.pt"))
    ap.add_argument("--n-prompts", type=int, default=2000)
    ap.add_argument("--k", type=int, default=6, help="每题采样回答数")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(HERE / "pref_train.jsonl"))
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
    print(f"policy: {sum(p.numel() for p in model.parameters()):,} 参数", flush=True)

    prompts = []
    with open(STAGE8 / "qa_train.jsonl", encoding="utf-8") as f:
        for ln in f:
            prompts.append(json.loads(ln)["prompt"])
    random.shuffle(prompts)
    prompts = prompts[:args.n_prompts]

    made, skipped = 0, 0
    with open(args.out, "w", encoding="utf-8") as out:
        for i, prompt in enumerate(prompts):
            samples = []
            for _ in range(args.k):
                ids = generate_cached_ids(model, tok, prompt, args.max_new,
                                          temperature=args.temperature,
                                          top_k=0, top_p=0.9, rep_penalty=1.1,
                                          eos_id=EOS_ID)
                samples.append(tok.decode(ids).strip())
            scored = [(rule_score(tok, prompt, t), t) for t in samples]
            scored.sort(key=lambda x: x[0])
            best, worst = scored[-1], scored[0]
            if best[0] - worst[0] > 0 and best[1] and worst[1]:
                out.write(json.dumps({"prompt": prompt, "chosen": best[1],
                                      "rejected": worst[1]}, ensure_ascii=False) + "\n")
                made += 1
            else:
                skipped += 1
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(prompts)} | 成对 {made} | 弃 {skipped}",
                      flush=True)

    print(f"偏好对: {made} | 弃(无法分档): {skipped}", flush=True)
    print("样例:")
    with open(args.out, encoding="utf-8") as f:
        for j, ln in enumerate(f):
            if j >= 5:
                break
            ex = json.loads(ln)
            print(f"  P: {ex['prompt'][:34]}")
            print(f"  ✓ : {ex['chosen'][:48]!r}")
            print(f"  ✗ : {ex['rejected'][:48]!r}")


if __name__ == "__main__":
    main()
