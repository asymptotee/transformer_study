"""train_sft10.py —— 10.1 第 1 步:在现代基座 94_full 上做 SFT(DPO 的起点)

DPO 需要一个"已经会应答"的 policy(SFT),否则 chosen/rejected 无从谈起。
本脚本 = stage8 train_sft7.py 的机制(stage8 的 QA 数据 + loss 掩码 + 全量
微调),唯一区别:基座换成 stage9 的现代 GPT(ckpt_94_full.pt,model_modern),
数据直接复用 stage8 的 qa_train/qa_test.jsonl(同一套 60k 补全式问答)。

产出 ckpt_10_1_sft.pt:既是 policy 起点,也是 DPO 的冻结参考模型(ref)。
阶段 8 的对照数字(旧架构 96.6M 基座):全量 SFT test loss 3.87。

用法(Spark):
  ~/llm_study/.venv/bin/python train_sft10.py
"""

import argparse
import json
import multiprocessing as mp
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
STAGE9 = HERE.parent / "stage9_modern_gpt"
STAGE8 = HERE.parent / "stage8_knowledge_sft"
STAGE7 = HERE.parent / "stage7_gpu_scale"
sys.path.insert(0, str(HERE.parent / "stage4_scaling_bpe"))
sys.path.insert(0, str(STAGE9))
from bpe import BPETokenizer, EOS_ID, PAD_ID          # noqa: E402
from model_modern import GPT, GPTConfig               # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IGNORE = -100   # cross_entropy 忽略标签(问题部分与 padding)


def load_qa(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f]


_TOK = None


def _init(tok):
    global _TOK
    _TOK = tok


def _enc_many(texts):
    return [_TOK.encode(t) for t in texts]


def build_examples(qa_list, tok, max_len=128, workers=10):
    """同 stage8:full = prompt + answer + EOS;y 里问题部分标 IGNORE。"""
    ctx = mp.get_context("fork")
    prompts = [ex["prompt"] for ex in qa_list]
    answers = [ex["answer"] for ex in qa_list]

    def chunks(seq, n=500):
        return [seq[i:i + n] for i in range(0, len(seq), n)]

    with ctx.Pool(processes=workers, initializer=_init, initargs=(tok,)) as pool:
        p_enc = pool.map(_enc_many, chunks(prompts))
        a_enc = pool.map(_enc_many, chunks(answers))
    p_ids_all = [i for c in p_enc for i in c]
    a_ids_all = [i for c in a_enc for i in c]

    out = []
    for p_ids, a_ids in zip(p_ids_all, a_ids_all):
        full = p_ids + a_ids + [EOS_ID]
        if len(full) > max_len or len(full) < 4:
            continue
        x = torch.tensor(full[:-1], dtype=torch.long)
        y = torch.tensor(full[1:], dtype=torch.long)
        y[:len(p_ids) - 1] = IGNORE
        out.append((x, y))
    return out


def collate(batch):
    max_len = max(len(x) for x, _ in batch)
    xs, ys = [], []
    for x, y in batch:
        pad = max_len - len(x)
        xs.append(torch.cat([x, torch.full((pad,), PAD_ID, dtype=torch.long)]))
        ys.append(torch.cat([y, torch.full((pad,), IGNORE, dtype=torch.long)]))
    return torch.stack(xs).to(DEVICE), torch.stack(ys).to(DEVICE)


@torch.no_grad()
def eval_loss(model, examples, vocab, batch_size=32):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(examples), batch_size):
        x, y = collate(examples[i:i + batch_size])
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1),
                               ignore_index=IGNORE, reduction="sum")
        tot += loss.item()
        n += (y != IGNORE).sum().item()
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", default=str(STAGE9 / "ckpt_94_full.pt"))
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=300)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ckpt-out", default=str(HERE / "ckpt_10_1_sft.pt"))
    args = ap.parse_args()

    global DEVICE
    if args.device != "auto":
        DEVICE = args.device
    torch.manual_seed(42); random.seed(42)
    print(f"设备: {DEVICE}", flush=True)

    tok = BPETokenizer.load(STAGE7 / "cache" / "bpe.json")
    train_ex = build_examples(load_qa(STAGE8 / "qa_train.jsonl"), tok,
                              workers=args.workers)
    test_ex = build_examples(load_qa(STAGE8 / "qa_test.jsonl"), tok,
                             workers=args.workers)
    print(f"样本: 训练 {len(train_ex)} | 测试 {len(test_ex)}", flush=True)

    ckpt = torch.load(args.base_ckpt)
    cfg = GPTConfig(**ckpt["config"])     # 94_full:rms/silu/rope/GQA6kv
    model = GPT(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    for p in model.parameters():
        p.requires_grad = True
    print(f"基座: {sum(p.numel() for p in model.parameters()):,} 参数 | "
          f"norm={cfg.norm} ff={cfg.ff} pos={cfg.pos} kv={cfg.n_kv_heads}",
          flush=True)
    print(f"微调前 test loss: {eval_loss(model, test_ex, len(tok)):.4f}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for step in range(1, args.steps + 1):
        model.train()
        batch = random.sample(train_ex, args.batch_size)
        x, y = collate(batch)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, len(tok)), y.reshape(-1),
                               ignore_index=IGNORE)
        opt.zero_grad(); loss.backward(); opt.step()
        if step == 1 or step % args.eval_every == 0:
            tl = eval_loss(model, test_ex, len(tok))
            print(f"  step {step:5d}/{args.steps} | train {loss.item():.4f} "
                  f"| test {tl:.4f}", flush=True)

    torch.save({"model": model.state_dict(), "config": vars(cfg),
                "step": args.steps, "sft_of": str(args.base_ckpt)}, args.ckpt_out)
    print(f"\n已保存 -> {args.ckpt_out}", flush=True)


if __name__ == "__main__":
    main()
