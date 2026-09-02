"""train_sft7.py —— 阶段八：在 100M 百科基座上做 SFT（知识问答）

与阶段六 train_sft.py 同机制（loss 掩码、全量/LoRA 两轴），只换了两样东西：
  基座：stage7 的 100M 维基模型（ckpt_large.pt + cache/bpe.json）
  数据：stage8 的 qa_train.jsonl（维基补全式问答）

要验证的命题（承接阶段七实验一）：
  预训练已注入知识（基座"知道"北京=首都），SFT 把"回答姿势"接通——
  高频事实答得对（知识来自预训练），低频答不出（知识边界仍在）。

运行（DGX Spark）：
  python -u train_sft7.py            # 全量 SFT
  python -u train_sft7.py --lora     # LoRA SFT
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
STAGE7 = HERE.parent / "stage7_gpu_scale"
sys.path.insert(0, str(HERE.parent / "stage3_gpt"))
sys.path.insert(0, str(HERE.parent / "stage4_scaling_bpe"))
from bpe import BPETokenizer, EOS_ID, PAD_ID
from model_gpt import GPT, GPTConfig
sys.path.insert(0, str(HERE.parent / "stage6_sft"))
from lora import inject_lora

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IGNORE = -100   # cross_entropy 的忽略标签（用于掩码问题部分和 padding）


def load_qa(name):
    with open(HERE / name, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f]


_TOK = None


def _init(tok):
    global _TOK
    _TOK = tok


def _enc_many(texts):
    return [_TOK.encode(t) for t in texts]


def build_examples(qa_list, tok, max_len=128, workers=10):
    """把 (prompt, answer) 变成 (x, y)，并对问题部分做 loss 掩码。

    编码是纯 Python BPE（慢），用 fork 多进程批量编码。
    注意：必须在加载 CUDA 模型之前调用（fork 子进程不能带 GPU context）。
    """
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
        y[:len(p_ids) - 1] = IGNORE          # 掩码：预测问题 token 的位置不算 loss
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


@torch.no_grad()
def answer(model, tok, prompt, max_new=40):
    """给一个'问：…答：'提示，贪心生成答案直到 EOS。"""
    model.eval()
    ids = tok.encode(prompt)
    for _ in range(max_new):
        ctx = torch.tensor([ids[-model.cfg.max_len:]], dtype=torch.long,
                           device=DEVICE)
        nxt = int(model(ctx)[0, -1].argmax().item())
        if nxt == EOS_ID:
            break
        ids.append(nxt)
    gen = ids[len(tok.encode(prompt)):]
    return tok.decode(gen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", action="store_true", help="用 LoRA 而非全量微调")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=300)
    ap.add_argument("--device", default="auto", help="cpu/cuda；默认 auto")
    ap.add_argument("--workers", type=int, default=10, help="编码用进程数")
    ap.add_argument("--ckpt-out", default=None)
    args = ap.parse_args()

    global DEVICE
    if args.device != "auto":
        DEVICE = args.device
    print(f"设备: {DEVICE}")

    torch.manual_seed(42); random.seed(42)

    # 词表（先加载，build_examples 编码要用；编码必须在 CUDA 之前做）
    tok = BPETokenizer.load(STAGE7 / "cache" / "bpe.json")

    # 数据（fork 多进程编码，此时还没有 GPU context）
    train_ex = build_examples(load_qa("qa_train.jsonl"), tok, workers=args.workers)
    test_ex = build_examples(load_qa("qa_test.jsonl"), tok, workers=args.workers)
    print(f"样本: 训练 {len(train_ex)} | 测试 {len(test_ex)}")

    # 加载 stage7 基座
    ckpt = torch.load(STAGE7 / "ckpt_large.pt")
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    total = sum(p.numel() for p in model.parameters())

    mode = "LoRA SFT" if args.lora else "全量 SFT"
    if args.lora:
        trainable = inject_lora(model, r=args.lora_r)
        print(f"[{mode}] 总参数 {total:,} | 可训练 {trainable:,} "
              f"({trainable/total*100:.2f}%)  r={args.lora_r}")
    else:
        for p in model.parameters():
            p.requires_grad = True
        print(f"[{mode}] 总参数 {total:,} | 可训练 {total:,} (100%)")

    print(f"微调前测试 loss: {eval_loss(model, test_ex, len(tok)):.4f}")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)

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
                  f"| test {tl:.4f}")

    out = args.ckpt_out or (HERE / ("ckpt_sft7_lora.pt" if args.lora else "ckpt_sft7_full.pt"))
    torch.save({"model": model.state_dict(), "config": vars(cfg),
                "lora": args.lora, "lora_r": args.lora_r}, out)
    print(f"\n已保存 -> {out}")

    # 生成样例（测试集前几条）
    print("\n生成样例（贪心，直到 EOS）:")
    for ex in load_qa("qa_test.jsonl")[:5]:
        gen = answer(model, tok, ex["prompt"])
        print(f"  {ex['prompt'].splitlines()[0]}")
        print(f"    期望: {ex['answer']}")
        print(f"    生成: {gen}")


if __name__ == "__main__":
    main()
