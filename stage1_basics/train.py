"""训练脚本

用法示例：
    python train.py --task reverse --epochs 15     # 反转任务（验证架构）
    python train.py --task complete --epochs 15    # 续写任务（文本生成）
    python train.py --task reverse --smoke         # 单 batch 过拟合冒烟测试

训练方式：teacher forcing —— decoder 的输入是真实目标序列（右移一位），
标签是目标序列左移一位，所有位置并行计算交叉熵。
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import (PAD_ID, CharTokenizer, Seq2SeqDataset, collate,
                  load_corpus, make_pairs, random_lines, split_train_val)
from model import Transformer, TransformerConfig

HERE = Path(__file__).parent


def evaluate(model, loader, device):
    """在验证集上评估，返回 (平均 loss, token 准确率, 整句精确匹配率)。

    精确匹配率（exact match）是 reverse 任务最硬的指标：
    整句一字不差才算对，能到 90%+ 说明架构和训练管道完全正确。
    """
    model.eval()
    total_loss = total_tokens = correct = exact = total_seqs = 0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            logits = model(src, tgt[:, :-1])
            labels = tgt[:, 1:]
            mask = labels != PAD_ID  # 只在真实 token 上统计
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=PAD_ID,
            )
            n = mask.sum().item()
            total_loss += loss.item() * n
            total_tokens += n
            pred = logits.argmax(dim=-1)
            correct += ((pred == labels) & mask).sum().item()
            # 整句匹配：所有真实位置都预测对（pad 位置视为自动通过）
            exact += (((pred == labels) | ~mask).all(dim=1)).sum().item()
            total_seqs += src.size(0)
    return total_loss / total_tokens, correct / total_tokens, exact / total_seqs


def run_smoke(model, device, lr, steps):
    """冒烟测试：在单个 batch 上反复训练直到过拟合。

    loss 应该能降到接近 0 —— 如果降不下去，说明模型或管道有 bug。
    这是调试神经网络的第一步，先证明"学得动"，再谈泛化。
    """
    print("冒烟测试：单 batch 过拟合，loss 应降到 ~0")
    src = torch.randint(4, 100, (16, 10), device=device)
    tgt = torch.randint(4, 100, (16, 11), device=device)
    tgt[:, 0] = 1  # <bos>
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(1, steps + 1):
        model.train()
        logits = model(src, tgt[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt[:, 1:].reshape(-1),
            ignore_index=PAD_ID,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 1 or step % 50 == 0:
            print(f"  step {step:4d} | loss {loss.item():.4f}")


def main():
    ap = argparse.ArgumentParser(description="训练字符级 Transformer")
    ap.add_argument("--task", choices=["reverse", "complete"], default="reverse")
    ap.add_argument("--data", default=str(HERE / "corpus.txt"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--ckpt", default=str(HERE / "ckpt.pt"))
    ap.add_argument("--select-by", choices=["val_loss", "train_loss"], default="val_loss",
                    help="按哪个指标挑选最佳 checkpoint：会泛化的任务（reverse）用 val_loss，"
                         "靠记忆的任务（complete）用 train_loss")
    ap.add_argument("--synthetic", type=int, default=4000,
                    help="reverse 任务的随机字符串增强样本数（防背诵、逼模型学算法，见 data.random_lines）")
    ap.add_argument("--smoke", action="store_true", help="单 batch 过拟合冒烟测试")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    lines = load_corpus(args.data)
    tokenizer = CharTokenizer(lines)  # 字符级词表很小，直接用全量语料建表
    cfg = TransformerConfig(vocab_size=len(tokenizer), max_len=args.max_len, pad_id=PAD_ID)
    model = Transformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"任务: {args.task} | 设备: {device} | 词表: {len(tokenizer)} | 参数: {n_params:,}")

    if args.smoke:
        run_smoke(model, device, args.lr, steps=args.epochs if args.epochs != 15 else 300)
        return

    train_lines, val_lines = split_train_val(lines)
    if args.task == "reverse" and args.synthetic > 0:
        # 反转任务靠真实语料泛化不了（样本太少会被背下来），用随机字符串增强训练集
        chars = tokenizer.itos[4:]  # 跳过特殊 token
        train_lines = train_lines + random_lines(chars, args.synthetic, max_len=args.max_len - 2)
    train_ds = Seq2SeqDataset(make_pairs(train_lines, args.task), tokenizer, args.max_len)
    val_ds = Seq2SeqDataset(make_pairs(val_lines, args.task), tokenizer, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate)
    print(f"样本: 训练 {len(train_ds)} / 验证 {len(val_ds)}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_score = float("inf")
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_tokens = 0.0, 0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            # teacher forcing：输入 tgt 去掉最后一个 token，标签去掉第一个（<bos>）
            logits = model(src, tgt[:, :-1])
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt[:, 1:].reshape(-1),
                ignore_index=PAD_ID,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            n = (tgt[:, 1:] != PAD_ID).sum().item()
            total_loss += loss.item() * n
            total_tokens += n
        val_loss, val_acc, val_exact = evaluate(model, val_loader, device)
        print(
            f"epoch {epoch:3d}/{args.epochs} | train_loss {total_loss / total_tokens:.4f}"
            f" | val_loss {val_loss:.4f} | val_token_acc {val_acc:.3f} | val_exact {val_exact:.3f}"
        )
        # 只保存表现最好的那一轮：末期指标常会震荡，最后一轮未必最好。
        # 挑选标准取决于任务性质：reverse 要泛化（看 val_loss），complete 靠记忆（看 train_loss）
        score = val_loss if args.select_by == "val_loss" else total_loss / total_tokens
        if score < best_score:
            best_score, best_epoch = score, epoch
            torch.save(
                {"model": model.state_dict(), "config": vars(cfg),
                 "vocab": tokenizer.itos, "task": args.task},
                args.ckpt,
            )

    print(f"已保存最佳 checkpoint（epoch {best_epoch}, {args.select_by} {best_score:.4f}）-> {args.ckpt}")


if __name__ == "__main__":
    main()
