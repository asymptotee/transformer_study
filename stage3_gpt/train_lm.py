"""train_lm.py —— next-token 语言模型训练（GPT 范式）

与阶段一 seq2seq 训练的关键区别（本脚本的核心教学点）：

  阶段一（seq2seq）：
    数据是 (src, tgt) 对；decoder 吃 src 编码 + tgt 右移；只算 tgt 位置的损失。
    —— "把输入映射成输出"

  阶段三（语言模型）：
    没有 src/tgt 之分。**输入和目标就是同一段文本错一位**：
        输入  x = text[i : i+T]
        目标  y = text[i+1 : i+T+1]
    对每个位置都算"预测下一个 token"的交叉熵。
    —— "建模序列的概率分布"（LLM 的本质）

  数据组织（nanoGPT 做法）：把整个语料拼成一条长序列，每步随机切 batch 个
  长度为 block_size 的窗口。比按行喂更省、上下文更连贯。

运行：
  python train_lm.py
  python train_lm.py --data 你的真实文本.txt
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent

import sys
sys.path.insert(0, str(HERE.parent / "stage1_basics"))
from data import CharTokenizer, PAD_ID          # 复用阶段一字符分词器
from model_gpt import GPT, GPTConfig


def main():
    ap = argparse.ArgumentParser(description="训练字符级 GPT 语言模型")
    ap.add_argument("--data", default=str(HERE / "corpus.txt"))
    ap.add_argument("--block-size", type=int, default=64, help="每个训练窗口的长度")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--ckpt", default=str(HERE / "ckpt_gpt.pt"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    # ---------- 数据：整段文本 -> token 流 ----------
    text = Path(args.data).read_text(encoding="utf-8")
    tok = CharTokenizer([text])                 # 词表含文本里所有字符（包括换行符）
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    print(f"语料: {len(text):,} 字符 | 词表 {len(tok)} | token 数 {len(ids):,} | 设备 {device}")

    # 90/10 切 train/val（按文本顺序，val 取末尾）
    n_val = int(len(ids) * 0.1)
    train_ids, val_ids = ids[:-n_val], ids[-n_val:]

    cfg = GPTConfig(vocab_size=len(tok), pad_id=PAD_ID,
                    max_len=max(args.block_size, 256))
    model = GPT(cfg).to(device)
    print(f"参数: {sum(p.numel() for p in model.parameters()):,}  "
          f"(d_model={cfg.d_model}, {cfg.n_layers} 层, {cfg.n_heads} 头)")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    B, T = args.batch_size, args.block_size

    def get_batch(split):
        """随机取 B 个长度 T 的窗口，返回 (输入, 目标=输入右移一位)。"""
        data = train_ids if split == "train" else val_ids
        ix = torch.randint(len(data) - T - 1, (B,))
        x = torch.stack([data[i:i + T] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + 1 + T] for i in ix]).to(device)
        return x, y

    # ---------- 训练循环 ----------
    for step in range(1, args.steps + 1):
        model.train()
        x, y = get_batch("train")
        logits = model(x)                       # (B, T, vocab)
        loss = F.cross_entropy(logits.reshape(-1, len(tok)), y.reshape(-1),
                               ignore_index=PAD_ID)
        opt.zero_grad(); loss.backward(); opt.step()

        if step == 1 or step % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                xv, yv = get_batch("val")
                vl = F.cross_entropy(model(xv).reshape(-1, len(tok)),
                                     yv.reshape(-1), ignore_index=PAD_ID)
            print(f"step {step:5d}/{args.steps} | train_loss {loss.item():.4f} "
                  f"| val_loss {vl.item():.4f}")

    torch.save({"model": model.state_dict(), "config": vars(cfg), "vocab": tok.itos},
               args.ckpt)
    print(f"\n已保存 checkpoint -> {args.ckpt}")
    print("试用: python generate_text.py --prompt '山' --temperature 0.8")


if __name__ == "__main__":
    main()
