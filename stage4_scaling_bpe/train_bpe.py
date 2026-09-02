"""train_bpe.py —— 用 BPE 分词训练 GPT，并和字符级做公平对比

公平对比的关键：不同分词器的 loss **不能直接比**（一个 token 含义不同）。
要换算成"每字符 loss"（nats/char）才能比：

    nats/char = val_loss(每 token) ÷ (每 token 含几个字符)

它衡量"平均每个字符让模型多惊讶"，与分词方式无关——越低说明模型压缩得越好。

对比三件事：
  1. 序列缩短：BPE 把 N 个字符压成多少 token（压缩率）
  2. nats/char：BPE vs 字符级，同样参数/步数下谁压缩得更好
  3. 生成样例：BPE 按"词块"生成，观感是否更连贯

运行（约 10 分钟，建议等缩放实验跑完再开，避免抢 CPU）：
  python -u train_bpe.py
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "stage3_gpt"))      # 复用 GPT
sys.path.insert(0, str(HERE.parent / "stage1_basics"))   # 复用字符分词器
from data import CharTokenizer
from model_gpt import GPT, GPTConfig

from bpe import BPETokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CORPUS = HERE.parent / "stage3_gpt" / "luxun_clean.txt"
BLOCK, BATCH, LR = 64, 32, 3e-4
# 两个模型用完全相同的配置和训练预算，唯一变量是分词方式
KW = dict(d_model=128, n_layers=4, n_heads=4, d_ff=256)


def get_batch(data, vocab):
    ix = torch.randint(len(data) - BLOCK - 1, (BATCH,))
    x = torch.stack([data[i:i + BLOCK] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + BLOCK] for i in ix])
    return x, y


def train_model(ids, vocab, steps, label):
    torch.manual_seed(42)
    cfg = GPTConfig(vocab_size=vocab, pad_id=0, max_len=max(BLOCK, 256), **KW)
    model = GPT(cfg).to(DEVICE)
    n = sum(p.numel() for p in model.parameters())
    print(f"\n=== {label} ===  词表 {vocab} | 参数 {n:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    n_val = int(len(ids) * 0.1)
    train_ids, val_ids = ids[:-n_val].to(DEVICE), ids[-n_val:].to(DEVICE)
    best = float("inf")
    for step in range(1, steps + 1):
        model.train()
        x, y = get_batch(train_ids, vocab)
        loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1), ignore_index=0)
        opt.zero_grad(); loss.backward(); opt.step()
        if step == 1 or step % (steps // 5) == 0:
            model.eval()
            with torch.no_grad():
                xv, yv = get_batch(val_ids, vocab)
                vl = F.cross_entropy(model(xv).reshape(-1, vocab), yv.reshape(-1), ignore_index=0)
            best = min(best, vl.item())
            print(f"  step {step:5d} | train {loss.item():.3f} | val {vl.item():.3f}")
    return model, best, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merges", type=int, default=600)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--device", default="auto", help="cpu/cuda；默认 auto（有 GPU 用 GPU）")
    args = ap.parse_args()

    global DEVICE
    if args.device != "auto":
        DEVICE = args.device
    print(f"设备: {DEVICE}")

    text = CORPUS.read_text(encoding="utf-8")

    # ---------- BPE ----------
    print(f"训练 BPE（{args.merges} 次合并）...")
    bpe = BPETokenizer.train(text, num_merges=args.merges, verbose=False)
    bpe.save(HERE / "bpe.json")
    print("编码全文（一次性）...")
    bpe_ids = torch.tensor(bpe.encode(text), dtype=torch.long)
    ratio = len(text) / len(bpe_ids)        # 每 token 含几个字符
    print(f"字符 {len(text):,} → BPE token {len(bpe_ids):,}  (压缩率 {ratio:.2f} 字符/token)")

    bpe_model, bpe_val, bpe_n = train_model(bpe_ids, len(bpe), args.steps, "BPE 模型")
    bpe_npc = bpe_val / ratio               # 换算成每字符 loss

    # ---------- 字符级基线（同配置同步数，保证公平）----------
    ctok = CharTokenizer([text])
    char_ids = torch.tensor(ctok.encode(text), dtype=torch.long)
    char_model, char_val, char_n = train_model(char_ids, len(ctok), args.steps, "字符级模型")
    char_npc = char_val                     # 字符级 1 token = 1 字符

    # ---------- 对比 ----------
    print("\n" + "=" * 62)
    print("公平对比（nats/char 越低越好，与分词方式无关）")
    print("=" * 62)
    print(f"{'':10}{'参数量':>12}{'val_loss/token':>16}{'nats/char':>12}")
    print(f"{'字符级':10}{char_n:>12,}{char_val:>16.4f}{char_npc:>12.4f}")
    print(f"{'BPE':10}{bpe_n:>12,}{bpe_val:>16.4f}{bpe_npc:>12.4f}")
    print(f"\nBPE 把序列压到原来的 {1/ratio*100:.0f}%：同样 64 的窗口，"
          f"BPE 看到约 {int(64*ratio)} 个字符，字符级只看到 64 个。")

    # ---------- 生成对比 ----------
    prompt = "我冒了严寒"
    print(f"\n生成对比（prompt='{prompt}', temperature=0.7）:")
    for label, model, tok in [("字符级", char_model, ctok), ("BPE", bpe_model, bpe)]:
        ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=DEVICE)
        out = model.generate(ids, 40, temperature=0.7, top_k=15)
        print(f"  [{label}] {tok.decode(out[0].tolist())}")


if __name__ == "__main__":
    main()
