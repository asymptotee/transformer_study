"""scaling_experiment.py —— 缩放实验：模型越大，loss 越低，越连贯

控制变量：**同一份鲁迅语料、同样的训练步数，只变模型大小。**
观察：val_loss 随参数量下降（scaling law 的缩影），生成质量随之改善。
这直接回答阶段三留下的问题"为什么小模型句子不通、逻辑破碎"——核心是规模不够。

注意：这是 CPU 上的微缩演示。真实 LLM 的 scaling law 跨越十几个数量级，
这里只在小范围内展示同一个**趋势**：给同样的数据，更大的模型能压缩出更多规律。

运行（约 15~20 分钟，建议后台）：
  python -u scaling_experiment.py
输出：scaling_curve.png + 三个规模的生成对比
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "stage3_gpt"))      # 复用 GPT
sys.path.insert(0, str(HERE.parent / "stage1_basics"))   # 复用字符分词器
from data import CharTokenizer
from model_gpt import GPT, GPTConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CORPUS = HERE.parent / "stage3_gpt" / "luxun_clean.txt"

# 三个规模：只变 d_model / 层数 / 头数 / d_ff
CONFIGS = [
    ("small",  dict(d_model=64,  n_layers=2, n_heads=4, d_ff=128)),
    ("medium", dict(d_model=128, n_layers=3, n_heads=4, d_ff=256)),
    ("large",  dict(d_model=192, n_layers=4, n_heads=4, d_ff=384)),
]

STEPS, BATCH, BLOCK, LR = 2000, 32, 64, 3e-4


def get_batch(data, vocab):
    ix = torch.randint(len(data) - BLOCK - 1, (BATCH,))
    x = torch.stack([data[i:i + BLOCK] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + BLOCK] for i in ix])
    return x, y


def train_one(name, kw, train_ids, val_ids, vocab):
    torch.manual_seed(42)
    cfg = GPTConfig(vocab_size=vocab, pad_id=0, max_len=max(BLOCK, 256), **kw)
    model = GPT(cfg).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    print(f"\n=== {name} ===  参数 {n_params:,}")

    best_val = float("inf")
    for step in range(1, STEPS + 1):
        model.train()
        x, y = get_batch(train_ids, vocab)
        loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1), ignore_index=0)
        opt.zero_grad(); loss.backward(); opt.step()
        if step == 1 or step % (STEPS // 5) == 0:
            model.eval()
            with torch.no_grad():
                xv, yv = get_batch(val_ids, vocab)
                vl = F.cross_entropy(model(xv).reshape(-1, vocab), yv.reshape(-1), ignore_index=0)
            best_val = min(best_val, vl.item())
            print(f"  step {step:5d} | train {loss.item():.3f} | val {vl.item():.3f}")
    return n_params, best_val, model, cfg


def sample(model, tok, prompt="我冒了严寒", n=40, temperature=0.7, top_k=15):
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=DEVICE)
    out = model.generate(ids, n, temperature=temperature, top_k=top_k)
    return tok.decode(out[0].tolist())


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = [r[0] for r in results]
    ys = [r[1] for r in results]
    plt.figure(figsize=(7, 4.5))
    plt.plot(xs, ys, "o-", color="#2563eb", linewidth=2, markersize=9)
    for (x, y, name) in zip(xs, ys, [r[2] for r in results]):
        plt.annotate(f"{name}\n{y:.3f}", (x, y), textcoords="offset points",
                     xytext=(8, 8), fontsize=9)
    plt.xscale("log")
    plt.xlabel("parameters (log scale)")
    plt.ylabel("best val_loss")
    plt.title("Scaling: bigger model -> lower loss (same data, same steps)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = HERE / "scaling_curve.png"
    plt.savefig(out, dpi=110)
    print(f"\n曲线已保存: {out}")


def main():
    text = CORPUS.read_text(encoding="utf-8")
    tok = CharTokenizer([text])
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    n_val = int(len(ids) * 0.1)
    train_ids, val_ids = ids[:-n_val].to(DEVICE), ids[-n_val:].to(DEVICE)
    print(f"语料 {len(ids):,} token | 词表 {len(tok)} | 设备 {DEVICE}")

    results = []   # (n_params, best_val, name, model, tok)
    for name, kw in CONFIGS:
        n, val, model, cfg = train_one(name, kw, train_ids, val_ids, len(tok))
        results.append((n, val, name, model))

    # 总结表
    print("\n" + "=" * 56)
    print(f"{'规模':8}{'参数量':>14}{'最佳 val_loss':>16}")
    for n, val, name, _ in results:
        print(f"{name:8}{n:>14,}{val:>16.4f}")
    print("=" * 56)
    print("趋势：参数量越大 → val_loss 越低（从同样数据里压缩出更多规律）")

    plot([(n, val, name) for n, val, name, _ in results])

    # 生成对比：同一 prompt，看连贯性随规模改善
    print("\n生成对比（prompt='我冒了严寒', temperature=0.7）:")
    for n, val, name, model in results:
        print(f"\n[{name}, {n:,} 参数, val_loss {val:.3f}]")
        print("  " + sample(model, tok).replace("\n", "\n  "))


if __name__ == "__main__":
    main()
