"""demo_matrices.py — 用真实 torch 训练迷你 Transformer，dump 真实矩阵

复用 model.py 的 Transformer（不重写）。在随机字符串上训练反转任务，
用 forward hook 抓取注意力权重，展示《ARCHITECTURE.md 第九节》里那些
示意矩阵的**真实版本**：

  1. embedding 表（真实学到的值）
  2. 交叉注意力：初始化时一片噪声 → 训练后出现反对角线（反转的本质）
  3. logits 矩阵 + argmax vs 标签
  4. 自回归生成的逐步轨迹
  5. 在没见过的字符串上测泛化

运行：
  python demo_matrices.py
"""

import random

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from data import BOS_ID, EOS_ID, PAD_ID, CharTokenizer
from model import Transformer, TransformerConfig

# ---------- 缩小版参数（为了矩阵能打印出来）----------
CHARS = "abcde"
torch.manual_seed(42)
random.seed(123)

tok = CharTokenizer([CHARS])           # 词表 = 4 特殊符 + a b c d e，共 9 个
V = len(tok)
cfg = TransformerConfig(vocab_size=V, d_model=16, n_heads=2, n_layers=2,
                        d_ff=32, dropout=0.1, max_len=16, pad_id=PAD_ID)
model = Transformer(cfg)
print(f"词表 {V} 个: {tok.itos}")
print(f"参数: {sum(p.numel() for p in model.parameters()):,}  "
      f"(d_model=16, 2 头, d_k=8, 2 层)\n")


# ---------- 用 hook 抓注意力权重 ----------
# MultiHeadAttention.forward 返回 (out, attn)，hook 把 attn 截下来
CAP = {"on": False, "data": {}}
def make_hook(name):
    def hook(mod, inp, out):
        if CAP["on"]:
            CAP["data"][name] = out[1].detach()
    return hook
for i, block in enumerate(model.decoder):
    block.self_attn.register_forward_hook(make_hook(f"dec{i}.self"))
    block.cross_attn.register_forward_hook(make_hook(f"dec{i}.cross"))


# ---------- 打印工具 ----------
def disp(i):
    """token id -> 可读名字"""
    s = tok.itos[i]
    return s if not s.startswith("<") else s
RAMP = " ░▒▓█"
def shade(v):
    return RAMP[min(len(RAMP) - 1, int(max(0.0, min(1.0, v)) * len(RAMP)))]

def heatmap(mat, row_labels, col_labels, title):
    """mat: 2D list/array，打印数值网格 + 每行一条明暗条"""
    print(title)
    print("         " + " ".join(f"{c:>4}" for c in col_labels))
    for r, row in enumerate(mat):
        nums = " ".join(f"{v:4.2f}" for v in row)
        bar = "".join(shade(v) for v in row)
        print(f"  {row_labels[r]:>5} [ {nums} ]  {bar}")
    print()


def probe(src_str):
    """teacher-forcing 前向，抓注意力。返回 (src, tgt, logits)"""
    CAP["data"].clear(); CAP["on"] = True
    src = torch.tensor([tok.encode(src_str)])
    tgt = torch.tensor([[BOS_ID] + tok.encode(src_str[::-1]) + [EOS_ID]])
    model.eval()
    with torch.no_grad():
        logits = model(src, tgt[:, :-1])
    CAP["on"] = False
    return src, tgt, logits


# ============================================================
# ① embedding 表（初始化后，尚未训练）
# ============================================================
print("=" * 64)
print("① embedding 表 (9 词 × 16 维) —— 此时还是随机初始化")
print("=" * 64)
W = model.embedding.weight.detach()
print("       " + " ".join(f"{j:>5}" for j in range(16)))
for i in range(V):
    print(f"  {disp(i):>4} " + " ".join(f"{v:5.2f}" for v in W[i].tolist()))
print("  （<pad> 行恒为 0；输出头 lm_head 与这张表共享权重）\n")


# ============================================================
# ② 训练前的交叉注意力：一片噪声
# ============================================================
print("=" * 64)
print("② 训练前，src='abcde' 的交叉注意力（最后一层）")
print("    行 = decoder 输入位置，列 = src 位置。未训练 → 接近均匀噪声")
print("=" * 64)
src, tgt, _ = probe("abcde")
row_labels = [disp(i) for i in tgt[0, :-1].tolist()]   # decoder 输入
col_labels = [disp(i) for i in src[0].tolist()]        # src
cross = CAP["data"]["dec1.cross"][0]                   # (H, Tq, Tk)
for h in range(cfg.n_heads):
    heatmap(cross[h].tolist(), row_labels, col_labels, f"  头 {h}:")


# ============================================================
# ③ 训练：随机字符串反转（逼模型学算法，而非背诵）
# ============================================================
print("=" * 64)
print("③ 训练 2500 步（每步 batch=64 的随机字符串反转）")
print("=" * 64)
g = random.Random(7)
def make_batch(n):
    srcs, tgts = [], []
    for _ in range(n):
        s = "".join(g.choice(CHARS) for _ in range(g.randint(3, 6)))
        srcs.append(torch.tensor(tok.encode(s)))
        tgts.append(torch.tensor([BOS_ID] + tok.encode(s[::-1]) + [EOS_ID]))
    src = pad_sequence(srcs, batch_first=True, padding_value=PAD_ID)
    tgt = pad_sequence(tgts, batch_first=True, padding_value=PAD_ID)
    return src, tgt

opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
for step in range(1, 2501):
    model.train()
    src, tgt = make_batch(64)
    logits = model(src, tgt[:, :-1])
    loss = F.cross_entropy(logits.reshape(-1, V), tgt[:, 1:].reshape(-1),
                           ignore_index=PAD_ID)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 250 == 0:
        print(f"  step {step:4d} | loss {loss.item():.4f}")
print()


# ============================================================
# ④ 训练后的交叉注意力：反对角线出现了
# ============================================================
print("=" * 64)
print("④ 训练后，src='abcde' 的交叉注意力（最后一层）")
print("    预测 e→看src的e(最右)，预测 d→看d…… 形成反对角线！")
print("=" * 64)
src, tgt, logits = probe("abcde")
row_labels = [disp(i) for i in tgt[0, :-1].tolist()]
col_labels = [disp(i) for i in src[0].tolist()]
cross = CAP["data"]["dec1.cross"][0]
for h in range(cfg.n_heads):
    heatmap(cross[h].tolist(), row_labels, col_labels, f"  头 {h}:")


# ============================================================
# ⑤ logits 矩阵 + argmax vs 标签
# ============================================================
print("=" * 64)
print("⑤ 训练后 logits（src='abcde'），每行预测一个位置，argmax 应等于标签")
print("=" * 64)
labels = tgt[0, 1:].tolist()
lg = logits[0].tolist()
print("  位置  输入→标签  " + " ".join(f"{disp(j):>5}" for j in range(V)) + "   argmax")
for p, row in enumerate(lg):
    inp = disp(tgt[0, :-1][p].item())
    lab = disp(labels[p])
    am = int(max(range(V), key=lambda j: row[j]))
    mark = "✓" if am == labels[p] else "✗"
    print(f"   {p}    {inp}→{lab}   " + " ".join(f"{v:5.1f}" for v in row)
          + f"   {disp(am)} {mark}")
print()


# ============================================================
# ⑥ 泛化：没见过的字符串也能反转
# ============================================================
print("=" * 64)
print("⑥ 泛化测试（这些字符串训练时几乎没见过）")
print("=" * 64)
model.eval()
tests = ["abcde", "bead", "cade", "ee", "decade"[:6], "ab"]
n_ok = 0
for s in tests:
    src = torch.tensor([tok.encode(s)])
    with torch.no_grad():
        out = model.generate(src, BOS_ID, EOS_ID, max_new_tokens=10)
    got = tok.decode(out[0].tolist())
    ok = got == s[::-1]
    n_ok += ok
    print(f"  {s!r:>10} -> {got!r:<10} 期望 {s[::-1]!r:<10} {'OK' if ok else 'MISS'}")
print()


# ============================================================
# ⑦ 自回归生成逐步轨迹
# ============================================================
print("=" * 64)
print("⑦ 自回归生成轨迹：src='abc'，一个 token 一个 token 长出来")
print("=" * 64)
src = torch.tensor([tok.encode("abc")])
enc_out, src_mask = model.encode(src)
ys = torch.tensor([[BOS_ID]])
print("  步骤  已生成 ys        末位 logits[" + " ".join(disp(j) for j in range(V)) + "]  → 选出")
for step in range(10):
    with torch.no_grad():
        last = model.decode(ys, enc_out, src_mask)[:, -1]      # (1, V)
    nxt = int(last.argmax(-1).item())
    ys_str = "".join(disp(i) for i in ys[0].tolist())
    lg_str = " ".join(f"{v:4.1f}" for v in last[0].tolist())
    print(f"   {step}    {ys_str:<14} [ {lg_str} ]  → {disp(nxt)}")
    ys = torch.cat([ys, torch.tensor([[nxt]])], dim=1)
    if nxt == EOS_ID:
        break
print(f"\n  输出（去掉 <bos>）: {tok.decode(ys[0, 1:].tolist())!r}")
