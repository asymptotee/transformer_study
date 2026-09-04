"""eval_extrap.py —— 长度外推实验:短训长测(9.2 核心实验)

问题:模型只在 256 长的窗口上训练过,给更长的上下文(512/1024/2048),
困惑度会怎样?
  · sinusoidal(正弦位置编码):位置表只建到 cfg.max_len(512),
    超长输入直接越界——"查表式位置编码"的天花板;
  · RoPE:频率表预计算到 rope_ctx(4096),任意长都能算——
    但训练只见过 256,能否"外推"?这就是本实验要测的。

方法:在验证集上取**长度 L 的窗口**(L = 训练长度/2~8 倍),整窗前向算
next-token 困惑度。关键设计:
  1. 每个 L 的窗口只抽一次(RNG 固定),所有方案共用同一批窗口 →
     "换位置编码方案"的对比是零噪声的(和 check_isomorphic 同哲学);
  2. 方案(scheme)全是**推理期改表,不动权重**:
       plain     = 训练时的表(θ=训练配置)
       theta1e4  = 把 base 频换成 θ=1e4 重算表(换底实验)
       yarnN     = YaRN,original=训练长度 256,factor=N(高频插值、低频保留)
   对同一个训练好的 rope 模型,各方案在同批窗口上直接比 ppl。

用法(Spark):
  ~/llm_study/.venv/bin/python eval_extrap.py --ckpt ckpt_92_rope.pt \
      --label 92-rope --schemes plain,theta1e4,yarn4,yarn8
  输出:results/extrap_<label>.csv + .png(曲线)

注意:YaRN 的 original_max 应等于**训练窗口长**(默认 256,若以后换
block-size 训练要传 --yarn-orig 修正)。
"""

import argparse
import csv
import importlib
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "stage3_gpt"))
sys.path.insert(0, str(REPO / "stage4_scaling_bpe"))
sys.path.insert(0, str(HERE))
from bpe import BPETokenizer                     # noqa: E402
from model_modern import GPT, GPTConfig          # noqa: E402
from rope import precompute_freqs_cis            # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_LENGTHS = "256,384,512,768,1024,1536,2048"


def build_model(ckpt_path, module):
    ckpt = torch.load(ckpt_path)
    mod = importlib.import_module(module)
    cfg = mod.GPTConfig(**ckpt["config"])
    m = mod.GPT(cfg).to(DEVICE)
    m.load_state_dict(ckpt["model"])
    m.eval()
    return m, cfg


def swap_freqs(model, head_dim, ctx_len, theta, yarn=None):
    """推理期重算 RoPE 表(换底/加 YaRN),不改权重。"""
    cos, sin = precompute_freqs_cis(head_dim, ctx_len, theta,
                                    rope_scaling=yarn)
    model.freqs_cos, model.freqs_sin = cos.to(DEVICE), sin.to(DEVICE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bpe", default=str(REPO / "stage7_gpu_scale" / "cache" / "bpe.json"))
    ap.add_argument("--tokens", default=str(REPO / "stage7_gpu_scale" / "cache" / "tokens.pt"))
    ap.add_argument("--model-module", default="model_modern")
    ap.add_argument("--label", default="")
    ap.add_argument("--lengths", default=DEFAULT_LENGTHS)
    ap.add_argument("--schemes", default="plain,theta1e4,yarn4,yarn8",
                    help="rope 模型才生效;sinusoidal 忽略")
    ap.add_argument("--train-ctx", type=int, default=256,
                    help="训练窗口长(YaRN 的 original_max、参考线)")
    ap.add_argument("--yarn-orig", type=int, default=256)
    ap.add_argument("--n-batch", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    global DEVICE
    if args.device != "auto":
        DEVICE = args.device

    lengths = [int(x) for x in args.lengths.split(",")]
    model, cfg = build_model(args.ckpt, args.model_module)
    tok = BPETokenizer.load(args.bpe)
    is_rope = cfg.pos == "rope"
    head_dim = cfg.d_model // cfg.n_heads
    label = args.label or Path(args.ckpt).stem
    print(f"=== {label} | pos={cfg.pos} | 训练窗口 {args.train_ctx}"
          f" | 参数 {sum(p.numel() for p in model.parameters()):,}")

    ids = torch.load(args.tokens)
    val_ids = ids[-int(len(ids) * 0.05):].to(DEVICE)
    max_L = cfg.rope_ctx if is_rope else cfg.max_len   # sinusoidal 的查表天花板

    # 每长度抽一次窗口,所有方案共用(零噪声对比)
    windows = {}
    torch.manual_seed(0)
    for L in lengths:
        if L > max_L:
            continue
        ws = []
        for _ in range(args.n_batch):
            ix = torch.randint(len(val_ids) - L - 1, (args.batch_size,), device=DEVICE)
            ws.append((torch.stack([val_ids[i:i + L] for i in ix]),
                       torch.stack([val_ids[i + 1:i + 1 + L] for i in ix])))
        windows[L] = ws

    schemes = ["plain"] if not is_rope else args.schemes.split(",")
    rows = []
    for scheme in schemes:
        if scheme == "plain":
            pass                                            # 训练时的表
        elif scheme == "theta1e4":
            swap_freqs(model, head_dim, cfg.rope_ctx, 1e4)
        elif scheme.startswith("yarn"):
            factor = float(scheme[4:])
            swap_freqs(model, head_dim, cfg.rope_ctx, cfg.rope_theta,
                       yarn={"original_max_position_embeddings": args.yarn_orig,
                             "factor": factor, "beta_fast": 32.0, "beta_slow": 1.0})
        print(f"--- scheme: {scheme}")
        for L in sorted(windows):
            tot, n = 0.0, 0
            with torch.no_grad():
                for x, y in windows[L]:
                    logits = model(x)
                    tot += F.cross_entropy(logits.reshape(-1, len(tok)),
                                           y.reshape(-1),
                                           ignore_index=0).item() * x.numel()
                    n += x.numel()
            ppl = math.exp(tot / n)
            print(f"    L={L:5d}  loss {tot/n:.3f}  ppl {ppl:8.1f}")
            rows.append({"model": label, "scheme": scheme, "L": L,
                         "loss": round(tot / n, 3), "ppl": round(ppl, 1)})

    out_csv = HERE / "results" / f"extrap_{label}.csv"
    out_csv.parent.mkdir(exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "scheme", "L", "loss", "ppl"])
        w.writeheader()
        w.writerows(rows)
    print(f"CSV: {out_csv}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 5))
        for scheme in schemes:
            pts = [r for r in rows if r["scheme"] == scheme]
            plt.plot([p["L"] for p in pts], [p["ppl"] for p in pts],
                     marker="o", label=scheme)
        plt.axvline(args.train_ctx, ls="--", color="gray", alpha=0.7)
        plt.text(args.train_ctx, plt.ylim()[1] * 0.95, f"train {args.train_ctx}",
                 ha="right", fontsize=8)
        plt.xlabel("context length L")
        plt.ylabel("perplexity")
        plt.yscale("log")
        plt.legend()
        plt.title(f"length extrapolation: {label} (pos={cfg.pos})")
        out_png = HERE / "results" / f"extrap_{label}.png"
        plt.savefig(out_png, dpi=120, bbox_inches="tight")
        print(f"PNG: {out_png}")
    except Exception as e:                                # matplotlib 缺失也能出 CSV
        print(f"(无 PNG: {e})")


if __name__ == "__main__":
    main()
