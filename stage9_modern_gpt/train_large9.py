"""train_large9.py —— stage9 零件 A/B 训练脚本(9.1 起)

stage7 train_large.py 的参数化版:模型从 model_modern.py 导入(带 norm/ff
开关),其余(数据缓存、val 口径、lr 调度、best-ckpt)与 stage7 逐字一致。

**A/B 公平性约定**(9.1 起所有零件对照都守这条):
  · 复用 stage7 的 cache/tokens.pt(同一个 5% val 切分)——不重新编码
  · --seed 默认 42:两臂同 seed = 看到完全相同的训练 batch 与 val batch,
    观察到的差异只能来自被换的那个零件(唯一变量)
  · 3000 步探路(≈20min)+ 16-batch 口径定论(见 9.0 噪声分析)

**ckpt 存盘规则**(11.5 教训,2026-09-15):训练内 val 只有 4 batch,
噪声 ±0.05,而余弦末段的真实改善远小于噪声 → "best val" 选点等于抽彩票。
11.5 长跑因此把 186k 的终点丢在了 130k 的幸运低点上,29h 的收敛终态丢失。
现规则:
  · best-ckpt 照旧(用于崩溃恢复,不代表终态)
  · **终步无条件存盘**到 --save-final(默认 ckpt 名加 _final)
  · 续训(--init-from/--start-step)把权重加载与 lr 位置解耦,专供故障恢复

用法(Spark):
  PY=~/llm_study/.venv/bin/python
  # 4 臂(control / 单换 Norm / 单换 FFN / 全换):
  $PY -u train_large9.py --norm layernorm --ff gelu --ckpt ckpt_91_control.pt
  $PY -u train_large9.py --norm rms      --ff gelu --ckpt ckpt_91_rms.pt
  $PY -u train_large9.py --norm layernorm --ff silu --ckpt ckpt_91_swiglu.pt
  $PY -u train_large9.py --norm rms      --ff silu --ckpt ckpt_91_combo.pt
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "stage4_scaling_bpe"))
sys.path.insert(0, str(REPO / "stage5_capstone"))
from bpe import BPETokenizer, EOS_ID            # noqa: E402
from bpb import build_token_bytes, ce_nats_and_bytes, bpb   # noqa: E402
from model_modern import GPT, GPTConfig         # noqa: E402
from sampling import generate                    # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_batch(data, block, batch):
    ix = torch.randint(len(data) - block - 1, (batch,), device=data.device)
    x = torch.stack([data[i:i + block] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block] for i in ix])
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(REPO / "stage7_gpu_scale" / "cache"),
                    help="stage7 的 BPE + token 缓存(勿删,两臂共用才能 A/B)")
    ap.add_argument("--norm", choices=["layernorm", "rms"], default="layernorm")
    ap.add_argument("--ff", choices=["gelu", "silu"], default="gelu")
    ap.add_argument("--pos", choices=["sinusoidal", "rope"], default="sinusoidal")
    ap.add_argument("--rope-theta", type=float, default=1e6)
    ap.add_argument("--rope-ctx", type=int, default=4096,
                    help="RoPE 预计算表长(外推上限,不是训练长度)")
    ap.add_argument("--n-kv-heads", type=int, default=None,
                    help="GQA 的 K/V 头数;None=与 Q 同头数(MHA)")
    ap.add_argument("--d-model", type=int, default=768)
    ap.add_argument("--n-layers", type=int, default=12)
    ap.add_argument("--n-heads", type=int, default=12)
    ap.add_argument("--d-ff", type=int, default=3072)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42, help="两臂同 seed(见文件头)")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ckpt", default=str(HERE / "ckpt_91_control.pt"))
    ap.add_argument("--save-final", default=None,
                    help="终步无条件存盘路径(默认 ckpt 名加 _final;见文件头)")
    ap.add_argument("--init-from", default=None,
                    help="只加载权重续训(故障恢复用;lr 位置由 --start-step 决定)")
    ap.add_argument("--start-step", type=int, default=0,
                    help="续训起始步:决定 lr 在余弦上的位置(步数按原 schedule 计)")
    args = ap.parse_args()

    global DEVICE
    if args.device != "auto":
        DEVICE = args.device
    use_amp = DEVICE.startswith("cuda") and not args.no_amp

    torch.manual_seed(args.seed)
    cache = Path(args.cache_dir)
    tok = BPETokenizer.load(cache / "bpe.json")
    token_bytes = build_token_bytes(tok)         # bpb 用:token→UTF-8 字节数
    ids = torch.load(cache / "tokens.pt")
    n_val = int(len(ids) * 0.05)
    train_ids = ids[:-n_val].to(DEVICE)
    val_ids = ids[-n_val:].to(DEVICE)

    cfg = GPTConfig(vocab_size=len(tok), pad_id=0,
                    max_len=max(args.block_size, 512),
                    d_model=args.d_model, n_layers=args.n_layers,
                    n_heads=args.n_heads, d_ff=args.d_ff,
                    norm=args.norm, ff=args.ff,
                    pos=args.pos, rope_theta=args.rope_theta,
                    rope_ctx=args.rope_ctx, n_kv_heads=args.n_kv_heads)
    model = GPT(cfg).to(DEVICE)
    if args.init_from:
        ck = torch.load(args.init_from, map_location="cpu")
        model.load_state_dict(ck["model"])       # strict:配置不匹配立刻暴露
        print(f"续训: 权重 ← {args.init_from}(原存盘 step {ck.get('step')}),"
              f"lr 位置 = step {args.start_step}", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"设备: {DEVICE} | AMP: {use_amp} | 参数: {n_params:,}"
          f" | norm={cfg.norm} ff={cfg.ff} pos={cfg.pos}"
          f"(θ={cfg.rope_theta:g}, 表长 {cfg.rope_ctx}) | seed={args.seed}"
          f" | 词表: {len(tok):,} | 训练 token: {len(train_ids):,}"
          f" | block={args.block_size} batch={args.batch_size}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * t))

    def val_loss():
        """返回 (按 token 平均 CE, bpb)。bpb = 词表无关口径(见 bpb.py)。"""
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=use_amp):
            tot, n = 0.0, 0
            nats_all, bytes_all = 0.0, 0.0
            for _ in range(4):                      # 训练内口径:4 batch(探路用)
                x, y = get_batch(val_ids, args.block_size, args.batch_size)
                logits = model(x)
                nats, nbytes = ce_nats_and_bytes(logits, y, token_bytes,
                                                 ignore_index=0)
                loss = F.cross_entropy(logits.reshape(-1, len(tok)),
                                       y.reshape(-1), ignore_index=0)
                tot += loss.item() * x.numel()
                n += x.numel()
                nats_all += nats
                bytes_all += nbytes
        return tot / n, bpb(nats_all, bytes_all)

    def show_samples():
        model.eval()
        for seed in ["中国的首都是", "长城位于", "水的化学式是"]:
            out = generate(model, tok, seed, max_new=40, temperature=0.8,
                           top_k=0, top_p=0.9, rep_penalty=1.2)
            print(f"    {seed!r} → {out[:60]!r}", flush=True)

    best = float("inf")
    t0 = time.time()
    for step in range(args.start_step + 1, args.steps + 1):
        model.train()
        opt.param_groups[0]["lr"] = lr_at(step)
        x, y = get_batch(train_ids, args.block_size, args.batch_size)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss = F.cross_entropy(model(x).reshape(-1, len(tok)),
                                   y.reshape(-1), ignore_index=0)
        opt.zero_grad(); loss.backward(); opt.step()

        if step == 1 or step % args.eval_every == 0:
            vl, vb = val_loss()
            sps = (step - args.start_step) / (time.time() - t0)
            print(f"  step {step:5d}/{args.steps} | train {loss.item():.3f} "
                  f"| val {vl:.3f} | bpb {vb:.4f} | "
                  f"lr {opt.param_groups[0]['lr']:.2e} "
                  f"| {sps:.1f} step/s", flush=True)
            if vl < best:
                best = vl
                torch.save({"model": model.state_dict(), "config": vars(cfg),
                            "step": step}, args.ckpt)
                print(f"    ✓ 新最佳 val {best:.3f} → {args.ckpt}", flush=True)
            if step > args.warmup:
                show_samples()

    # 终步无条件存盘(见文件头:4-batch 噪声会让 best-ckpt 选错点,终态必须另存)
    final_path = args.save_final or args.ckpt.replace(".pt", "_final.pt")
    torch.save({"model": model.state_dict(), "config": vars(cfg),
                "step": args.steps}, final_path)
    print(f"\n完成 [{args.norm}/{args.ff}]: 最佳 val {best:.3f} | "
          f"总耗时 {time.time()-t0:.0f}s ({(args.steps-args.start_step)/(time.time()-t0):.1f} step/s)",
          flush=True)
    print(f"✓ 终步 {args.steps} 无条件存盘 → {final_path}", flush=True)


if __name__ == "__main__":
    main()
