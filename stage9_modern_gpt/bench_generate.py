"""bench_generate.py —— KV cache 基准(9.3):一致性自检 + 加速比 + KV 显存账

三个输出:
  1. **一致性自检**:同 seed 下,无缓存生成(sampling.generate)与缓存生成
     (generate_cached)逐 token 完全一致吗?
     ——KV cache 的实现若正确,两者 logits 数学上相等,采样序列必然相同。
       文本一致 = cache 没算错;这是比"loss 差不多"强得多的正确性证明。
  2. **墙钟加速比**:同参数(温度/top-p/重复惩罚)、同长度,N 次取平均。
     理论省的是"每步重算旧 token 的注意力"——序列越长、省得越多。
  3. **KV 显存账**:生成到 L token 时,每层缓存 (k,v) 占多少显存?
     MHA vs GQA 各报一份,看 GQA 压缩在哪儿(9.3 先 cache 后 GQA 的动机)。

用法(Spark):
  ~/llm_study/.venv/bin/python bench_generate.py --ckpt ckpt_92_rope.pt
"""

import argparse
import importlib
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "stage3_gpt"))
sys.path.insert(0, str(REPO / "stage4_scaling_bpe"))
sys.path.insert(0, str(REPO / "stage5_capstone"))
sys.path.insert(0, str(HERE))
from bpe import BPETokenizer                                # noqa: E402
from generate_cached import generate_cached_ids             # noqa: E402
from sampling import generate as generate_uncached          # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bpe", default=str(REPO / "stage7_gpu_scale" / "cache" / "bpe.json"))
    ap.add_argument("--model-module", default="model_modern")
    ap.add_argument("--prompt", default="北京是中国的首都,也是中国历史最悠久的城市之一。")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=3, help="计时重复次数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kv-len", type=int, default=512, help="KV 显存账按多长算")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    global DEVICE
    if args.device != "auto":
        DEVICE = args.device

    mod = importlib.import_module(args.model_module)
    ckpt = torch.load(args.ckpt)
    cfg = mod.GPTConfig(**ckpt["config"])
    model = mod.GPT(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = BPETokenizer.load(args.bpe)
    n_prompt = len(tok.encode(args.prompt))
    assert n_prompt < cfg.max_len, "prompt 不能触发旧路径的截断,否则两路输入不等价"
    assert cfg.pos == "rope", "bench 需要 rope 模型(旧路径上限 max_len)"

    # ---- [1] 一致性自检 ----
    torch.manual_seed(args.seed)
    out_uncached = generate_uncached(model, tok, args.prompt, args.max_new,
                                     0.8, 0, 0.9, 1.2)
    torch.manual_seed(args.seed)
    ids_cached = generate_cached_ids(model, tok, args.prompt, args.max_new,
                                     temperature=0.8, top_k=0, top_p=0.9,
                                     rep_penalty=1.2)
    same = out_uncached == tok.decode(tok.encode(args.prompt) + ids_cached)
    print(f"[1] 一致性自检: 无缓存 vs 有缓存 文本一致 = {same}"
          f"({'✓' if same else '✗ 有 bug!'})")

    # ---- [2] 墙钟加速比(各 repeats 次取平均)----
    def bench(fn):
        ts = []
        for _ in range(args.repeats):
            torch.manual_seed(args.seed)
            t0 = time.time()
            fn()
            ts.append(time.time() - t0)
        return sum(ts) / len(ts)

    t_old = bench(lambda: generate_uncached(model, tok, args.prompt,
                                            args.max_new, 0.8, 0, 0.9, 1.2))
    t_new = bench(lambda: generate_cached_ids(model, tok, args.prompt,
                                              args.max_new, temperature=0.8,
                                              top_k=0, top_p=0.9, rep_penalty=1.2))
    print(f"[2] 墙钟(均值 {args.repeats} 次, {args.max_new} new tokens, "
          f"prompt {n_prompt}): 无缓存 {t_old:.2f}s ({args.max_new / t_old:.0f} "
          f"tok/s) | 有缓存 {t_new:.2f}s ({args.max_new / t_new:.0f} tok/s) "
          f"| 加速 {t_old / t_new:.1f}×")

    # ---- [3] KV 显存账 ----
    head_dim = cfg.d_model // cfg.n_heads
    print(f"[3] KV 缓存显存(L={args.kv_len}, 单序列, bf16 2B/元素, 每层 "
          f"k+v 两组):")
    for name, h in (("MHA", cfg.n_heads), ("GQA", cfg.n_kv_heads or cfg.n_heads)):
        mb = 2 * h * head_dim * 2 * args.kv_len * cfg.n_layers / 2 ** 20
        print(f"    {name}({h} kv 头): {mb:.1f} MB")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型: {n_params:,} 参数 | n_heads={cfg.n_heads} "
          f"n_kv_heads={cfg.n_kv_heads} | {DEVICE}")


if __name__ == "__main__":
    main()
