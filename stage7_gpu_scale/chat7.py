"""chat7.py —— 交互式测试阶段七模型：输入前半句，模型即时续写

用法（DGX Spark 上）：
  python -u chat7.py                # 默认探路模型
  python -u chat7.py --ckpt ckpt_large.pt
  python -u chat7.py --temperature 0.3   # 更保守

进入后输入任意前半句回车即可；exit / quit / Ctrl-D 退出。
"""

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "stage3_gpt"))
sys.path.insert(0, str(HERE.parent / "stage4_scaling_bpe"))
sys.path.insert(0, str(HERE.parent / "stage5_capstone"))
from bpe import BPETokenizer
from model_gpt import GPT, GPTConfig
from sampling import generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "ckpt_probe.pt"))
    ap.add_argument("--bpe", default=str(HERE / "cache" / "bpe.json"))
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = BPETokenizer.load(args.bpe)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型 {n_params/1e6:.1f}M | 训练 {ckpt.get('step', '?')} 步 | "
          f"T={args.temperature} | 输入前半句续写，exit 退出")

    while True:
        try:
            line = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in ("exit", "quit"):
            break
        out = generate(model, tok, line, args.max_new_tokens,
                       args.temperature, 0, args.top_p, 1.2)
        print(f"模型> {out}")


if __name__ == "__main__":
    main()
