"""generate_best.py —— 用阶段五最佳模型 + 改进解码续写

加载 train_best.py 训出的模型和 BPE，用 sampling.py 的改进解码
（温度 + top-p + 重复惩罚）续写。

用法：
  PY=python
  $PY generate_best.py --prompt "我冒了严寒"
  $PY generate_best.py --prompt "鲁镇的酒店的格局" --temperature 0.7 --top-p 0.9
  $PY generate_best.py --prompt "在我的后园" --rep-penalty 1.3   # 更强抗重复
"""

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "stage3_gpt"))
sys.path.insert(0, str(HERE.parent / "stage4_scaling_bpe"))
from bpe import BPETokenizer
from model_gpt import GPT, GPTConfig

from sampling import generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "ckpt_best.pt"))
    ap.add_argument("--bpe", default=str(HERE / "bpe_best.json"))
    ap.add_argument("--prompt", default="我冒了严寒")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--rep-penalty", type=float, default=1.2)
    ap.add_argument("--device", default="auto", help="cpu/cuda；默认 auto（有 GPU 用 GPU）")
    args = ap.parse_args()

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = BPETokenizer.load(args.bpe)

    out = generate(model, tok, args.prompt, args.max_new_tokens,
                   args.temperature, args.top_k, args.top_p, args.rep_penalty)
    print(f"prompt: {args.prompt!r} | T={args.temperature} "
          f"top_p={args.top_p} rep_penalty={args.rep_penalty}")
    print("-" * 50)
    print(out)


if __name__ == "__main__":
    main()
