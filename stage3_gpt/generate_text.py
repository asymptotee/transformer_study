"""generate_text.py —— prompt 续写（GPT 的标准用法）

加载训好的 GPT，从一段 prompt 自回归续写。这是语言模型区别于 seq2seq 的地方：
不是"输入→输出"，而是"给个开头，接着写"。

用法：
  python generate_text.py                          # 默认从换行符开始，生成几行新"诗"
  python generate_text.py --prompt "山峰"          # 从"山峰"续写
  python generate_text.py --temperature 0.5        # 更保守（更确定）
  python generate_text.py --temperature 1.2 --top-k 20   # 更发散
"""

import argparse
from pathlib import Path

import torch

HERE = Path(__file__).parent

import sys
sys.path.insert(0, str(HERE.parent / "stage1_basics"))
from data import CharTokenizer
from model_gpt import GPT, GPTConfig


def main():
    ap = argparse.ArgumentParser(description="用训好的 GPT 续写文本")
    ap.add_argument("--ckpt", default=str(HERE / "ckpt_gpt.pt"))
    ap.add_argument("--prompt", default="\n", help="续写起点；默认换行符=另起一行")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--device", default="auto", help="cpu/cuda；默认 auto（有 GPU 用 GPU）")
    args = ap.parse_args()

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt)                # 内容仅张量+基本类型，默认 weights_only 即可
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = CharTokenizer.from_vocab(ckpt["vocab"])

    ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(ids, args.max_new_tokens, args.temperature, args.top_k)
    text = tok.decode(out[0].tolist())

    print(f"prompt: {args.prompt!r} | temperature={args.temperature} | top_k={args.top_k}")
    print("-" * 40)
    print(text)


if __name__ == "__main__":
    main()
