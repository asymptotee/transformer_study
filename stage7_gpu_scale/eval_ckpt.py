"""eval_ckpt.py —— 评估训好的阶段七模型：多 prompt 生成，看"学到什么程度"

用改进解码（温度 + top-p + 重复惩罚，阶段五的 sampling.py）对一组固定 prompt 续写，
直观展示模型当前水平：词表/词序/句子/知识，各学到哪一层。

用法：
  python -u eval_ckpt.py --ckpt ckpt_probe.pt
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

PROMPTS = [
    "中国的首都是",
    "北京是",
    "数学是",
    "水是一种",
    "长城位于",
    "鲁迅是",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "ckpt_large.pt"))
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
    print(f"模型: {n_params:,} 参数 | 词表 {len(tok):,} | 训练步数: {ckpt.get('step', '?')}"
          f" | 设备 {device}\n")

    for p in PROMPTS:
        out = generate(model, tok, p, args.max_new_tokens, args.temperature,
                       0, args.top_p, 1.2)
        print(f"【{p}】")
        print(f"  {out[:100]}\n")


if __name__ == "__main__":
    main()
