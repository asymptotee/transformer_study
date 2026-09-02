"""生成脚本：加载 checkpoint，对 prompt 做自回归生成

用法示例：
    echo "一二三四五" | python generate.py            # 管道输入（reverse 任务期望: 五四三二一）
    python generate.py                                # 交互模式，逐行输入 prompt
    python generate.py --temperature 0.8              # 温度采样（默认 0 = 贪心）

任务类型存在 checkpoint 里，无需再指定：
- reverse 任务：输入整行文本，输出反转结果
- complete 任务：输入前半句，输出模型续写的后半句
"""

import argparse
import sys
from pathlib import Path

import torch

from data import BOS_ID, EOS_ID, CharTokenizer
from model import Transformer, TransformerConfig

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser(description="用训练好的 Transformer 做生成")
    ap.add_argument("--ckpt", default=str(HERE / "ckpt.pt"))
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="采样温度：0 为贪心解码，>0 时按概率采样，越高越发散")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt)  # 内容只有张量和基本类型，weights_only 默认即可
    cfg = TransformerConfig(**ckpt["config"])
    model = Transformer(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tokenizer = CharTokenizer.from_vocab(ckpt["vocab"])
    task = ckpt["task"]

    def run(prompt: str) -> str:
        src = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
        out_ids = model.generate(
            src, bos_id=BOS_ID, eos_id=EOS_ID,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        )
        return tokenizer.decode(out_ids[0].tolist())

    hint = "输入整行文本看反转" if task == "reverse" else "输入前半句看续写"
    if not sys.stdin.isatty():  # 管道输入：逐行处理
        for line in sys.stdin:
            line = line.strip()
            if line:
                print(run(line))
    else:  # 交互模式
        print(f"任务: {task} | {hint} | Ctrl-D 退出")
        while True:
            try:
                prompt = input("prompt> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if prompt.strip():
                print(run(prompt.strip()))


if __name__ == "__main__":
    main()
