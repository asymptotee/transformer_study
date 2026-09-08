"""eval_chat10.py —— 10.3 同格式对比:我们的 chat-SFT 模型 vs 官方 64M

chat 格式 20 题:user 消息渲染成训练时同款模板文本
  <|im_start|>user\n{问题}<|im_end|>\n<|im_start|>assistant\n
(与 render_sft_chat.py 用官方 apply_chat_template 渲染的形态一致——无
system 时模板就是这个样子)。贪心续写,遇 <|im_end|> 或 EOS 或 100 token 停。

对照锚点(9.5 已测):官方 minimind-3 64M 在 chat 格式 = 6/20(5 真 1 假)。
若我们的 chat-SFT 达到/超过它 → "数据剂量 + 格式工艺"补齐的定论。

用法(Spark):
  ~/llm_study/.venv/bin/python eval_chat10.py
"""

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
S11 = HERE.parent / "stage11_datascale"
sys.path.insert(0, str(HERE.parent / "stage4_scaling_bpe"))
sys.path.insert(0, str(HERE.parent / "stage9_modern_gpt"))
from bpe import BPETokenizer, EOS_ID                       # noqa: E402
from eval_harness import HIGH, LOW                         # noqa: E402
from model_modern import GPT, GPTConfig                    # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IM_END = "<|im_end|>"
MAX_NEW = 100


def chat_prompt(q):
    return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "ckpt_10_chat.pt"))
    ap.add_argument("--bpe", default=str(S11 / "cache_mm" / "bpe.json"))
    ap.add_argument("--label", default="10-chat")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    global DEVICE
    if args.device != "auto":
        DEVICE = args.device

    tok = BPETokenizer.load(args.bpe)
    ckpt = torch.load(args.ckpt)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"=== {args.label} | {sum(p.numel() for p in model.parameters())/1e6:.1f}M",
          flush=True)

    def answer(q):
        ids = tok.encode(chat_prompt(q))
        start = len(ids)
        for _ in range(MAX_NEW):
            ctx = torch.tensor([ids[-cfg.max_len:]], device=DEVICE)
            nxt = int(model(ctx)[0, -1].argmax().item())
            ids.append(nxt)
            text = tok.decode(ids[start:])
            if IM_END in text or nxt == EOS_ID:
                break
        return text.split(IM_END)[0].strip()

    ok = hi = lo = 0
    for i, (q, ans) in enumerate(HIGH + LOW):
        out = answer(q)
        hit = ans.lower() in out.lower()
        ok += hit
        if i < 10:
            hi += hit
        else:
            lo += hit
        mark = "✓" if hit else "✗"
        print(f"  {mark} {q}")
        print(f"      期望 {ans} | {out[:70]!r}")
    print(f"\n答对 {ok}/20(高频 {hi}/10 | 低频 {lo}/10)")
    print("对照:官方 minimind-3 64M chat 格式 6/20(真 ~5) | 11-mm raw 格式 ≈4/20")


if __name__ == "__main__":
    main()
