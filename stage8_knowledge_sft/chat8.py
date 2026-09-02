"""chat8.py —— 交互式问答测试：输入问题，SFT 模型即时作答

用法（DGX Spark）：
  python -u chat8.py                # 全量 SFT
  python -u chat8.py --model lora   # LoRA SFT
  python -u chat8.py --model base   # 基座对比（只会续写）

输入任意问题回车即可；exit / quit / Ctrl-D 退出。
--raw 则不加"问/答"模板，纯续写。
"""

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
STAGE7 = HERE.parent / "stage7_gpu_scale"
sys.path.insert(0, str(HERE.parent / "stage3_gpt"))
sys.path.insert(0, str(HERE.parent / "stage4_scaling_bpe"))
sys.path.insert(0, str(HERE.parent / "stage6_sft"))
from bpe import BPETokenizer, EOS_ID
from model_gpt import GPT, GPTConfig
from lora import inject_lora

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load(which):
    tok = BPETokenizer.load(STAGE7 / "cache" / "bpe.json")
    if which == "base":
        ckpt = torch.load(STAGE7 / "ckpt_large.pt")
        cfg = GPTConfig(**ckpt["config"])
        m = GPT(cfg).to(DEVICE); m.load_state_dict(ckpt["model"]); m.eval()
        return m, tok
    lora = (which == "lora")
    name = "ckpt_sft7_lora.pt" if lora else "ckpt_sft7_full.pt"
    ckpt = torch.load(HERE / name)
    cfg = GPTConfig(**ckpt["config"])
    m = GPT(cfg).to(DEVICE)
    if lora:
        inject_lora(m, r=ckpt.get("lora_r", 8))
    m.load_state_dict(ckpt["model"]); m.eval()
    return m, tok


@torch.no_grad()
def gen(model, tok, prompt, max_new=50):
    ids = tok.encode(prompt)
    start = len(ids)
    for _ in range(max_new):
        ctx = torch.tensor([ids[-model.cfg.max_len:]], dtype=torch.long,
                           device=DEVICE)
        nxt = int(model(ctx)[0, -1].argmax().item())
        if nxt == EOS_ID:
            break
        ids.append(nxt)
    return tok.decode(ids[start:]).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["full", "lora", "base"], default="full")
    ap.add_argument("--max-new-tokens", type=int, default=50)
    ap.add_argument("--raw", action="store_true", help="不加问答模板，纯续写")
    args = ap.parse_args()

    model, tok = load(args.model)
    label = {"full": "全量SFT", "lora": "LoRA SFT", "base": "基座"}[args.model]
    print(f"已加载 [{label}]（{DEVICE}）。输入问题看它怎么答；exit 退出。")

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
        prompt = line if args.raw else f"问：{line}\n答："
        print(f"答> {gen(model, tok, prompt, args.max_new_tokens)}")


if __name__ == "__main__":
    main()
