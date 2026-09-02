"""chat.py —— 交互式问答测试：输入句子前半，SFT 模型即时补全

像阶段三 generate_text.py 那样的交互体验，但针对阶段六的问答场景：
你输入一句话的前半，模型按 SFT 学到的格式给出"答案"。

用法：
  PY=python
  $PY chat.py                     # 默认用全量 SFT
  $PY chat.py --model lora        # 用 LoRA SFT
  $PY chat.py --model base        # 用基座（对比：它只会乱续，不懂应答）
  $PY chat.py --raw               # 不套问答模板，纯续写你的输入

进入后输入句子前半回车即可；输入 exit / quit 或 Ctrl-D 退出。
"""

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "stage3_gpt"))
sys.path.insert(0, str(HERE.parent / "stage4_scaling_bpe"))
from bpe import BPETokenizer, EOS_ID
from model_gpt import GPT, GPTConfig
from lora import inject_lora

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load(model_choice):
    tok = BPETokenizer.load(HERE.parent / "stage5_capstone" / "bpe_best.json")
    if model_choice == "base":
        ckpt = torch.load(HERE.parent / "stage5_capstone" / "ckpt_best.pt")
        cfg = GPTConfig(**ckpt["config"])
        m = GPT(cfg).to(DEVICE); m.load_state_dict(ckpt["model"])
    else:
        lora = (model_choice == "lora")
        name = "ckpt_sft_lora.pt" if lora else "ckpt_sft_full.pt"
        ckpt = torch.load(HERE / name)
        cfg = GPTConfig(**ckpt["config"])
        m = GPT(cfg).to(DEVICE)
        if lora:
            inject_lora(m, r=ckpt.get("lora_r", 8))
        m.load_state_dict(ckpt["model"])
    m.eval()
    return m, tok


@torch.no_grad()
def gen(model, tok, prompt, max_new=60):
    """贪心生成直到 EOS，返回新生成的文本。"""
    ids = tok.encode(prompt)
    start = len(ids)
    for _ in range(max_new):
        ctx = torch.tensor([ids[-model.cfg.max_len:]], dtype=torch.long, device=DEVICE)
        nxt = int(model(ctx)[0, -1].argmax().item())
        if nxt == EOS_ID:
            break
        ids.append(nxt)
    return tok.decode(ids[start:]).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["full", "lora", "base"], default="full")
    ap.add_argument("--max-new-tokens", type=int, default=60)
    ap.add_argument("--raw", action="store_true", help="不套问答模板，直接续写输入")
    args = ap.parse_args()

    print(f"设备: {DEVICE}")
    model, tok = load(args.model)
    label = {"full": "全量SFT", "lora": "LoRA SFT", "base": "基座"}[args.model]
    print(f"已加载 [{label}]。输入句子前半看它怎么答；exit 退出。")
    if not args.raw:
        print("（自动套模板：问：请补全这句话：“…” 答：）")

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
        prompt = line if args.raw else f'问：请补全这句话：“{line}”\n答：'
        print(f"答> {gen(model, tok, prompt, args.max_new_tokens)}")


if __name__ == "__main__":
    main()
