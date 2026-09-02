"""compare.py —— 对比基座 / 全量SFT / LoRA SFT 三个模型

展示两件事：
  1. 行为转变：基座模型只会"续写"，SFT 后学会"应答"（格式），全量与 LoRA 几乎一样。
  2. 知识边界：SFT 只教格式、不注入知识——问语料外的问题，模型答不出（还是鲁迅腔）。

运行：
  python compare.py
"""

import json
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


def load_base():
    ckpt = torch.load(HERE.parent / "stage5_capstone" / "ckpt_best.pt")
    cfg = GPTConfig(**ckpt["config"])
    m = GPT(cfg).to(DEVICE); m.load_state_dict(ckpt["model"]); m.eval()
    tok = BPETokenizer.load(HERE.parent / "stage5_capstone" / "bpe_best.json")
    return m, tok, cfg


def load_sft(name, lora, r=8):
    ckpt = torch.load(HERE / name)
    cfg = GPTConfig(**ckpt["config"])
    m = GPT(cfg).to(DEVICE)
    if lora:
        inject_lora(m, r=r)
    m.load_state_dict(ckpt["model"]); m.eval()
    return m


@torch.no_grad()
def gen(model, tok, prompt, max_new=50):
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
    base, tok, cfg = load_base()
    full = load_sft("ckpt_sft_full.pt", lora=False)
    lora = load_sft("ckpt_sft_lora.pt", lora=True, r=8)

    n_base = sum(p.numel() for p in base.parameters())
    n_lora_train = 32768
    print(f"参数量: 基座/全量SFT = {n_base:,} | LoRA SFT 可训练 = {n_lora_train:,} "
          f"({n_lora_train/n_base*100:.2f}%)\n")

    # ---- 1. 行为转变：同一提示，三个模型的反应 ----
    print("=" * 70)
    print("① 行为转变：给一个'问：…答：'提示，看三个模型怎么回应")
    print("=" * 70)
    prompts = [ex["prompt"] for ex in
               (json.loads(l) for l in open(HERE / "qa_test.jsonl", encoding="utf-8"))][:3]
    for p in prompts:
        q = p.splitlines()[0]
        print(f"\n提示: {q}")
        print(f"  [基座]    {gen(base, tok, p)!r}")
        print(f"  [全量SFT] {gen(full, tok, p)!r}")
        print(f"  [LoRA]    {gen(lora, tok, p)!r}")

    # ---- 2. 知识边界：问语料外的问题 ----
    print("\n" + "=" * 70)
    print("② 知识边界：问一个鲁迅语料之外的问题（SFT 不注入知识）")
    print("=" * 70)
    ooc = ['问：中国的首都是哪里？\n答：',
           '问：水的化学式是什么？\n答：']
    for p in ooc:
        print(f"\n提示: {p.splitlines()[0]}")
        print(f"  [全量SFT] {gen(full, tok, p)!r}")
        print(f"  [LoRA]    {gen(lora, tok, p)!r}")
    print("\n（预期：答不出'北京/H2O'，只会给鲁迅风格的句子——知识没被注入）")


if __name__ == "__main__":
    main()
