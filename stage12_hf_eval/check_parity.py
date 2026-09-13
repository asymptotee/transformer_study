"""check_parity.py —— 包装一致性自检(stage12 的守门员)

包装 HF 必须有零偏差证明(本项目传统:同构/一致性自检):
  1. **logits 逐位一致**:同一批 input_ids,原版 GPT 与 HF 包装版的前向
     输出应完全相同(fp16 权重下允许 0 差异:两边同一份权重、同一计算图)
  2. **分词一致**:HF tokenizer.encode == 原版 bpe.encode(逐 id),含
     控制符/空白/英文混排样本
  3. chat template 渲染结果 == 我们的 render 函数(格式契约)

任一不过,lm_eval 的数字就没有意义 —— 先修包装再跑评测。

用法(Spark):
  ~/llm_study/.venv/bin/python check_parity.py \
      --ckpt ../stage11_datascale/ckpt_11_4_chat.pt \
      --bpe ../stage11_datascale/cache_mm10g/bpe.json \
      --hf-dir hf_11_4_chat
"""

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from hf_wrap import (MiniGPTConfig, MiniGPTForCausalLM, OurBPETokenizer,  # noqa: E402
                     register, save_pretrained_from_ckpt)
sys.path.insert(0, str(HERE.parent / "stage9_modern_gpt"))
sys.path.insert(0, str(HERE.parent / "stage4_scaling_bpe"))
from bpe import BPETokenizer                                # noqa: E402
from model_modern import GPT, GPTConfig                     # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TEXTS = [
    "中国的首都是哪里？\n答：",
    "Hello world, this is a 中英混排 test 123.",
    "换行\n制表\t符\x00与空格 　全角",
    "机器学习是人工智能的一个分支。",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bpe", required=True)
    ap.add_argument("--hf-dir", required=True)
    args = ap.parse_args()

    # ---- 1. logits 逐位对比(fp32,同权重同图,应 0 差异)----
    ckpt = torch.load(args.ckpt, map_location="cpu")
    gc = ckpt["config"]
    raw = GPT(GPTConfig(**gc)).to(DEV)
    raw.load_state_dict(ckpt["model"])
    raw.eval()

    register()
    from transformers import AutoModelForCausalLM
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.hf_dir, dtype=torch.float32).to(DEV).eval()

    # 诊断 0:权重逐位相同 + tie 生效(不一致时先看这里)
    w_raw = ckpt["model"]["embedding.weight"]
    w_hf = hf_model.model.embedding.weight.detach().cpu()
    wdiff = (w_raw.float() - w_hf.float()).abs().max().item()
    tied = (hf_model.model.lm_head.weight.data_ptr()
            == hf_model.model.embedding.weight.data_ptr())
    print(f"[0] 权重 max-abs-diff: {wdiff:.3e} {'✓' if wdiff == 0 else '✗'} | "
          f"lm_head/embedding 绑定: {'✓' if tied else '✗'}")

    torch.manual_seed(0)
    ids = torch.randint(4, gc["vocab_size"], (2, 64), device=DEV)
    with torch.no_grad():
        a = raw(ids).float()
        b = hf_model(ids).logits.float()
    diff = (a - b).abs().max().item()
    print(f"[1] logits max-abs-diff(fp32): {diff:.3e}  "
          f"{'一致 ✓' if diff == 0 else '不一致 ✗'}")

    # ---- 1b. fp16 加载冒烟(不比较数值,验证半精度路径不崩)----
    hf16 = AutoModelForCausalLM.from_pretrained(
        args.hf_dir, torch_dtype=torch.float16).to(DEV).eval()
    with torch.no_grad():
        out = hf16(ids).logits
    ok = torch.isfinite(out).all().item()
    print(f"[1b] fp16 前向: 输出有限 {'✓' if ok else '✗'}(dtype={out.dtype})")

    # ---- 2. 分词一致 ----
    bpe = BPETokenizer.load(args.bpe)
    from transformers import AutoTokenizer
    hf_tok = AutoTokenizer.from_pretrained(args.hf_dir)
    bad = 0
    for t in TEXTS:
        if bpe.encode(t) != hf_tok.encode(t, add_special_tokens=False):
            bad += 1
            print(f"   分词不一致: {t[:20]!r}")
    print(f"[2] 分词对比 {len(TEXTS)} 条: {'全部一致 ✓' if bad == 0 else f'{bad} 条不一致 ✗'}")

    # ---- 3. chat template 契约 ----
    q = "中国的首都是哪里？"
    rendered = hf_tok.apply_chat_template([{"role": "user", "content": q}],
                                          tokenize=False,
                                          add_generation_prompt=True)
    expect = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    print(f"[3] chat template: {'与训练渲染一致 ✓' if rendered == expect else '不一致 ✗'}")
    if rendered != expect:
        print("    got:", repr(rendered))


if __name__ == "__main__":
    main()
