"""test_holmes.py —— 在 Spark 上跑 Holmes v1(0.5B 稠密,150B token 的 chat 版)

**为什么跑它**:它就是我们前面讨论的那个"专注推理、原生思维链、舍弃知识"的参照实现。
关键是看**它的效果到底怎样** —— 因为:

  · 没找到公开的 benchmark 数字(知乎专栏要登录,HF 被墙)
  · 而它的"效果"恰恰是我们要判断"这条路值不值得走"的依据

**四道题,每题针对一个我们自己的已知边界:**

  ① 他们自己的例题(chat_demo.py 里注释的那道)—— 多步算术应用题,答案 31
  ② 一道 2 位数加法 —— **我们的 126M 是 0%**
  ③ 一道三段论 —— 我们语料里有这类数据,但没测过基座
  ④ 一个通用知识问题 —— 看它"舍弃知识"舍弃到什么程度

⚠️ 我们的 126M 在 ① 上是**必然失败**的(不会算 + 不会处理多步)。所以这组题
   同时是对 Holmes 的测试和对我们的标尺。

用法(Spark):
  ~/llm_study/.venv/bin/python test_holmes.py
"""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = Path.home() / "llm_study" / "holmes_v1" / "model"

QUESTIONS = [
    ("① 多步算术应用题(他们的例题,答案 31)",
     "小明去文具店买学习用品，一支钢笔售价 12 元，一本笔记本售价 5 元。"
     "他买了 2 支钢笔和 3 本笔记本，付款时店员给他减免了 3 元。"
     "请问小明最终需要支付多少钱？"),
    ("② 2 位数加法(我们 126M = 0%)",
     "计算 47 + 86 等于多少？"),
    ("③ 三段论逻辑",
     "所有的狗都喜欢吃肉，这只猫喜欢吃肉，所以这只猫是一只狗。这个推理正确吗？为什么？"),
    ("④ 通用知识(看它舍弃到什么程度)",
     "地球绕着什么转？"),
]


def main():
    print(f"加载 {MODEL_DIR} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), torch_dtype="auto", trust_remote_code=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"参数 {n/1e6:.1f}M | 词表 {len(tok)} | 设备 {dev}", flush=True)

    for title, q in QUESTIONS:
        msgs = [{"role": "user", "content": q}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=512, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id,
                                 eos_token_id=tok.eos_token_id)
        ans = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        print("\n" + "=" * 72)
        print(title)
        print("问:" + q)
        print("-" * 72)
        print(ans[:1400] + ("…" if len(ans) > 1400 else ""), flush=True)


if __name__ == "__main__":
    main()
