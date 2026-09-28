"""test_holmes_arith.py —— 把 Holmes v1 的算术能力钉死(n=60,不是 n=1)

**为什么要多测**:前面只测了 2 道算术题(47+86 → 101 错、应用题 → 24 错),
n=1 说明不了"偶尔错"还是"系统性错"。这个量到 60 道。

**两个关键设计(因为 Holmes 停不下来)**:

  1. **只取输出里第一个整数当答案**。它会无限复读同一个答案,如果按"输出里
     有没有出现正确答案"判,一旦它某次碰巧复读到正确答案就会误判成会做。
     取第一个数才是它的真实"第一反应"。
  2. **max_new_tokens 给 96 就够** —— 第一个答案在第 10 个 token 内就出来了,
     剩下的全是复读。给多了只是浪费时间。

**难度阶梯**:1 位 / 2 位 / 3 位各 20 道,和 stage16 我们自己的 126M 同口径
(虽然格式不同 —— 我们的是逐位空格,它的是自然文本)。

用法(Spark):
  PYTHONPATH=/home/zhangxu/llm_study/tf450 ~/llm_study/.venv/bin/python test_holmes_arith.py
"""

import random
import re
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = Path.home() / "llm_study" / "holmes_v1" / "model"
N_PER_RUNG = 20
INT_RE = re.compile(r"-?\d+")
qf = "计算 {a} + {b} 等于多少？"    # 与其 chat_demo 里例题的问法一致


def degenerate(txt):
    """复读检测:最高频字符占比 > 60%(它的复读很严重,阈值可以高一点)。"""
    if len(txt) < 40:
        return False
    c, n = Counter(txt).most_common(1)[0]
    return n / len(txt) > 0.6


def main():
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), torch_dtype="auto", trust_remote_code=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev).eval()

    rng = random.Random(42)
    rungs = []
    for d in (1, 2, 3):
        lo, hi = (0, 9) if d == 1 else (10 ** (d - 1), 10 ** d - 1)
        rungs.append((d, [(rng.randint(lo, hi), rng.randint(lo, hi))
                          for _ in range(N_PER_RUNG)]))

    print(f"=== Holmes v1 | 算术 n={N_PER_RUNG}×3 ===", flush=True)
    print("  %-6s %10s %12s %10s" % ("位数", "答对", "没给出数", "复读"))
    tot = [0, 0, 0, 0]
    samples = []
    for d, pairs in rungs:
        ok = nodigit = degen = 0
        for a, b in pairs:
            q = qf.format(a=a, b=b)      # 用问句格式 —— 裸算式对它不利,见文件头
            txt = tok.apply_chat_template([{"role": "user", "content": q}],
                                          tokenize=False, add_generation_prompt=True)
            ids = tok(txt, return_tensors="pt").input_ids.to(dev)
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=96, do_sample=False,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id,
                                     eos_token_id=tok.eos_token_id)
            ans = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
            m = INT_RE.search(ans)          # ★ 只取**第一个**整数
            got = int(m.group()) if m else None
            if got is None:
                nodigit += 1
            elif got == a + b:
                ok += 1
            degen += degenerate(ans)
            if len(samples) < 4:
                samples.append((f"{a}+{b}", a + b, ans[:80].replace("\n", " ")))
        n = len(pairs)
        tot = [tot[0] + ok, tot[1] + nodigit, tot[2] + degen, tot[3] + n]
        print("  %-6d %7d/%-3d %9d/%-3d %7d/%-3d"
              % (d, ok, n, nodigit, n, degen, n), flush=True)
    print("  %-6s %7d/%-3d %9d/%-3d %7d/%-3d"
          % ("合计", tot[0], tot[3], tot[1], tot[3], tot[2], tot[3]))

    print("\n  样例(题 | 正确 | 它的输出开头):")
    for q, want, got in samples:
        print("    %-10s %-6s %r" % (q, want, got))


if __name__ == "__main__":
    main()
