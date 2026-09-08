"""render_sft_chat.py —— 10.3 数据准备:官方 sft 语料 → 三段式 chat 样本

官方 sft_t2t_mini.jsonl 是 {"conversations":[{role,content}...]}。要让
**我们的模型**(自己的 BPE,无模板 token)学 chat 格式,先把每个对话用官方
tokenizer 的 chat_template 渲染成规范文本,再按"assistant 内容"边界拆三段:

  user_side = 模板开头 … <|im_start|>assistant\n        (掩码外)
  answer    = assistant 的 content                     (掩码内)
  tail      = <|im_end|> 及其后                        (掩码内,让模型学会收尾标记)

训练时 x/y = enc(user_side)+enc(answer)+enc(tail),loss 只盖 answer+tail
——与 minimind SFTDataset 的"assistant 段 + eos 段"掩码语义一致。

为机制演示取子集:只用单轮对(user+assistant 各一条)且 answer 不太长
(多轮与超长样本留给以后);渲染要调官方 tokenizer → 在 Spark 跑(有
minimind 仓库 + transformers)。

用法(Spark):
  ~/llm_study/.venv/bin/python render_sft_chat.py --n 40000 \
      --out chat_train.jsonl --out-test chat_test.jsonl
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

MM = Path.home() / "llm_study" / "minimind"
sys.path.insert(0, str(MM))
from transformers import AutoTokenizer                    # noqa: E402

MARKER = "<|im_start|>assistant\n"
THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def split_sample(tok, convs, strip_think=False):
    """渲染单轮对并拆三段;不合规返回 None。

    strip_think:官方训练对空 think 有 80% 剥除(见其 lm_dataset 的
    post_processing_chat);我们 40k 全抽到 think 风格区块后模型学到
    "先空想再绕圈"。去噪版把 think 块整体剥掉,只留实质回答。
    """
    if len(convs) != 2 or convs[0]["role"] != "user" or convs[1]["role"] != "assistant":
        return None
    text = tok.apply_chat_template(convs, tokenize=False, add_generation_prompt=False)
    i = text.find(MARKER)
    if i < 0:
        return None
    s = i + len(MARKER)
    e = text.find("<|im_end|>", s)
    if e < 0:
        return None
    answer = text[s:e]
    if strip_think:
        answer = THINK_RE.sub("", answer).strip()
        if not answer:
            return None
    return {"user_side": text[:s], "answer": answer, "tail": text[e:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(Path.home() / "llm_study" / "mm_data" /
                                         "sft_t2t_mini.jsonl"))
    ap.add_argument("--n", type=int, default=40000, help="训练样本目标数")
    ap.add_argument("--max-answer-chars", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strip-think", action="store_true",
                    help="剥掉回答里的 think 块(去噪消融版)")
    ap.add_argument("--out", default="chat_train.jsonl")
    ap.add_argument("--out-test", default="chat_test.jsonl")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MM / "model")
    random.seed(args.seed)
    out_tr, out_te = [], []
    with open(args.src, encoding="utf-8") as f:
        for ln in f:
            convs = json.loads(ln)["conversations"]
            s = split_sample(tok, convs, strip_think=args.strip_think)
            if s is None or len(s["answer"]) > args.max_answer_chars:
                continue
            (out_te if random.random() < 0.02 else out_tr).append(s)
            if len(out_tr) >= args.n:
                break
    # 测试集固定 500
    out_te = out_te[:500]
    for path, rows in ((args.out, out_tr), (args.out_test, out_te)):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"训练 {len(out_tr)} | 测试 {len(out_te)}", flush=True)
    ex = out_tr[0]
    print("样例 user_side:", ex["user_side"][-40:].replace("\n", "⏎"))
    print("样例 answer   :", ex["answer"][:50].replace("\n", "⏎"))
    print("样例 tail     :", ex["tail"].replace("\n", "⏎"))


if __name__ == "__main__":
    main()
