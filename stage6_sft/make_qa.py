"""make_qa.py —— 从鲁迅语料构造"补全式"问答数据集（用于 SFT）

为什么这样造数据：
  基座模型（阶段五）只在鲁迅语料上预训练过，它"知道"的只有鲁迅文本。
  所以我们造它**答得出**的题——补全语料里的句子。这样 SFT 后能看到：
    · 模型学会了"应答格式"（看到"问：…答："就给出答案，而不是接着乱续）
    · 但知识仍来自背下来的语料（答不出语料外的东西）
  正好演示 CONCEPTS 第九节："SFT 激发行为，不注入知识"。

格式（SFT 会只对"答："之后的部分算 loss）：
    问：请补全这句话："我冒了严寒，"
    答：回到相隔二千余里，别了二十余年的故乡去。

输出 qa_train.jsonl / qa_test.jsonl，每行 {"prompt":..., "answer":...}
"""

import json
import random
import re
from pathlib import Path

HERE = Path(__file__).parent
CORPUS = HERE.parent / "stage3_gpt" / "luxun_clean.txt"

# 几种问法模板，增加多样性，让模型学会"格式"而非死记某一种措辞
TEMPLATES = [
    '问：请补全这句话：“{h}”\n答：',
    '问：“{h}”的下一句是？\n答：',
    '问：续写鲁迅的句子：“{h}”\n答：',
]


def clean(line):
    """去掉行首全角空格和电子书残留的章节标记。"""
    line = line.strip().lstrip("　").strip()
    return line


def split_sentence(s):
    """把句子拆成 (前半, 后半)。优先在逗号处切，否则中点切。"""
    s = s.rstrip("。！？…")  # 去掉句末标点，答案里再补
    idx = s.find("，")
    if 4 <= idx <= len(s) - 5:           # 逗号位置合适，两半都够长
        return s[:idx + 1], s[idx + 1:]
    if len(s) >= 12:                      # 没合适逗号，按中点切
        mid = len(s) // 2
        return s[:mid], s[mid:]
    return None, None


def main():
    rng = random.Random(7)
    text = CORPUS.read_text(encoding="utf-8")

    # 按句末标点 + 换行切句
    raw = re.split(r"[。！？\n]+", text)
    examples = []
    for chunk in raw:
        s = clean(chunk)
        if not s or re.match(r"第\d+章", s):   # 跳过章节标题
            continue
        h, t = split_sentence(s)
        if not h or len(t) < 4 or len(h) > 30:
            continue
        prompt = rng.choice(TEMPLATES).format(h=h)
        answer = t + "。"
        examples.append({"prompt": prompt, "answer": answer})

    rng.shuffle(examples)
    n_test = 400
    test = examples[:n_test]
    train = examples[n_test:]

    for name, data in [("qa_train.jsonl", train), ("qa_test.jsonl", test)]:
        with open(HERE / name, "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"构造问答对: 训练 {len(train)} | 测试 {len(test)}")
    print("\n样例:")
    for ex in train[:4]:
        print(f"  {ex['prompt']!r}")
        print(f"    → {ex['answer']!r}")


if __name__ == "__main__":
    main()
