"""make_qa.py —— 从维基语料构造问答数据（阶段八：知识问答 SFT）

设计思路（和阶段六 make_qa.py 同源，但语料从鲁迅换成了 150MB 维基百科）：
  训练数据 = 补全式问答对。从文章里抽出含"是"的句子，在逗号处切成两半：
      问：请补全这句话："北京是"   答：中华人民共和国的首都，简称京。
  模型学的是"如何用百科知识补全句子"——和预训练分布贴近，数据量大、干净。

  评估数据 = 真实问答（eval_facts.py 里手工定义 20 题）。
  关键验证：只用"补全式"训练的模型，能否泛化到"X 是哪里？"这种真问句？
  —— 知识来自预训练（基座已经"知道"北京=首都），SFT 只教"回答的姿势"。

输出：qa_train.jsonl / qa_test.jsonl（与阶段六同格式）
"""

import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).parent
CORPUS = HERE.parent / "stage7_gpu_scale" / "corpus_wiki.jsonl"

# 三种问法模板（增加多样性，仿阶段六）
TEMPLATES = [
    "问：请补全这句话：“{x}”\n答：",
    "问：续写维基百科中的句子：“{x}”\n答：",
    "问：“{x}”的下一句是？\n答：",
]

MIN_X, MIN_Y = 4, 6          # 前后两半的最短长度
MAX_LEN = 96                 # 超过就跳过（装不进 block）


def split_sentence(sent):
    """在逗号/顿号/分号处把句子切成 (x, y)，要求两半都不太短。"""
    for ch in "，；、":
        if ch in sent:
            i = sent.index(ch)
            x, y = sent[:i].strip(), sent[i + 1:].strip()
            if len(x) >= MIN_X and len(y) >= MIN_Y and len(x) + len(y) <= MAX_LEN:
                return x, y
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=60000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    pairs = []
    with open(CORPUS, encoding="utf-8") as f:
        for line in f:
            text = json.loads(line)["text"]
            for sent in text.replace("\n", "").split("。"):
                if "是" not in sent:
                    continue
                cut = split_sentence(sent)
                if cut:
                    pairs.append(cut)

    random.shuffle(pairs)
    train_pairs = pairs[:args.n_train]
    test_pairs = pairs[args.n_test:args.n_test * 2]

    def write(name, data):
        with open(HERE / name, "w", encoding="utf-8") as out:
            for x, y in data:
                prompt = random.choice(TEMPLATES).format(x=x)
                out.write(json.dumps({"prompt": prompt, "answer": y},
                                     ensure_ascii=False) + "\n")
        print(f"  {name}: {len(data)} 条")

    print(f"共切出 {len(pairs):,} 个可用的句对")
    write("qa_train.jsonl", train_pairs)
    write("qa_test.jsonl", test_pairs)

    # 展示几条
    print("\n样例:")
    with open(HERE / "qa_train.jsonl", encoding="utf-8") as f:
        for _ in range(3):
            ex = json.loads(f.readline())
            print(f"  {ex['prompt']!r}")
            print(f"    答: {ex['answer']!r}")


if __name__ == "__main__":
    main()
