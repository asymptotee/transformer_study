"""bpe.py —— 字节对编码（Byte-Pair Encoding）分词器

阶段四的第二个杠杆。为什么需要 BPE：
  字符级把语义打散了——"逻辑"是两个无关 token，模型要从零学它们的关联；
  序列也太长，上下文窗口装的内容少。
  BPE 从字符出发，反复合并语料里**最高频的相邻对**，自动学出一套"子词"词表：
  高频组合（"我们""不知道""逻辑"）会合并成单个 token。于是
    · 序列更短（同样的窗口看到更多内容）
    · 语义更聚（常见词作为一个整体）
  这是真实 LLM 的标准分词方式（GPT 系列都用 BPE 或其变体）。

实现：纯 Python 学合并规则（统计相邻对频率→合并最高频对→迭代），含 encode/decode。
"""

import json
from collections import Counter
from pathlib import Path

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3


def _apply_merge(tokens, pair, new):
    """把 tokens 里所有相邻的 pair=(a,b) 替换成 new。"""
    out, i, a, b = [], 0, pair[0], pair[1]
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
            out.append(new); i += 2
        else:
            out.append(tokens[i]); i += 1
    return out


class BPETokenizer:
    def __init__(self, itos, merges):
        self.itos = itos
        self.stoi = {t: i for i, t in enumerate(itos)}
        self.merges = merges          # [( (a,b), new ), ...] 按学习顺序

    @classmethod
    def train(cls, text, num_merges=600, sample_chars=60000, verbose=True):
        """在一段样本上学合并规则（用样本是为了快，规则可推广到全文）。"""
        sample = text[:sample_chars]
        seqs = [list(line) for line in sample.split("\n") if line]
        merges = []
        for step in range(num_merges):
            pairs = Counter()
            for seq in seqs:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            pair = max(pairs, key=pairs.get)
            new = pair[0] + pair[1]
            merges.append((pair, new))
            seqs = [_apply_merge(seq, pair, new) for seq in seqs]
            if verbose and (step + 1) % 100 == 0:
                print(f"  merge {step + 1}/{num_merges}: {pair} -> '{new}'  (出现 {pairs[pair]} 次)")
        chars = sorted({c for c in text})   # 字符词表覆盖全文（避免生僻字变 UNK）
        itos = SPECIALS + chars + [new for _, new in merges]
        return cls(itos, merges)

    def __len__(self):
        return len(self.itos)

    def encode(self, text):
        """字符序列 → 依次应用所有合并 → 转 id。"""
        tokens = list(text)
        for pair, new in self.merges:
            tokens = _apply_merge(tokens, pair, new)
        return [self.stoi.get(t, UNK_ID) for t in tokens]

    def decode(self, ids):
        out = []
        for i in ids:
            if i == EOS_ID:
                break
            if i >= len(SPECIALS):
                out.append(self.itos[i])
        return "".join(out)

    def save(self, path):
        Path(path).write_text(
            json.dumps({"itos": self.itos, "merges": self.merges}, ensure_ascii=False),
            encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        merges = [(tuple(m[0]), m[1]) for m in data["merges"]]
        return cls(data["itos"], merges)


if __name__ == "__main__":
    # 自测：在鲁迅语料上学 BPE，看序列缩短效果
    text = (Path(__file__).parent.parent / "stage3_gpt" / "luxun_clean.txt").read_text(encoding="utf-8")
    tok = BPETokenizer.train(text, num_merges=600)
    print(f"\n词表: {len(tok)} (4 特殊 + 字符 + 600 合并)")
    sample = "我冒了严寒，回到相隔二千余里的故乡去。"
    ids = tok.encode(sample)
    print(f"\n原文 {len(sample)} 字符: {sample}")
    print(f"BPE  {len(ids)} token:  {[tok.itos[i] for i in ids]}")
    print(f"压缩率: {len(sample)/len(ids):.2f} 字符/token")
    assert tok.decode(ids) == sample, "encode/decode 往返不一致！"
    print("encode/decode 往返一致 ✓")
