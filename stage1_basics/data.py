"""字符级分词器与 seq2seq 数据集

两个学习任务共用这套数据管道，只通过 make_pairs() 的切分方式区分：
- reverse：  (整行文本, 反转后的文本)      —— 验证架构正确性的经典基准
- complete：(行的前半段, 行的后半段)       —— 真正的文本续写/生成
"""

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

# 特殊 token：放在词表最前面，索引固定
PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3


class CharTokenizer:
    """字符级分词器：一个汉字（或字母）就是一个 token

    词表结构：前 4 位是特殊 token，其余是语料中出现过的所有字符（排序后固定顺序）。
    字符级的好处是词表极小（几百个），不需要任何外部分词工具。
    """

    def __init__(self, lines):
        chars = sorted({ch for line in lines for ch in line})
        self.itos = SPECIALS + chars          # id -> 字符
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}  # 字符 -> id

    @classmethod
    def from_vocab(cls, itos):
        """从 checkpoint 里保存的词表恢复，推理时用。"""
        obj = cls.__new__(cls)
        obj.itos = list(itos)
        obj.stoi = {ch: i for i, ch in enumerate(obj.itos)}
        return obj

    def __len__(self):
        return len(self.itos)

    def encode(self, s):
        """字符串 -> token id 列表，未登录字符映射到 <unk>"""
        return [self.stoi.get(ch, UNK_ID) for ch in s]

    def decode(self, ids):
        """token id -> 字符串：遇到 <eos> 截断，跳过其余特殊 token"""
        out = []
        for i in ids:
            if i == EOS_ID:
                break
            if i >= len(SPECIALS):
                out.append(self.itos[i])
        return "".join(out)


def load_corpus(path):
    """读语料：一行一个样本，跳过空行。"""
    return [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]


def make_pairs(lines, task):
    """按任务把每行文本切成 (src, tgt) 文本对。"""
    pairs = []
    for line in lines:
        if task == "reverse":
            # 反转任务：源和目标等长，注意力必须学会"远距离对齐"
            pairs.append((line, line[::-1]))
        elif task == "complete":
            # 续写任务：从中点切开，decoder 只负责生成后半段
            mid = len(line) // 2
            pairs.append((line[:mid], line[mid:]))
        else:
            raise ValueError(f"未知任务: {task}（支持 reverse / complete）")
    return pairs


class Seq2SeqDataset(Dataset):
    """把文本对编码成张量。

    目标序列两端加特殊符号：<bos> 开头（decoder 的起始输入）、
    <eos> 结尾（让模型学会"何时停笔"）。
    """

    def __init__(self, pairs, tokenizer, max_len):
        self.samples = []
        for src, tgt in pairs:
            if not src or not tgt or max(len(src), len(tgt)) + 2 > max_len:
                continue  # 跳过空样本和超长样本（+2 是给 bos/eos 留的位置）
            src_ids = tokenizer.encode(src)
            tgt_ids = [BOS_ID] + tokenizer.encode(tgt) + [EOS_ID]
            self.samples.append((torch.tensor(src_ids), torch.tensor(tgt_ids)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate(batch):
    """把一个 batch 里长短不一的序列动态填充到该 batch 内的最大长度。

    比固定长度填充省算力；<pad> 位置会在模型里被填充掩码屏蔽、
    在损失函数里被 ignore_index 忽略。
    """
    srcs, tgts = zip(*batch)
    src = torch.nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=PAD_ID)
    tgt = torch.nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=PAD_ID)
    return src, tgt


def split_train_val(lines, val_ratio=0.1, seed=42):
    """固定随机种子切分训练/验证集，保证结果可复现。"""
    lines = lines[:]
    random.Random(seed).shuffle(lines)
    n_val = max(1, int(len(lines) * val_ratio))
    return lines[n_val:], lines[:n_val]


def random_lines(chars, n, min_len=3, max_len=16, seed=0):
    """从字符集随机生成字符串，作为 reverse 任务的训练数据补充。

    反转是算法任务：只有几百行真实语料时，模型会直接背下所有样本，
    学不到"第 i 个输出对应第 L-1-i 个输入"的通用对齐规则，验证集上自然全军覆没。
    随机字符串的空间是天文数字，背不下来，模型被迫学会用注意力做位置对齐 ——
    这是形式语言基准（formal language benchmark）的标准做法。
    验证集不加合成数据，保持真实语料，才是诚实的泛化测试。
    """
    rng = random.Random(seed)
    return [
        "".join(rng.choice(chars) for _ in range(rng.randint(min_len, max_len)))
        for _ in range(n)
    ]
