"""eval_transfer.py —— 11.1 迁移探针(修正版):11-mm 在 wiki 域文本上的 loss

supervisor 里最初的探针有设计硬伤:拿旧分词器的 token 直接喂新词表模型
(跨分词器不可比,CE 维度直接崩)。正确做法:**同一份 wiki 文本,用新 BPE
重新编码**,再量 11-mm 的 loss——"换语料训练后,wiki 域的文本还预测得动吗"。

读数方式(诚实边界):
  · 与 11-mm 语料内 val(2.267)比:差 = 换域的代价(期望明显更高)
  · 与 94_full 的 3.776 不直接比(分词器不同);若想同词表比,需要另训一个
    wiki×新BPE 的模型——那是下一步的事,本探针先给绝对值
  · 顺带用固定 wiki prompt 生成几条,看"wiki 腔"还在不在(定性)

用法(Spark):
  ~/llm_study/.venv/bin/python eval_transfer.py
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
STAGE7 = HERE.parent / "stage7_gpu_scale"
STAGE9 = HERE.parent / "stage9_modern_gpt"
sys.path.insert(0, str(HERE.parent / "stage4_scaling_bpe"))
sys.path.insert(0, str(STAGE9))
from bpe import BPETokenizer, EOS_ID                        # noqa: E402
from model_modern import GPT, GPTConfig                     # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK, BATCH, NB = 256, 32, 16


@torch.no_grad()
def val_loss(model, ids, tok):
    tot, n = 0.0, 0
    for _ in range(NB):
        ix = torch.randint(len(ids) - BLOCK - 1, (BATCH,), device=DEVICE)
        x = torch.stack([ids[i:i + BLOCK] for i in ix])
        y = torch.stack([ids[i + 1:i + 1 + BLOCK] for i in ix])
        loss = F.cross_entropy(model(x).reshape(-1, len(tok)),
                               y.reshape(-1), ignore_index=0)
        tot += loss.item() * x.numel()
        n += x.numel()
    return tot / n


def main():
    tok = BPETokenizer.load(HERE / "cache_mm" / "bpe.json")
    ckpt = torch.load(HERE / "ckpt_11_1_mm.pt")
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"11-mm | vocab {len(tok)} | {sum(p.numel() for p in model.parameters())/1e6:.1f}M",
          flush=True)

    # wiki 文本(前 1000 篇 ≈ 8M 字符,编码 ~5 分钟)用新 BPE 编码
    texts = []
    with open(STAGE7 / "corpus_wiki.jsonl", encoding="utf-8") as f:
        for i, ln in enumerate(f):
            if i >= 1000:
                break
            texts.append(json.loads(ln)["text"])
    ids = []
    for t in texts:
        ids.extend(tok.encode(t))
        ids.append(EOS_ID)
    ids = torch.tensor(ids, device=DEVICE)
    print(f"wiki 前 2000 篇 → {len(ids):,} token(新 BPE)", flush=True)

    torch.manual_seed(0)
    vl = val_loss(model, ids, tok)
    import math
    print(f"11-mm 在 wiki 域(wiki文本×新BPE): {vl:.3f}(ppl {math.exp(vl):.1f})")
    print(f"对照: 11-mm 语料内 val 2.267(ppl 9.7) | 94_full 在 wiki×旧BPE 3.776(不同词表,仅供量级参照)")

    print("\nwiki prompt 续写(贪心):")
    for p in ["北京是", "长城位于", "数学是", "鲁迅是"]:
        pids = tok.encode(p)
        start = len(pids)
        for _ in range(40):
            ctx = torch.tensor([pids[-512:]], device=DEVICE)
            nxt = int(model(ctx)[0, -1].argmax().item())
            if nxt == EOS_ID:
                break
            pids.append(nxt)
        print(f"  【{p}】{tok.decode(pids[start:])[:60]}")


if __name__ == "__main__":
    main()
