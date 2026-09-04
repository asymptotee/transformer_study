"""check_isomorphic.py —— 零件换代"同构自检"(9.1 引入,9.2/9.3 每次必跑)

model_modern.py 每次换/加零件后,先证明一件事:
**在默认(旧)开关下,新模块与 stage3 model_gpt 在同一份权重上逐位同构。**
做不到这一点,A/B 里"新代码引入的意外差异"会和"零件本身的差异"混在一起,
实验就不干净了(9.1 的零噪声验证就是本脚本的前身)。

方法:同一进程里加载两个模块(同一份 ckpt),在**同一批固定窗口**上
比较 CE loss 与 logits——不是两次独立运行的统计比较,是逐位比较:

  同窗口零噪声:任何差异都只能来自代码路径,与采样噪声无关。

用法(Spark):
  ~/llm_study/.venv/bin/python check_isomorphic.py \
      --ckpt ../stage7_gpu_scale/ckpt_large.pt

预期输出:
  loss: old=... new=... Δ=+0.000000     (差到 6 位小数)
  logits max-abs-diff: 0.000e+00        (逐位一致)

一旦默认开关转正(9.4 合体后旧路径退役),本脚本使命结束;在那之前,
每个零件落地时它都是 A/B 可信度的守门员。
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "stage3_gpt"))   # model_gpt(旧参照)
sys.path.insert(0, str(REPO / "stage4_scaling_bpe"))
sys.path.insert(0, str(HERE))                  # model_modern(新模块)
from bpe import BPETokenizer           # noqa: E402
from model_gpt import GPT as GPT_OLD, GPTConfig as CfgOLD   # noqa: E402
from model_modern import GPT as GPT_NEW, GPTConfig as CfgNEW  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(REPO / "stage7_gpu_scale" / "ckpt_large.pt"))
    ap.add_argument("--tokens", default=str(REPO / "stage7_gpu_scale" / "cache" / "tokens.pt"))
    ap.add_argument("--bpe", default=str(REPO / "stage7_gpu_scale" / "cache" / "bpe.json"))
    ap.add_argument("--blocks", type=int, default=8, help="每段窗口数(同构自检 8 个足够)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=256)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    tok = BPETokenizer.load(args.bpe)
    m_old = GPT_OLD(CfgOLD(**ckpt["config"])).to(DEVICE)
    m_old.load_state_dict(ckpt["model"]); m_old.eval()
    m_new = GPT_NEW(CfgNEW(**ckpt["config"])).to(DEVICE)
    m_new.load_state_dict(ckpt["model"]); m_new.eval()

    torch.manual_seed(0)                 # 固定窗口:与采样无关
    ids = torch.load(args.tokens)
    val_ids = ids[-int(len(ids) * 0.05):].to(DEVICE)
    batches = []
    for _ in range(4):
        ix = torch.randint(len(val_ids) - args.block_size - 1,
                           (args.blocks,), device=DEVICE)
        x = torch.stack([val_ids[i:i + args.block_size] for i in ix])
        y = torch.stack([val_ids[i + 1:i + 1 + args.block_size] for i in ix])
        batches.append((x, y))

    losses, ref_logits = {}, None
    with torch.no_grad():
        for name, m in (("old", m_old), ("new", m_new)):
            tot, n = 0.0, 0
            for x, y in batches:
                logits = m(x)
                if name == "old":
                    ref_logits = logits
                tot += F.cross_entropy(logits.reshape(-1, len(tok)),
                                       y.reshape(-1), ignore_index=0).item() * x.numel()
                n += x.numel()
            losses[name] = tot / n
        maxdiff = (ref_logits - m_new(batches[0][0])).abs().max().item()

    print(f"loss: old={losses['old']:.6f} new={losses['new']:.6f} "
          f"Δ={losses['new'] - losses['old']:+.6f}")
    print(f"logits max-abs-diff(同一 batch): {maxdiff:.3e}")
    ok = losses["new"] == losses["old"] and maxdiff == 0.0
    print("同构 ✓" if ok else "同构 ✗ 不一致——先查新模块的默认路径,再做 A/B")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
