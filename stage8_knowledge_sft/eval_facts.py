"""eval_facts.py —— 20 道常识题统计答对率（阶段八核心评估）

验证：SFT 后模型能否用"预训练注入的知识"回答真问题？
  高频 10 题（语料里共现几百次，预期答对一部分）
  低频 10 题（语料里有、但问法不常见，预期答不对——知识边界）

对比三个模型：基座（只会续写）/ 全量SFT / LoRA SFT。

用法（DGX Spark）：
  python -u eval_facts.py
"""

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
STAGE7 = HERE.parent / "stage7_gpu_scale"
sys.path.insert(0, str(HERE.parent / "stage3_gpt"))
sys.path.insert(0, str(HERE.parent / "stage4_scaling_bpe"))
from bpe import BPETokenizer, EOS_ID
from model_gpt import GPT, GPTConfig
sys.path.insert(0, str(HERE.parent / "stage6_sft"))
from lora import inject_lora

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 高频题：语料里共现极强（北京↔首都 之类），预期 SFT 后能答对一部分
HIGH = [
    ("中国的首都是哪里？", "北京"),
    ("中华人民共和国的首都是哪里？", "北京"),
    ("故宫位于哪座城市？", "北京"),
    ("中国的第一大城市是？", "上海"),
    ("长江是中国的第几长河？", "第一"),
    ("中国的第二长河是？", "黄河"),
    ("世界上最高的山峰是？", "珠穆朗玛峰"),
    ("大熊猫主要生活在中国哪个省份？", "四川"),
    ("中国有多少个省级行政区？", "34"),
    ("秦始皇统一六国后建立的第一个朝代是？", "秦"),
]
# 低频题：语料里有、但"问法"罕见，预期答不对——知识边界
LOW = [
    ("水的化学式是什么？", "H2O"),
    ("世界上最小的国家是？", "梵蒂冈"),
    ("爱因斯坦出生于哪个国家？", "德国"),
    ("第一次世界大战爆发于哪一年？", "1914"),
    ("中国最大的沙漠是？", "塔克拉玛干"),
    ("恐龙大约在多少万年前灭绝？", "6500"),
    ("世界上面积最大的国家是？", "俄罗斯"),
    ("诺基亚公司来自哪个国家？", "芬兰"),
    ("圆周率的近似值是多少？", "3.14"),
    ("珠穆朗玛峰的海拔约为多少米？", "8848"),
]


def load_model(which):
    tok = BPETokenizer.load(STAGE7 / "cache" / "bpe.json")
    if which == "base":
        ckpt = torch.load(STAGE7 / "ckpt_large.pt")
        cfg = GPTConfig(**ckpt["config"])
        m = GPT(cfg).to(DEVICE); m.load_state_dict(ckpt["model"]); m.eval()
        return m, tok
    lora = (which == "lora")
    name = "ckpt_sft7_lora.pt" if lora else "ckpt_sft7_full.pt"
    ckpt = torch.load(HERE / name)
    cfg = GPTConfig(**ckpt["config"])
    m = GPT(cfg).to(DEVICE)
    if lora:
        inject_lora(m, r=ckpt.get("lora_r", 8))
    m.load_state_dict(ckpt["model"]); m.eval()
    return m, tok


# 补全格式版本（--format fill）：用训练分布里的句式，直接测"知识在不在"
FILL_HIGH = [
    ("中国的首都是", "北京"),
    ("中华人民共和国的首都是", "北京"),
    ("故宫位于", "北京"),
    ("中国的第一大城市是", "上海"),
    ("长江是中国的", "第一大河"),
    ("中国的第二长河是", "黄河"),
    ("世界上最高的山峰是", "珠穆朗玛峰"),
    ("大熊猫主要生活在", "四川"),
    ("中国的省级行政区有", "34"),
    ("秦始皇统一六国后建立的第一个朝代是", "秦"),
]
FILL_LOW = [
    ("水的化学式是", "H2O"),
    ("世界上最小的国家是", "梵蒂冈"),
    ("爱因斯坦出生于", "德国"),
    ("第一次世界大战爆发于", "1914"),
    ("中国最大的沙漠是", "塔克拉玛干"),
    ("恐龙大约在", "6500"),
    ("世界上面积最大的国家是", "俄罗斯"),
    ("诺基亚公司来自", "芬兰"),
    ("圆周率的近似值是", "3.14"),
    ("珠穆朗玛峰的海拔约为", "8848"),
]


@torch.no_grad()
def answer(model, tok, question, fmt, max_new=30):
    """贪心作答：ask 用问句格式，fill 用训练分布的补全格式。"""
    if fmt == "fill":
        ids = tok.encode(f'问：请补全这句话：“{question}”\n答：')
    else:
        ids = tok.encode(f"问：{question}\n答：")
    start = len(ids)
    for _ in range(max_new):
        ctx = torch.tensor([ids[-model.cfg.max_len:]], dtype=torch.long,
                           device=DEVICE)
        nxt = int(model(ctx)[0, -1].argmax().item())
        if nxt == EOS_ID:
            break
        ids.append(nxt)
    return tok.decode(ids[start:]).strip()


def eval_questions(model, tok, questions, fmt="ask"):
    results = []
    for q, ans in questions:
        out = answer(model, tok, q, fmt)
        ok = ans.lower() in out.lower()
        results.append((q, ans, out, ok))
    return results


def report(label, results):
    n_ok = sum(1 for _, _, _, ok in results)
    print(f"\n=== {label}（答对 {n_ok}/{len(results)}）===")
    for q, ans, out, ok in results:
        mark = "✓" if ok else "✗"
        print(f"  {mark} {q}")
        print(f"      期望: {ans} | 回答: {out[:40]!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=["ask", "fill"], default="ask",
                    help="ask=问句格式；fill=补全格式（训练分布内）")
    args = ap.parse_args()
    fmt = args.format
    questions = FILL_HIGH + FILL_LOW if fmt == "fill" else HIGH + LOW

    base, tok = load_model("base")
    full, _ = load_model("full")
    lora, _ = load_model("lora")
    print(f"设备 {DEVICE} | 词表 {len(tok):,} | 格式: {fmt}\n" + "=" * 60)

    for label, model in [
        ("基座（只会续写）", base),
        ("全量 SFT（20 题）", full),
        ("LoRA SFT（20 题）", lora),
    ]:
        all_res = eval_questions(model, tok, questions, fmt)
        report(label, all_res)
        high_ok = sum(1 for r in all_res[:10] if r[3])
        low_ok = sum(1 for r in all_res[10:] if r[3])
        print(f"  → 高频 {high_ok}/10 | 低频 {low_ok}/10")

    print("\n" + "=" * 60)
    print("解读：答对 = 预训练注入的知识被 SFT 接通；答错高频 = 知识还没到那层；")
    print("低频几乎全错 = 知识边界（预训练注入是有范围的）")


if __name__ == "__main__":
    main()
