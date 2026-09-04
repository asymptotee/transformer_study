"""eval_harness.py —— stage9 统一评估出口(9.0 定标尺)

一个命令出齐 stage9 全程要用的全部数字(跨里程碑可比):

  [1] val loss / perplexity    ← 与 stage7 train_large.py 同口径:
      缓存 token 的最后 5% 当验证集,随机 4 个 batch(32×256)平均,bf16
  [2] 20 道常识题(ask + fill)  ← 与 stage8 eval_facts.py 同题、同模板、同判定
      (answer 含关键词即算对;模板字符串逐字一致才可比)
  [3] 固定 prompt 生成样例      ← 与 stage7 eval_ckpt.py 同 prompt、同解码参数

为什么"逐字一致"很重要:stage8 的教训是评估设计偏差。跨里程碑比数字时,
题面/模板/判定/解码参数任何一个变了,数字都不可比。所以本文件把三处
"评估素材"原样收编,并各自注明与哪个脚本保持同步。

运行(模型、语料、ckpt 在 DGX Spark 上):
  python -u eval_harness.py --ckpt ../stage7_gpu_scale/ckpt_large.pt \
      --label stage7-baseline --json-out results/stage7-baseline.json

9.1 之后零件 A/B 也用同一命令,只换 --ckpt 与 --label;
9.4 现代架构落地后传 --model-module model_modern 即可(harness 不改)。

用法示例见 README.md(9.0)。
"""

import argparse
import json
import math
import shlex
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
REPO = HERE.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 复用仓库共享代码:模型(stage3)、分词(stage4)、解码(stage5)
for sub in ("stage3_gpt", "stage4_scaling_bpe", "stage5_capstone"):
    sys.path.insert(0, str(REPO / sub))
from bpe import BPETokenizer, EOS_ID          # noqa: E402
from sampling import generate                  # noqa: E402

# ---------------------------------------------------------------------------
# [2] 20 道常识题 —— 与 stage8_knowledge_sft/eval_facts.py 逐字同步(HIGH/LOW/
# FILL_HIGH/FILL_LOW 及模板串)。改题面 = 换题库,先想清楚可比性再动。
# ---------------------------------------------------------------------------
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

# [3] 固定 prompt —— 与 stage7_gpu_scale/eval_ckpt.py 的 PROMPTS 逐字同步;
# 解码参数同该脚本默认(temperature 0.7 / top-p 0.9 / 重复惩罚 1.2)。
PROMPTS = [
    "中国的首都是",
    "北京是",
    "数学是",
    "水是一种",
    "长城位于",
    "鲁迅是",
]


def load_model(args, ckpt):
    """从 ckpt 恢复模型:ckpt = {"model": state_dict, "config": GPTConfig 字段,
    "step": int}。--model-module 让 9.4 起可指到现代架构模块(同接口)。"""
    import importlib
    mod = importlib.import_module(args.model_module)
    cfg = mod.GPTConfig(**ckpt["config"])
    model = mod.GPT(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model"])          # strict:架构不匹配立刻暴露
    model.eval()
    return model, cfg


@torch.no_grad()
def val_loss(model, tok, tokens_path, frac, batches, block, batch):
    """与 stage7 train_large.py 的 val_loss 同口径:取缓存 token 末尾 frac
    当验证集,随机取 batches 个 (batch×block) 窗口,CE(ignore_index=0)
    按 token 数加权平均;cuda 上套 bf16 autocast(与训练时一致)。"""
    ids = torch.load(tokens_path)
    n_val = int(len(ids) * frac)
    val_ids = ids[-n_val:].to(DEVICE)
    use_amp = DEVICE.startswith("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        tot, n = 0.0, 0
        for _ in range(batches):
            ix = torch.randint(len(val_ids) - block - 1, (batch,),
                               device=val_ids.device)
            x = torch.stack([val_ids[i:i + block] for i in ix])
            y = torch.stack([val_ids[i + 1:i + 1 + block] for i in ix])
            loss = F.cross_entropy(model(x).reshape(-1, len(tok)),
                                   y.reshape(-1), ignore_index=0)
            tot += loss.item() * x.numel()
            n += x.numel()
    return tot / n


def fact_answer(model, tok, question, fmt, max_new=30):
    """贪心作答(与 eval_facts.py 的 answer 逐字同源)。"""
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


def eval_facts(model, tok, questions, fmt):
    rows = []
    for q, ans in questions:
        out = fact_answer(model, tok, q, fmt)
        rows.append({"q": q, "ans": ans, "out": out, "ok": ans.lower() in out.lower()})
    return rows


def report_facts(model_name, rows):
    ok = sum(1 for r in rows if r["ok"])
    high = sum(1 for r in rows[:10] if r["ok"])
    low = sum(1 for r in rows[10:] if r["ok"])
    print(f"\n--- 常识题: {model_name}(答对 {ok}/20 = 高频 {high}/10 | 低频 {low}/10)")
    for r in rows:
        mark = "✓" if r["ok"] else "✗"
        print(f"  {mark} {r['q']}")
        print(f"      期望: {r['ans']} | 回答: {r['out'][:40]!r}")
    return {"high": high, "low": low, "ok": ok, "rows": rows}


def main():
    ap = argparse.ArgumentParser(description="stage9 统一评估出口(见文件头)")
    ap.add_argument("--ckpt", required=True, help="模型权重(含 config/step)")
    ap.add_argument("--bpe", help="BPE 词表 json(默认 stage7 cache/bpe.json)")
    ap.add_argument("--tokens", help="缓存 token(默认 stage7 cache/tokens.pt;用于 val loss)")
    ap.add_argument("--model-module", default="model_gpt",
                    help="模型模块名(9.4 起可指 model_modern)")
    ap.add_argument("--label", default="", help="本次运行标签(写进报告/JSON)")
    ap.add_argument("--no-val", action="store_true", help="跳过 val loss 段")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--val-batches", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--fact-format", choices=["both", "ask", "fill"], default="both")
    ap.add_argument("--no-facts", action="store_true")
    ap.add_argument("--no-samples", action="store_true")
    ap.add_argument("--max-new", type=int, default=80, help="prompt 续写长度")
    ap.add_argument("--json-out", help="机器可读结果(json,跨里程碑汇总用)")
    ap.add_argument("--device", default="auto", help="cpu/cuda/auto")
    args = ap.parse_args()

    global DEVICE
    if args.device != "auto":
        DEVICE = args.device

    bpe_path = Path(args.bpe) if args.bpe else REPO / "stage7_gpu_scale" / "cache" / "bpe.json"
    tokens_path = Path(args.tokens) if args.tokens else REPO / "stage7_gpu_scale" / "cache" / "tokens.pt"

    ckpt = torch.load(args.ckpt)
    model, cfg = load_model(args, ckpt)
    tok = BPETokenizer.load(bpe_path)
    n_params = sum(p.numel() for p in model.parameters())
    label = args.label or Path(args.ckpt).name
    print(f"=== {label} | 参数 {n_params:,} | 词表 {len(tok):,} | "
          f"训练步 {ckpt.get('step', '?')} | 设备 {DEVICE} | "
          f"cmd: {shlex.join(sys.argv)}")
    print(f"    模型: d_model={cfg.d_model} n_layers={cfg.n_layers} "
          f"n_heads={cfg.n_heads} d_ff={cfg.d_ff} max_len={cfg.max_len}")

    report = {"label": label, "ckpt": str(args.ckpt), "ckpt_step": ckpt.get("step"),
              "params": n_params, "vocab": len(tok), "config": vars(cfg),
              "cmd": shlex.join(sys.argv), "time": time.strftime("%Y-%m-%d %H:%M:%S")}

    # [1] val loss(与 stage7 同口径)
    report["val"] = None
    if not args.no_val:
        if tokens_path.exists():
            vl = val_loss(model, tok, tokens_path, args.val_frac,
                          args.val_batches, args.block_size, args.batch_size)
            print(f"\n--- val loss: {vl:.3f}(ppl {math.exp(vl):.1f}) | "
                  f"口径: token 末 {args.val_frac:.0%}, {args.val_batches}×"
                  f"{args.batch_size}×{args.block_size}, 与 stage7 相同")
            report["val"] = {"loss": vl, "ppl": math.exp(vl)}
        else:
            print(f"\n(tokens 缓存不存在: {tokens_path},跳过 val loss)")

    # [2] 20 道常识题
    report["facts"] = {}
    if not args.no_facts:
        for fmt in ("ask", "fill"):
            if args.fact_format in ("both", fmt):
                questions = FILL_HIGH + FILL_LOW if fmt == "fill" else HIGH + LOW
                rows = eval_facts(model, tok, questions, fmt)
                stat = report_facts(f"{label} [{fmt}]", rows)
                report["facts"][fmt] = {k: stat[k] for k in ("high", "low", "ok")}
                report["facts"][fmt]["rows"] = [{
                    "q": r["q"], "ans": r["ans"], "ok": r["ok"],
                    "out": r["out"][:100]} for r in rows]

    # [3] 固定 prompt 生成样例(stage7 eval_ckpt 同参)
    report["samples"] = {}
    if not args.no_samples:
        print("\n--- 固定 prompt 续写(t=0.7 top-p=0.9 rep=1.2)")
        for p in PROMPTS:
            out = generate(model, tok, p, args.max_new, 0.7, 0, 0.9, 1.2)
            print(f"  【{p}】")
            print(f"    {out[:100]}\n")
            report["samples"][p] = out[:100]

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"JSON 已写: {args.json_out}")


if __name__ == "__main__":
    main()
