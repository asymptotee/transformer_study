"""eval_minimind.py —— 9.5 桥接实验:官方 minimind 权重跑同一套 20 题

目的:我们的 94_full(117.8M, 44M token 维基)答 20 常识题 ≈0/20;官方
minimind-3 dense(64M,官方语料)在同一套题上答得怎么样?分离三变量:
架构(9.4 后同族)/ 规模(64M vs 117.8M)/ 数据量(官方语料 vs 44M token)。
跨分词器 val loss 不可比(词表 6400 vs 15124),所以桥接只用行为指标。

**格式公平性**(stage8/9.0 的教训:模板错配会制造假失败):
  · raw 格式:问：xxx\n答：——我们 94_full 的训练分布,官方模型没学过 →
    预期答不好,**记录但不判它笨**
  · chat 格式:走官方 tokenizer 的 apply_chat_template(user 消息)——
    官方 SFT 模型的真实训练分布。raw/chat 双格式都给结论
评估按 greedy 解码;EOS 停(chat 用官方模板的结束符,raw 用官方 eos)。

用法(Spark,两个下载源任选其一):
  # Transformers 格式(推荐,若 minimind-3 文件夹已下载)
  python eval_minimind.py --hf-dir ~/llm_study/mm_weights/minimind-3
  # 原始 pth(需 ~/llm_study/minimind 仓库 + out/ 下权重)
  python eval_minimind.py --pth ~/llm_study/mm_weights/minimind-3-pytorch/full_sft_768.pth
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))                      # eval_harness(题库)
from eval_harness import HIGH, LOW, PROMPTS        # noqa: E402  20 题 + 固定 prompt 原样复用

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW = 60


def load_hf(hf_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(hf_dir, trust_remote_code=True,
                                                 torch_dtype=torch.float16).to(DEVICE)
    return model.eval(), tok, "hf"


def load_pth(pth_path, mm_repo):
    sys.path.insert(0, str(mm_repo))
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mm_repo / "model")
    # minimind-3 dense:d768 / 8 层 / 8q / 4kv(见其 README 模型表)
    cfg = MiniMindConfig(hidden_size=768, num_hidden_layers=8,
                         num_attention_heads=8, num_key_value_heads=4)
    model = MiniMindForCausalLM(cfg).to(DEVICE)
    weights = torch.load(pth_path, map_location=DEVICE)
    model.load_state_dict(weights, strict=True)
    return model.eval(), tok, "pth"


def greedy_answer(model, tok, input_ids, stop_at_eos=True):
    """贪心续写 input_ids(1,T),到模板结束符或 MAX_NEW 停,返回续写文本。"""
    ids = [int(t) for t in input_ids.flatten().tolist()]
    start = len(ids)
    with torch.no_grad():
        for _ in range(MAX_NEW):
            x = torch.tensor([ids[-1024:]], dtype=torch.long, device=DEVICE)
            out = model(x)
            logits = out.logits if hasattr(out, "logits") else out[0]
            nxt = int(logits[0, -1].argmax().item())
            ids.append(nxt)
            if stop_at_eos and nxt == tok.eos_token_id:
                break
    return tok.decode(ids[start:], skip_special_tokens=False).strip()


def run_facts(model, tok, fmt):
    """fmt: raw(问：…答：)或 chat(官方模板)。返回 20 题结果列表。"""
    rows = []
    for q, ans in HIGH + LOW:
        if fmt == "raw":
            prompt = f"问：{q}\n答："
            ids = tok(prompt, add_special_tokens=False)["input_ids"]
            stop = True
        else:
            prompt = tok.apply_chat_template([{"role": "user", "content": q}],
                                             tokenize=False, add_generation_prompt=True)
            ids = tok(prompt, add_special_tokens=False)["input_ids"]
            stop = True
        out = greedy_answer(model, tok, torch.tensor([ids], device=DEVICE), stop)
        rows.append({"q": q, "ans": ans, "out": out, "ok": ans.lower() in out.lower(),
                     "fmt": fmt, "prompt": prompt[:40]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", help="minimind-3 Transformers 格式文件夹")
    ap.add_argument("--pth", help="原始 pth 权重路径")
    ap.add_argument("--mm-repo", default=str(Path.home() / "llm_study" / "minimind"))
    ap.add_argument("--label", default="")
    ap.add_argument("--format", choices=["both", "raw", "chat"], default="both")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    assert (args.hf_dir is None) != (args.pth is None), "hf-dir 与 pth 二选一"
    if args.pth:
        model, tok, src = load_pth(Path(args.pth), Path(args.mm_repo))
    else:
        model, tok, src = load_hf(Path(args.hf_dir))
    label = args.label or (Path(args.pth).stem if args.pth else Path(args.hf_dir).name)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"=== {label} | {src} | {n_params/1e6:.1f}M | 词表 {len(tok)} | 设备 {DEVICE}")
    print(f"eos_token={tok.eos_token!r} ({tok.eos_token_id})")

    report = {"label": label, "source": src, "params": n_params,
              "vocab": len(tok), "facts": {}}
    for fmt in ("raw", "chat"):
        if args.format not in ("both", fmt):
            continue
        rows = run_facts(model, tok, fmt)
        ok = sum(1 for r in rows if r["ok"])
        hi = sum(1 for r in rows[:10] if r["ok"])
        lo = sum(1 for r in rows[10:] if r["ok"])
        print(f"\n--- 20 题 [{fmt}]: 答对 {ok}/20(高频 {hi}/10 | 低频 {lo}/10)")
        for r in rows:
            mark = "✓" if r["ok"] else "✗"
            print(f"  {mark} {r['q']}")
            print(f"      期望: {r['ans']} | 回答: {r['out'][:60]!r}")
        report["facts"][fmt] = {"ok": ok, "high": hi, "low": lo,
                                "rows": [{"q": r["q"], "ans": r["ans"], "ok": r["ok"],
                                          "out": r["out"][:100]} for r in rows]}

    print("\n--- 固定 prompt 续写(chat 格式)")
    report["samples"] = {}
    for p in PROMPTS:
        prompt = tok.apply_chat_template([{"role": "user", "content": p}],
                                         tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        out = greedy_answer(model, tok, torch.tensor([ids], device=DEVICE))
        print(f"  【{p}】{out[:80]!r}")
        report["samples"][p] = out[:100]

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nJSON: {args.json_out}")


if __name__ == "__main__":
    main()
