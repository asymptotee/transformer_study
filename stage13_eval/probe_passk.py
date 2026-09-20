"""probe_passk.py —— 采样探针:正确答案在采样分布里到底有没有质量

**为什么用它替掉 probe_verify.py(2026-09-20)**

上一版是"判断真伪"的验证探针,结果**无效且被阳性对照当场识破**:三个模型
(含贪心下明明答对的 11_5_chat_2000)全在 50% 附近,「是」倾向对真伪都是 +2~+3.8
—— 它们只是倾向说"是",没在做判断。这批模型没训过"判断对错",那是 OOD 任务。

换一个**更直接对应决策**的问法。原本要答的问题是:

    14_mix 丢掉的那 21 条知识,是"取不出来"还是"真没了"?
    (它见到的单轮知识数据比 11_5_chat_2000 更多,分却更低,所以怀疑是取用被压制)

但"在权重里" vs "取得出"不是能干净分开的两件事 —— 分布式存储下,"有没有"就等于
"前向传播在任何合理上下文里能不能把它顶上来"。所以把它**操作化**成 RL 关心的东西:

    **GRPO 靠采样+奖励工作。正确答案在采样分布里概率≈0 就拿不到信号;
      只要偶尔采得到(哪怕 5%),就能被放大。**

于是测 **pass@k**:温度采样 k 次,正确答案是否至少出现一次。

  · 出现过  → ✅ 有质量存在,RL 有抓手
  · 从未出现 → ❌ 质量可忽略,RL 救不回来

**没有 OOD 问题**(就是正常生成,只是采样而非贪心)。11_5_chat_2000 贪心下就答对,
是天然的阳性对照 —— 它必须接近 100%,否则说明这个探针也不可信。

用法(Spark):
  ~/llm_study/.venv/bin/python probe_passk.py --ckpt ... --bpe ... \
      --label 14_mix --k 20 --temperature 0.8
"""

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "stage4_scaling_bpe"))
sys.path.insert(0, str(REPO / "stage9_modern_gpt"))
sys.path.insert(0, str(HERE))
from bpe import BPETokenizer, EOS_ID               # noqa: E402
from model_modern import GPT, GPTConfig            # noqa: E402
from judge import judge                            # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IM_END = "<|im_end|>"


@torch.no_grad()
def sample(model, tok, prompt, max_new, temp, top_p, rng):
    """温度采样一次。返回 (生成的文本, 是否自然收尾)。"""
    ids = tok.encode(prompt)
    ctx = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    logits, past = model.forward_cached(ctx, None)
    gen_ids, stopped = [], False
    for _ in range(max_new):
        lg = logits[0, -1] / max(temp, 1e-6)
        srt, idx = torch.sort(lg, descending=True)
        probs = torch.softmax(srt, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        keep = (cum - probs) < top_p          # top-p 截断
        p = torch.softmax(srt[keep], dim=-1)
        nxt = int(idx[keep][torch.multinomial(p, 1, generator=rng)].item())
        if nxt == EOS_ID or IM_END in tok.decode(gen_ids):
            stopped = True
            break
        gen_ids.append(nxt)
        t = torch.tensor([[nxt]], dtype=torch.long, device=DEVICE)
        logits, past = model.forward_cached(t, past)
    return tok.decode(gen_ids).split(IM_END)[0].strip(), stopped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bpe", required=True)
    ap.add_argument("--probe", default=str(HERE / "probe_21.jsonl"))
    ap.add_argument("--qa", default=str(HERE / "qa_v2_freq.jsonl"))
    ap.add_argument("--label", default="")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    tok = BPETokenizer.load(args.bpe)
    ckpt = torch.load(args.ckpt)
    model = GPT(GPTConfig(**ckpt["config"])).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    label = args.label or Path(args.ckpt).stem

    gold = {json.loads(l)["q"]: json.loads(l)["a"]
            for l in open(args.qa, encoding="utf-8")}
    rows = [json.loads(ln) for ln in open(args.probe, encoding="utf-8")]
    prng = torch.Generator(device=DEVICE).manual_seed(args.seed)
    print(f"=== {label} | k={args.k} T={args.temperature} top_p={args.top_p} "
          f"| {len(rows)} 题", flush=True)

    out = []
    for i, r in enumerate(rows):
        al = gold.get(r["q"], [r["gold"]])
        hits, raw_hits, stops, samples = 0, 0, 0, []
        for _ in range(args.k):
            prompt = (f"<|im_start|>user\n{r['q']}{IM_END}\n"
                      f"<|im_start|>assistant\n")
            txt, st = sample(model, tok, prompt, args.max_new,
                             args.temperature, args.top_p, prng)
            stops += st
            # **严格判据**:走 judge(回声截断后),不是"答案词出现在任意位置"
            raw, real, _ = judge(txt, al, r["q"])
            raw_hits += raw
            hits += real
            if len(samples) < 3:
                samples.append(txt[:160])
        out.append({"q": r["q"], "gold": r["gold"], "src": r["src"],
                    "hits": hits, "raw_hits": raw_hits, "k": args.k,
                    "stop_rate": stops / args.k, "samples": samples})
        print(f"  real {hits:2d} raw {raw_hits:2d} /{args.k}  {r['q']}", flush=True)

    n = len(out)
    ever = sum(1 for x in out if x["hits"] > 0)
    ever_raw = sum(1 for x in out if x["raw_hits"] > 0)
    total = sum(x["hits"] for x in out)
    total_raw = sum(x["raw_hits"] for x in out)
    print(f"\n--- [{label}] 严格(real): 至少一次 {ever}/{n} = {ever/n:.0%} | "
          f"总体 {total}/{n*args.k} = {total/(n*args.k):.1%}")
    print(f"    宽松(raw) : 至少一次 {ever_raw}/{n} = {ever_raw/n:.0%} | "
          f"总体 {total_raw}/{n*args.k} = {total_raw/(n*args.k):.1%} "
          f"← 这一行是上一版报的数")
    print(f"    采样收尾率 {sum(x['stop_rate'] for x in out)/n:.0%}")

    if args.json_out:
        p = Path(args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"label": label, "k": args.k, "n": n, "ever": ever,
                       "total_hits": total, "ever_raw": ever_raw,
                       "total_raw": total_raw, "rows": out}, f,
                      ensure_ascii=False, indent=1)
        print(f"JSON → {p}")


if __name__ == "__main__":
    main()
