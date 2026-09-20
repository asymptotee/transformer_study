"""probe_verify.py —— 知识"在不在权重里"的验证探针(强制选择)

**要回答的问题(2026-09-20)**:14_mix 在 131 题上比 11_5_chat_2000 低 10 分
(73 vs 83),但**它见到的单轮知识数据反而更多**(45,000 vs 32,057 条)。
逐条审计后发现 21 条丢失**全部是真忘**(输出里根本没有答案词),而且错答有强模式:

    一战 1914→1949、辛亥 1911→1949、二战 1945→1949     ← 1949 出现 3 次
    加拿大首都 渥太华→华盛顿特区;唐朝开国 李渊→李世民;万有引力 牛顿→爱因斯坦

**错的是"高频吸引子",不是"没学过"** —— 1949、李世民、爱因斯坦在语料里都比正确答案
更高频。所以假设是:**知识在权重里,只是开放式生成时被高频邻居压住了。**

这个探针换个**取用路径**去问同一个事实:不给开放式问题,给一个陈述句问"对吗"。
验证(判断真伪)和生成是两条不同的路径 —— 如果知识在,验证应该能过。

  · 真陈述的「是」倾向 **>** 假陈述的「是」倾向  → 知识在,取用被压制 → RL 有希望
  · 两者差不多                                  → 知识真被覆盖     → RL 救不回来

**读的是 logit,不是生成结果** —— 只取最后一个位置的 logits,比较「是」与「否」的
差(不依赖模型会不会遵循"回答是或否"这个指令;那两个 token 的**相对**高低就是读数)。

假陈述的来源见 probe_21.jsonl 的 src 字段:
  · model = 模型自己的错答(最有信息量,它就是模型的实际吸引子)
  · sub   = 模型答的是"非答案"(如复读题干),另配一个同域可信的错误说法

**已知的方法论保留**:假陈述若"太不合理",模型可能靠常识合理性而非具体知识否掉它,
从而高估"知识在"。所以必须带 **11_5_chat_2000 作阳性对照**(它答对了这些题);
若它也分不开真伪,说明这个探针测的不是知识。

用法(Spark):
  ~/llm_study/.venv/bin/python probe_verify.py --ckpt ... --bpe ... --label 14_mix
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
from bpe import BPETokenizer                      # noqa: E402
from model_modern import GPT, GPTConfig           # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IM_END = "<|im_end|>"


def prompt_of(stmt):
    """把陈述句包装成"判断对错"的提问。"""
    return (f"<|im_start|>user\n陈述：{stmt}\n这个陈述对吗？回答"
            f"{IM_END}\n<|im_start|>assistant\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bpe", required=True)
    ap.add_argument("--probe", default=str(HERE / "probe_21.jsonl"))
    ap.add_argument("--label", default="")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    tok = BPETokenizer.load(args.bpe)
    ckpt = torch.load(args.ckpt)
    model = GPT(GPTConfig(**ckpt["config"])).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    label = args.label or Path(args.ckpt).stem

    yes_id, no_id = tok.encode("是")[0], tok.encode("否")[0]
    print(f"=== {label} | 「是」id={yes_id} 「否」id={no_id}", flush=True)

    rows = [json.loads(ln) for ln in open(args.probe, encoding="utf-8")]
    out = []
    with torch.no_grad():
        for r in rows:
            m = {}
            for k in ("true", "false"):
                ids = tok.encode(prompt_of(r[k]))
                ctx = torch.tensor([ids], dtype=torch.long, device=DEVICE)
                logits, _ = model.forward_cached(ctx, None)
                lg = logits[0, -1]
                m[k] = float(lg[yes_id] - lg[no_id])      # 「是」相对「否」的倾向
            out.append({"q": r["q"], "gold": r["gold"], "src": r["src"],
                        "margin_true": m["true"], "margin_false": m["false"],
                        "ok": m["true"] > m["false"]})
            if len(out) % 7 == 0:
                print(f"  …{len(out)}/{len(rows)}", flush=True)

    n = len(out)
    ok = sum(x["ok"] for x in out)
    mt = sum(x["margin_true"] for x in out) / n
    mf = sum(x["margin_false"] for x in out) / n
    print(f"\n--- [{label}] 真陈述 > 假陈述: {ok}/{n} = {ok/n:.0%} "
          f"(随机 50%)")
    print(f"    平均「是」倾向: 真陈述 {mt:+.2f} | 假陈述 {mf:+.2f} | "
          f"差 {mt-mf:+.2f}")
    # 只有 model 来源的那些(模型自己的吸引子)—— 这几个最能说明问题
    mo = [x for x in out if x["src"] == "model"]
    if mo:
        k = sum(x["ok"] for x in mo)
        print(f"    其中「模型自己的错答」那 {len(mo)} 条: {k}/{len(mo)} = "
              f"{k/len(mo):.0%}")

    if args.json_out:
        p = Path(args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"label": label, "n": n, "ok": ok, "rows": out}, f,
                      ensure_ascii=False, indent=1)
        print(f"JSON → {p}")


if __name__ == "__main__":
    main()
