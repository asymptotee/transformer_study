"""chat14.py —— 14.4:多轮对话工具(带历史)

**为什么不能直接用 chat11.py**:那个工具是**单轮无状态**的 —— 它每次提问都
渲染干净的单轮模板(`<|im_start|>user\\n{q}<|im_end|>\\n<|im_start|>assistant\\n`),
历史全丢掉。stage11 那么设计是对的(SFT 数据只有单轮对,多轮是外推,历史会污染),
但 **stage14 的 14_mt 就是拿多轮数据训出来的** —— 用单轮工具测它,等于把它的
训练分布扔掉,测的是另一个模型。

本工具保留完整历史,每轮用 minimind 的 chat_template 重新渲染
(`add_generation_prompt=True`),与训练数据走同一条代码路径。

**每轮重渲整个历史**(不跨轮复用 KV cache):对话就几百 token,prefill 很快,
而手工拼接"上一轮生成到哪、下一个分隔符是什么"极易错位 —— 用正确性换那点速度。

用法(Spark):
  # 交互模式(/reset 清空历史,/history 看历史,空行退出)
  ~/llm_study/.venv/bin/python chat14.py --ckpt ../stage11_datascale/ckpt_14_mt.pt

  # 一次跑一串轮次(注意:--ask 在这里是**同一段对话的连续轮**
  #   —— chat11.py 里是各自独立的单轮,语义不同)
  ~/llm_study/.venv/bin/python chat14.py --ckpt ... \
      --ask "中国的首都是哪里？" --ask "它有多少年建都史？"

实测结论(见 README 14.3):
  · 多轮模型(14_mt)带历史作答 72/131,污染率仅 8.1%
  · 单轮模型同条件下只有 34/131(污染率 63.9%)—— 一看见历史就被带跑
  · 历史回忆:多轮 87% / 单轮 95%(两个模型都读得到历史)
  · 但**「用历史做推理」(指代消解等)没学会** —— 手跑实测:第一轮问
    "中国的首都是哪里"、第二轮问"它是哪个国家的城市",模型不回答指代问题,
    只是继续谈北京。别对这个工具期望过高
"""

import argparse
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
REPO = HERE.parent
MM = Path.home() / "llm_study" / "minimind"
sys.path.insert(0, str(MM))
sys.path.insert(0, str(REPO / "stage4_scaling_bpe"))
sys.path.insert(0, str(REPO / "stage5_capstone"))
sys.path.insert(0, str(REPO / "stage9_modern_gpt"))
from transformers import AutoTokenizer                      # noqa: E402
from bpe import BPETokenizer, EOS_ID                        # noqa: E402
from model_modern import GPT, GPTConfig                     # noqa: E402
from sampling import sample_next                            # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IM_END = "<|im_end|>"
# 任何一个模板标记都该终止生成:模型有时会吐出 <|im_start|> 而不是 <|im_end|>
# (两者前缀相同),只查 IM_END 会让它继续吐到 max_new 截断,尾部漏出半截 `<|im_`
IM_ANY = "<|im"
# 必须整段剥 think:minimind 的 chat_template 会给**每个** assistant 轮自动插入
# `<think>\n\n</think>\n\n`,而训练数据是 --strip-think 整段剥掉的 —— 不剥就是
# 分布外输入,模型会退化(实测:第 2 轮直接复读上一轮答复)。
THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def load(ckpt_path, bpe_path):
    ckpt = torch.load(ckpt_path)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, BPETokenizer.load(bpe_path), cfg


@torch.no_grad()
def reply(model, tok, tok_mm, history, max_new=120, temperature=0.7, top_p=0.9,
          rep_penalty=1.1):
    """给定完整历史(messages 列表),生成下一条 assistant 回复。"""
    prompt = tok_mm.apply_chat_template(history, tokenize=False,
                                        add_generation_prompt=True)
    prompt = THINK_RE.sub("", prompt)          # 与训练数据同分布,见文件头 THINK_RE
    ids = tok.encode(prompt)
    ctx = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    logits, past = model.forward_cached(ctx, None)          # prefill 整段历史
    gen = []
    stopped = False
    for _ in range(max_new):
        nxt = sample_next(logits[0, -1], ids + gen, temperature, 0, top_p,
                          rep_penalty)
        gen.append(nxt)
        txt = tok.decode(gen)
        if IM_ANY in txt or nxt == EOS_ID:      # 任一模板标记都停,见 IM_ANY
            stopped = True
            break
        tk = torch.tensor([[nxt]], dtype=torch.long, device=DEVICE)
        logits, past = model.forward_cached(tk, past)       # 增量步
    return tok.decode(gen).split(IM_ANY)[0].strip(), stopped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "ckpt_14_mt.pt"))
    ap.add_argument("--bpe", default=str(REPO / "stage11_datascale" / "cache_mm10g" /
                                         "bpe.json"))
    ap.add_argument("--ask", action="append", default=[],
                    help="同一段对话的连续轮次(不是各自独立的单轮)")
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--rep-penalty", type=float, default=1.1)
    args = ap.parse_args()

    model, tok, cfg = load(args.ckpt, args.bpe)
    tok_mm = AutoTokenizer.from_pretrained(MM / "model")
    print(f"[chat14] {Path(args.ckpt).name} | "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M | vocab {len(tok)} "
          f"| 温度 {args.temperature} top_p {args.top_p} 重复惩罚 {args.rep_penalty} "
          f"| **多轮带历史**", flush=True)

    def say(q, k=None):
        """跑一轮并把结果 append 回历史。被 max_new 截断时给出警告 ——
        截断的轮次没有收尾标记,喂回历史会造出畸形对话,两个模型都只能猜
        (这个坑踩过:曾据此误判"单轮模型主题粘连停不下来",其实是工具造的)。"""
        history.append({"role": "user", "content": q})
        out, stopped = reply(model, tok, tok_mm, history, args.max_new,
                             args.temperature, args.top_p, args.rep_penalty)
        history.append({"role": "assistant", "content": out})
        if not stopped:
            print(f"⚠️  本轮被 max_new={args.max_new} 截断,没有收尾标记 —— "
                  f"历史里这一轮是残缺的,后续轮次的结果不可信,请调大 --max-new",
                  flush=True)
        return out

    history = []
    if args.ask:
        for k, q in enumerate(args.ask, 1):
            out = say(q, k)
            print(f"\n--- 第 {k} 轮 ---\n问:{q}\n答:{out}", flush=True)
        return

    print("交互模式(/reset 清空历史, /history 看历史, 空行退出)")
    while True:
        try:
            q = input(f"\n你[{len(history)//2 + 1}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        if q == "/reset":
            history = []
            print("(历史已清空)")
            continue
        if q == "/history":
            for m in history:
                print(f"  {m['role']}: {m['content'][:70]}")
            continue
        print(f"模型 > {say(q)}")


if __name__ == "__main__":
    main()
