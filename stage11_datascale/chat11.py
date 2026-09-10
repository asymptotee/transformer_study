"""chat11.py —— 11.3 最终模型的交互式对话(chat 模板 + KV cache)

用法(Spark):
  # 交互模式(多轮,输入 /reset 清空历史、/exit 退出)
  ~/llm_study/.venv/bin/python chat11.py

  # 单次问答(可多个 --ask,便于脚本批量测)
  ~/llm_study/.venv/bin/python chat11.py \
      --ask "中国的首都是哪里？" --ask "用一句话解释什么是机器学习"

说明:
  · 模板 = 训练时的官方 chat 形态(无 system 单轮):
      <|im_start|>user\\n{q}<|im_end|>\\n<|im_start|>assistant\\n
    多轮时把历史按同样模板拼进 context(训练是单轮,多轮属于外推,可玩)
  · 用 KV cache 增量解码(9.3);停止:生成文本出现 <|im_end|> 或 EOS
  · **事实题建议 --temperature 0**(贪心):采样(0.7)会引入幻觉
    ("水的化学式是CO2O");重复惩罚 1.2 会误伤召回(H2O/π 答丢),
    1.1 是安全线——复读拖尾是本模型固有弱点(无 RLHF),惩罚只能缓解
  · 上下文截断:训练窗口只有 256 token,超过会进入外推区胡说;render
    按 512 token 预算自动丢弃最老轮次
"""

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "stage4_scaling_bpe"))
sys.path.insert(0, str(HERE.parent / "stage5_capstone"))
sys.path.insert(0, str(HERE.parent / "stage9_modern_gpt"))
from bpe import BPETokenizer, EOS_ID                     # noqa: E402
from model_modern import GPT, GPTConfig                  # noqa: E402
from sampling import sample_next                         # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IM_END = "<|im_end|>"


def render(history, tok=None, cap=512):
    """history: [(role, content)],渲染成训练同款模板(结尾 assistant 待续)。

    截断:模型训练窗口只有 256 token,上下文过长会进入"位置外推区"导致
    胡言(实测 ~800 token 时崩)。给了 tok 就按 token 预算从最老的轮次
    开始丢,保证上下文不超出 cap(默认 512,留出生成空间)。
    """
    while True:
        s = ""
        for role, content in history:
            s += f"<|im_start|>{role}\n{content}{IM_END}\n"
        s += "<|im_start|>assistant\n"
        if tok is None or len(tok.encode(s)) <= cap or len(history) <= 2:
            return s
        history = history[2:]                      # 丢最老的一轮 user+assistant


@torch.no_grad()
def reply(model, tok, history, max_new=120, temperature=0.7, top_p=0.9,
          rep_penalty=1.2):
    prompt = render(history, tok)
    ids = tok.encode(prompt)
    ctx = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    logits, past = model.forward_cached(ctx, None)       # prefill
    gen = []
    for _ in range(max_new):
        nxt = sample_next(logits[0, -1], ids + gen, temperature, 0, top_p,
                          rep_penalty)
        gen.append(nxt)
        text = tok.decode(gen)
        if IM_END in text or nxt == EOS_ID:
            break
        tk = torch.tensor([[nxt]], dtype=torch.long, device=DEVICE)
        logits, past = model.forward_cached(tk, past)    # 增量步
    return tok.decode(gen).split(IM_END)[0].strip()


def load(ckpt_path, bpe_path):
    ckpt = torch.load(ckpt_path)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = BPETokenizer.load(bpe_path)
    return model, tok, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "ckpt_11_3_chat.pt"))
    ap.add_argument("--bpe", default=str(HERE / "cache_mm10g" / "bpe.json"))
    ap.add_argument("--ask", action="append", default=[],
                    help="单次问答(可重复);不给则进入交互模式")
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="事实题用 0(贪心),闲聊用 0.7")
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--rep-penalty", type=float, default=1.1,
                    help="1.2 会误伤知识召回,1.1 是安全线,1.0 关")
    args = ap.parse_args()

    model, tok, cfg = load(args.ckpt, args.bpe)
    print(f"[chat11] {Path(args.ckpt).name} | {sum(p.numel() for p in model.parameters())/1e6:.1f}M"
          f" | vocab {len(tok)} | 温度 {args.temperature} top_p {args.top_p} "
          f"重复惩罚 {args.rep_penalty}", flush=True)

    if args.ask:
        # 每题独立上下文(避免互相污染);想测多轮用交互模式
        for q in args.ask:
            out = reply(model, tok, [("user", q)], args.max_new,
                        args.temperature, args.top_p, args.rep_penalty)
            print(f"\n问:{q}\n答:{out}", flush=True)
        return

    history = []
    print("交互模式(空行退出;/reset 清历史)")
    while True:
        try:
            q = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        if q == "/reset":
            history = []
            print("(历史已清空)")
            continue
        history.append(("user", q))
        out = reply(model, tok, history, args.max_new, args.temperature,
                    args.top_p, args.rep_penalty)
        history.append(("assistant", out))
        print(f"模型 > {out}")


if __name__ == "__main__":
    main()
