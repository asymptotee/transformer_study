"""chat11.py —— 11.3 最终模型的问答工具(chat 模板 + KV cache,**单轮无状态**)

用法(Spark):
  # 交互模式(每问独立,输入空行退出)
  ~/llm_study/.venv/bin/python chat11.py

  # 单次问答(可多个 --ask,便于脚本批量测;同样每题独立)
  ~/llm_study/.venv/bin/python chat11.py \
      --ask "中国的首都是哪里？" --ask "用一句话解释什么是机器学习"

为什么默认单轮无状态(实测结论,见 README 11.3):
  · 本模型的 SFT 数据只有单轮对(user→assistant),从未见过"历史+新问题"
    结构——多轮对话是外推。表现:主题粘连(答完"你好"聊了 ML,下一问
    "介绍北京"答半句就滚回 ML 列表)、角色混乱("你是一个有意识的AI")、
    自问自答(把"问-答"表面模式当文风模仿)。
  · 因此每次提问都渲染干净的单轮模板:
      <|im_start|>user\\n{q}<|im_end|>\\n<|im_start|>assistant\\n
    轮与轮之间完全独立,上下文不互相污染。
  · 要真正的多轮,需要多轮 SFT(官方数据里有多轮对话,10.3 时被我们
    筛掉了)——那是后续实验,不是工具层能修的。

其他实测结论:
  · 事实题建议 --temperature 0(贪心):采样(0.7)会引入幻觉
    ("水的化学式是CO2O");重复惩罚 1.2 会误伤召回(H2O/π 答丢),
    1.1 是安全线——复读拖尾是本模型固有弱点(无 RLHF),惩罚只能缓解
  · 停止:生成文本出现 <|im_end|> 或 EOS;--max-new 封顶
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


def single_turn_prompt(q):
    """训练同款单轮模板(结尾 assistant 待续)。"""
    return f"<|im_start|>user\n{q}{IM_END}\n<|im_start|>assistant\n"


@torch.no_grad()
def reply(model, tok, question, max_new=120, temperature=0.7, top_p=0.9,
          rep_penalty=1.1):
    """单轮问答:渲染干净上下文(不带任何历史),KV cache 增量解码。"""
    ids = tok.encode(single_turn_prompt(question))
    ctx = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    logits, past = model.forward_cached(ctx, None)       # prefill
    gen = []
    for _ in range(max_new):
        nxt = sample_next(logits[0, -1], ids + gen, temperature, 0, top_p,
                          rep_penalty)
        gen.append(nxt)
        if IM_END in tok.decode(gen) or nxt == EOS_ID:
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
          f"重复惩罚 {args.rep_penalty} | 单轮无状态", flush=True)

    if args.ask:
        for q in args.ask:
            out = reply(model, tok, q, args.max_new, args.temperature,
                        args.top_p, args.rep_penalty)
            print(f"\n问:{q}\n答:{out}", flush=True)
        return

    print("交互模式(每问独立、不带历史;空行退出)")
    while True:
        try:
            q = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        out = reply(model, tok, q, args.max_new, args.temperature,
                    args.top_p, args.rep_penalty)
        print(f"模型 > {out}")


if __name__ == "__main__":
    main()
