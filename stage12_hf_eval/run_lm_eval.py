"""run_lm_eval.py —— stage12:用 lm-evaluation-harness 跑标准选择题评测

与 minimind README「客观评测」同一套任务:
  中文:ceval-valid、cmmlu      英文:arc_easy、piqa、openbookqa、hellaswag、social_iqa
方法:选择题 logprob 法(比较候选项的条件对数概率,取最大)——判分无歧义。

用法(Spark;首次跑数据集要联网,用 hf-mirror):
  HF_ENDPOINT=https://hf-mirror.com ~/llm_study/.venv/bin/python run_lm_eval.py \
      --hf-dir hf_11_4_chat --tasks ceval-valid --limit 20        # 冒烟
  HF_ENDPOINT=https://hf-mirror.com ~/llm_study/.venv/bin/python run_lm_eval.py \
      --hf-dir hf_11_4_chat --tasks ceval-valid,cmmlu --apply-chat-template \
      --out results_lmeval/11_4_chat_zh.json

说明:
  · --apply-chat-template:指令模型用(把题干按 chat 模板渲染),对齐 minimind
    的命令;base 模型不加
  · batch_size 默认 1:我们的 forward 不支持 padding(包装期决策)
  · 本进程内注册 MiniGPT 类,from_pretrained 走本地注册表,不需要
    trust_remote_code
"""

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import hf_wrap                                              # noqa: E402
hf_wrap.register()

import lm_eval                                              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--tasks", default="ceval-valid")
    ap.add_argument("--limit", type=int, default=None, help="每任务抽样数(冒烟用)")
    ap.add_argument("--apply-chat-template", action="store_true")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", help="float32/float16")
    ap.add_argument("--out", default=None, help="结果 JSON 保存路径")
    args = ap.parse_args()

    if "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    model_args = {"pretrained": str(Path(args.hf_dir).resolve()),
                  "dtype": args.dtype}

    # lm_eval 0.4.x:apply_chat_template 是 simple_evaluate 的顶层参数
    # (评测器层把题干按 chat 模板渲染);放进 model_args 会被透传到
    # from_pretrained → 模型构造函数,报 unexpected kwarg(stage12 踩过)
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=[t.strip() for t in args.tasks.split(",")],
        batch_size=args.batch_size,
        limit=args.limit,
        device=args.device,
        apply_chat_template=args.apply_chat_template,
    )

    print("\n===== 结果 =====")
    rows = []
    for task, m in sorted(results["results"].items()):
        acc = m.get("acc,none", m.get("acc_norm,none"))
        rows.append({"task": task, "acc": acc})
        print(f"  {task:20s} acc = {acc:.4f}" if acc is not None
              else f"  {task:20s} {m}")
    avg = sum(r["acc"] for r in rows if r["acc"] is not None) / max(len(rows), 1)
    print(f"  {'平均':20s} acc = {avg:.4f}")
    print(f"(随机基线:4 选项 ≈0.25,2 选项 ≈0.50;对照 minimind-3 64M:"
          f"ceval 24.89 / cmmlu 25.38)")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"config": vars(args), "results": results["results"]},
                      f, ensure_ascii=False, indent=2, default=str)
        print(f"JSON → {out}")


if __name__ == "__main__":
    main()
