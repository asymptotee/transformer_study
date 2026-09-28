#!/bin/bash
# stage16 回归:那 5,000 步纯算术训练,把基座的其他能力破坏了没有?
#
# 为什么要跑这个:stage15 刚教过 —— 工具数据 2,500 步就把对话能力打崩了
# (知识 chat 74→12、收尾率 94.7%→44.3%)。而我们在 stage16 训了 5,000 步
# **纯算术**,却没测代价。这个数直接决定一个战略问题:
#
#   · 破坏了  → "后加的不同类数据会覆盖主线"成立 → **从零预训练(在预训练阶段
#               就混入合成数据)才有理由**,因为只有那样才能让两种能力共存
#   · 没破坏  → 算术和对话不冲突 → 继续训练完全够用,29 小时的从零不划算
#
# 对照基线(基座 ckpt_11_5_final,已有记录):
#   knowledge raw 74 / chat 29   |   usability 84%(假高,中位只 12 token)
#
# 用法(Spark): nohup bash supervisor11.sh > /dev/null 2>&1 &
PY=~/llm_study/.venv/bin/python
cd ~/llm_study/transformer_study/stage16_reasoning
rm -f 16_REG.done
echo "[sup11] start $(date)" > sup11.log

# 三把尺子,与 stage13/stage14 完全同口径
$PY -u ../stage13_eval/eval_v2.py --ckpt ckpt_16_arith.pt \
    --bpe ../stage11_datascale/cache_mm10g/bpe.json --label 16_arith \
    --json-out ../stage13_eval/results_v2/16_arith.json > reg_v2.log 2>&1
echo "[sup11] eval_v2 exit $?" >> sup11.log

$PY -u ../stage13_eval/eval_usability.py --ckpt ckpt_16_arith.pt \
    --bpe ../stage11_datascale/cache_mm10g/bpe.json --label 16_arith \
    --json-out ../stage13_eval/results_u/16_arith.json > reg_use.log 2>&1
echo "[sup11] usability exit $?" >> sup11.log

$PY -u ../stage14_multiturn/eval_contamination.py --mode contam \
    --ckpt ckpt_16_arith.pt --bpe ../stage11_datascale/cache_mm10g/bpe.json \
    --baseline ../stage13_eval/results_v2/16_arith.json --label 16_arith \
    --json-out results/contam_16_arith.json > reg_contam.log 2>&1
echo "[sup11] contam exit $?" >> sup11.log

echo "[sup11] all done $(date)" >> sup11.log
echo DONE > 16_REG.done
