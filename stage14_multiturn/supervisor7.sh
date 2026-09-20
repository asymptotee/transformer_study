#!/bin/bash
# 14.2 多轮 SFT 链:训练 → 131 题双格式回归 → 标记
#   判别式问题:**多轮能不能靠 SFT 补出来?**
#   数据:122,668 条纯多轮里随机取 40,000 条渲染(768 窗口下活 36,653)
#   配置:基座 ckpt_11_5_final.pt | 2000 步 ≈ 0.87 epoch | lr 1e-4 | seed 42
#   窗口 768(不是 512):依据见 stage9 README 9.2 更正块
# 用法(Spark): nohup bash supervisor7.sh > /dev/null 2>&1 &
PY=~/llm_study/.venv/bin/python
cd ~/llm_study/transformer_study/stage14_multiturn
rm -f 14_MT.done
echo "[sup7] start $(date)" > sup7.log

# 1) 多轮 SFT(每轮都算 loss,minimind 的做法)
$PY -u train_mt.py \
    --base-ckpt ../stage11_datascale/ckpt_11_5_final.pt \
    --bpe ../stage11_datascale/cache_mm10g/bpe.json \
    --train-data mt_train_nt.jsonl --test-data mt_test_nt.jsonl \
    --steps 2000 --max-len 768 > sft_14_mt.log 2>&1
echo "[sup7] sft exit $? $(date)" >> sup7.log

# 2) 131 题双格式回归(测多轮 SFT 对单轮的代价)
$PY -u ../stage13_eval/eval_v2.py --ckpt ckpt_14_mt.pt \
    --bpe ../stage11_datascale/cache_mm10g/bpe.json --label 14_mt \
    --json-out ../stage13_eval/results_v2/14_mt.json > eval_v2_14_mt.log 2>&1
echo "[sup7] eval_v2 exit $?" >> sup7.log

echo "[sup7] all done $(date)" >> sup7.log
echo DONE > 14_MT.done
