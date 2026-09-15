#!/bin/bash
# 11.5 终态补齐链(~9h):从 130k 权重接余弦尾巴到 186k(lr 6.24e-5 → 0)
#   → harness 评估(终态 val + bpb)→ 131 题(eval_v2)→ 标记
# 背景:11.5 主跑因 4-batch val 噪声把 best-ckpt 锁在 130k,186k 终态丢失;
#       本链用 train_large9.py 新增的 --init-from/--start-step 做故障恢复。
# 用法(Spark): nohup bash supervisor4.sh > /dev/null 2>&1 &
PY=~/llm_study/.venv/bin/python
cd ~/llm_study/transformer_study/stage11_datascale
rm -f 11_5_FINAL.done
echo "[sup4] start $(date)" > sup4.log

# 1) 续训:130k → 186k(= 15.2 亿 token 整,ratio 12);终步无条件存盘
$PY -u ../stage9_modern_gpt/train_large9.py \
    --cache-dir cache_mm10g --norm rms --ff silu --pos rope --rope-theta 1e6 \
    --rope-ctx 4096 --n-kv-heads 6 --d-ff 3072 \
    --steps 186000 --start-step 130000 \
    --init-from ckpt_11_5_base.pt \
    --ckpt ckpt_11_5_tail.pt --save-final ckpt_11_5_final.pt \
    --eval-every 2000 > train_11_5_tail.log 2>&1
echo "[sup4] train exit $? $(date)" >> sup4.log

# 2) harness:终态稳定口径 val + bpb + 20 题锚点 + 样例
$PY -u ../stage9_modern_gpt/eval_harness.py --ckpt ckpt_11_5_final.pt \
    --model-module model_modern --bpe cache_mm10g/bpe.json \
    --tokens cache_mm10g/tokens.pt --val-batches 16 --label 11-5-final \
    --json-out results/11-5-final.json > eval_11_5_final.log 2>&1
echo "[sup4] harness exit $?" >> sup4.log

# 3) 131 题(双格式,真阳审计口径)
$PY -u ../stage13_eval/eval_v2.py --ckpt ckpt_11_5_final.pt \
    --bpe cache_mm10g/bpe.json --label 11_5_final \
    --json-out ../stage13_eval/results_v2/11_5_final.json > eval_v2_11_5_final.log 2>&1
echo "[sup4] eval_v2 exit $?" >> sup4.log

echo "[sup4] all done $(date)" >> sup4.log
echo DONE > 11_5_FINAL.done
