#!/bin/bash
# 11.5 过天链(~29h):从头训 186k 步(ratio 12,journal 口径见 stage11 README)
#   → harness 评估(val + bpb + 样例)→ 131 题(eval_v2)→ 标记
# 用法(Spark): nohup bash supervisor3.sh > /dev/null 2>&1 &
PY=~/llm_study/.venv/bin/python
cd ~/llm_study/transformer_study/stage11_datascale
rm -f 11_5_ALL.done
echo "[sup3] start $(date)" > sup3.log

# 1) 从头训:单条 cosine,186k 步 = 15.2 亿 token = ratio 12 = 0.76 epoch
$PY -u ../stage9_modern_gpt/train_large9.py \
    --cache-dir cache_mm10g --norm rms --ff silu --pos rope --rope-theta 1e6 \
    --rope-ctx 4096 --n-kv-heads 6 --d-ff 3072 --steps 186000 \
    --eval-every 2000 --ckpt ckpt_11_5_base.pt > train_11_5.log 2>&1
echo "[sup3] train exit $? $(date)" >> sup3.log

# 2) harness:稳定口径 val + bpb + 20 题锚点 + 样例
$PY -u ../stage9_modern_gpt/eval_harness.py --ckpt ckpt_11_5_base.pt \
    --model-module model_modern --bpe cache_mm10g/bpe.json \
    --tokens cache_mm10g/tokens.pt --val-batches 16 --label 11-5-base \
    --json-out results/11-5-base.json > eval_11_5.log 2>&1
echo "[sup3] harness exit $?" >> sup3.log

# 3) 131 题(双格式,真阳审计口径)
$PY -u ../stage13_eval/eval_v2.py --ckpt ckpt_11_5_base.pt \
    --bpe cache_mm10g/bpe.json --label 11_5_base \
    --json-out ../stage13_eval/results_v2/11_5_base.json > eval_v2_11_5.log 2>&1
echo "[sup3] eval_v2 exit $?" >> sup3.log

echo "[sup3] all done $(date)" >> sup3.log
echo DONE > 11_5_ALL.done
