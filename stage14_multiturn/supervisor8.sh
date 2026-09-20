#!/bin/bash
# 14.5 混训版:50:50 单轮+多轮 → 三把尺子(知识 / 可用性 / 多轮)
#   比例依据:自然比例(22% 多轮)实测只能把污染率从 63.9% 降到 53.8%,
#   远不如纯多轮的 8.1% —— 剂量非线性,要明显加重多轮。
#   步数 5100 ≈ 1 epoch(81,660 条 ÷ batch 16 = 5104 步;SFT 在 1 epoch 饱和)
# 用法(Spark): nohup bash supervisor8.sh > /dev/null 2>&1 &
PY=~/llm_study/.venv/bin/python
cd ~/llm_study/transformer_study/stage14_multiturn
rm -f 14_MIX.done
echo "[sup8] start $(date)" > sup8.log

# 1) 混训
$PY -u train_mt.py \
    --base-ckpt ../stage11_datascale/ckpt_11_5_final.pt \
    --bpe ../stage11_datascale/cache_mm10g/bpe.json \
    --train-data mix_train_nt.jsonl --test-data mix_test_nt.jsonl \
    --steps 5100 --max-len 768 --ckpt-out ckpt_14_mix.pt > sft_14_mix.log 2>&1
echo "[sup8] train exit $? $(date)" >> sup8.log

# 2) 尺子①:131 题双格式(知识)
$PY -u ../stage13_eval/eval_v2.py --ckpt ckpt_14_mix.pt \
    --bpe ../stage11_datascale/cache_mm10g/bpe.json --label 14_mix \
    --json-out ../stage13_eval/results_v2/14_mix.json > eval_v2_14_mix.log 2>&1
echo "[sup8] eval_v2 exit $?" >> sup8.log

# 3) 尺子②:可用性(自然收尾率 + 长度)
$PY -u ../stage13_eval/eval_usability.py --ckpt ckpt_14_mix.pt \
    --bpe ../stage11_datascale/cache_mm10g/bpe.json --label 14_mix \
    --json-out ../stage13_eval/results_u/14_mix.json > eval_use_14_mix.log 2>&1
echo "[sup8] usability exit $?" >> sup8.log

# 4) 尺子③:多轮 —— 污染 + 历史回忆(基线用刚出的单轮结果)
$PY -u eval_contamination.py --mode contam --ckpt ckpt_14_mix.pt \
    --bpe ../stage11_datascale/cache_mm10g/bpe.json \
    --baseline ../stage13_eval/results_v2/14_mix.json --label 14_mix \
    --json-out results/contam_14_mix.json > eval_contam_14_mix.log 2>&1
echo "[sup8] contam exit $?" >> sup8.log

$PY -u eval_contamination.py --mode recall --ckpt ckpt_14_mix.pt \
    --bpe ../stage11_datascale/cache_mm10g/bpe.json \
    --baseline ../stage13_eval/results_v2/14_mix.json --label 14_mix \
    --json-out results/recall_14_mix.json > eval_recall_14_mix.log 2>&1
echo "[sup8] recall exit $?" >> sup8.log

echo "[sup8] all done $(date)" >> sup8.log
echo DONE > 14_MIX.done
