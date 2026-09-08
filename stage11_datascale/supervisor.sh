#!/bin/bash
# 11.1 自驱链:等编码完成 -> 训练 10000 步 -> 两项评估 -> 汇总标记
# 用法(Spark): nohup bash supervisor.sh > /dev/null 2>&1 &
PY=~/llm_study/.venv/bin/python
cd ~/llm_study/transformer_study/stage11_datascale

# 1) 等编码(prepare_corpus.py 的 done 标记)
while ! test -f prep_ALL.done; do sleep 60; done
echo "[supervisor] 编码完成: $(cat prep_ALL.done)" > sup.log

# 2) 训练(94_full 配方,同 seed 42,10000 步)
$PY -u ../stage9_modern_gpt/train_large9.py \
    --cache-dir cache_mm --norm rms --ff silu --pos rope --rope-theta 1e6 \
    --rope-ctx 4096 --n-kv-heads 6 --d-ff 3072 --steps 10000 \
    --ckpt ckpt_11_mm.pt > train_11_mm.log 2>&1
echo "[supervisor] 训练退出码 $?" >> sup.log

# 3) 评估一:新语料内 val + 20 题 + 样例(目标域)
$PY -u ../stage9_modern_gpt/eval_harness.py --ckpt ckpt_11_mm.pt \
    --model-module model_modern --bpe cache_mm/bpe.json --tokens cache_mm/tokens.pt \
    --val-batches 16 --label 11-mm --json-out results/11-mm.json > eval_11_mm.log 2>&1

# 4) 评估二:旧 wiki val 迁移探针(harness 默认 stage7 缓存)
$PY -u ../stage9_modern_gpt/eval_harness.py --ckpt ckpt_11_mm.pt \
    --model-module model_modern --no-facts --no-samples --val-batches 16 \
    --label 11-mm-wikitransfer --json-out results/11-mm-wikitransfer.json > eval_11_wiki.log 2>&1

echo "[supervisor] 全部完成" >> sup.log
echo ALL-DONE > sup_ALL.done
