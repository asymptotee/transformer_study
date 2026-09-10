#!/bin/bash
# 11.2 过夜自驱链:下载 10GB 主线语料 -> BPE+编码 -> 长训练 -> 语料内评估
# 用法(Spark): nohup bash supervisor2.sh > /dev/null 2>&1 &
PY=~/llm_study/.venv/bin/python
cd ~/llm_study/transformer_study/stage11_datascale
STEPS=${STEPS:-20000}

echo "[sup2] start $(date)" > sup2.log

# 1) 下载主线语料(10GB);modelscope 必须用 CLI 二进制(python -m 无 __main__)
mkdir -p ~/llm_study/mm_data
if [ ! -s ~/llm_study/mm_data/pretrain_t2t.jsonl ]; then
  (cd ~/llm_study/mm_data && ~/llm_study/.venv/bin/modelscope download \
      --dataset gongjy/minimind_dataset --include pretrain_t2t.jsonl \
      --local_dir . > dl10g.log 2>&1)
  echo "[sup2] download exit $?" >> sup2.log
fi

# 2) 编码:同一份 26,566 词表,Rust 引擎执行(已验证与 python 逐位一致,326×)
#    python 侧解码保持一致;旧 python 编码分片已作废,先清空
rm -f cache_mm10g/tokens.pt cache_mm10g/shards/shard_*.pt
$PY -u engine_swap.py --encode > prep10g.log 2>&1
echo "[sup2] prep exit $? $(tail -1 prep10g.log)" >> sup2.log

# 3) 长训练(同配方;20000 步 ≈ 3h;数据 2.1B token ≈ 0.08 epoch——剂量实验)
$PY -u ../stage9_modern_gpt/train_large9.py \
    --cache-dir cache_mm10g --norm rms --ff silu --pos rope --rope-theta 1e6 \
    --rope-ctx 4096 --n-kv-heads 6 --d-ff 3072 --steps $STEPS \
    --ckpt ckpt_12_10g.pt > train_12_10g.log 2>&1
echo "[sup2] train exit $? ($STEPS steps)" >> sup2.log

# 4) 语料内评估(新词表 val + 20 题 + 样例)
$PY -u ../stage9_modern_gpt/eval_harness.py --ckpt ckpt_12_10g.pt \
    --model-module model_modern --bpe cache_mm10g/bpe.json --tokens cache_mm10g/tokens.pt \
    --val-batches 16 --label 12-10g --json-out results/12-10g.json > eval_12_10g.log 2>&1
echo "[sup2] eval exit $?" >> sup2.log
echo "[sup2] all done $(date)" >> sup2.log
echo ALL-DONE > sup2_ALL.done
