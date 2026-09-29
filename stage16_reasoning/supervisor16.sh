#!/bin/bash
# stage16 补充:N=30 加步数 —— 45.5% 是被训练量压着,还是被多样性压着?
#
# supervisor15 的结果:n30 的"见过的"只有 53.0%,T18带标签 7.8% / T13求和冒号 11.2%
# 一批训过的格式还很弱。**那个 45.5% 是在"一半骨架没学会"时拿到的。**
#
# 所以这一步只动一个量:**步数 5000 → 15000**(epoch 0.28 → 0.84),数据/骨架完全不变。
#
#   未见涨、见过也涨  → 还有空间,曲线没到顶,那时再谈 K 要多大、要不要 N=60
#   只有见过涨       → 那才是多样性的真天花板(而且 45.5% 要打折解读)
#
# ⚠️ 结果写成 bank_${LABEL}_*.json,**不能覆盖 bank_n30_*** —— 那是 5000 步的证据。
#
# 用法(Spark): nohup bash supervisor16.sh > /dev/null 2>&1 &
#   bash supervisor16.sh 15000 n30x3 30        # 步数 标签 训练骨架数
#   bash supervisor16.sh 15000 n6x3   6        # ★ 对照:骨架少也训够,看迁移到不到 78%
STEPS=${1:-15000}
LABEL=${2:-n30x3}
NSKEL=${3:-30}
PY=~/llm_study/.venv/bin/python
cd ~/llm_study/transformer_study/stage16_reasoning
rm -f 16_BANK_$LABEL.done
echo "[sup16] start $(date)  STEPS=$STEPS LABEL=$LABEL NSKEL=$NSKEL" > sup16_$LABEL.log
echo "[sup16] mem before: $(free -m | awk 'NR==2{printf "used %dGB / total %dGB, avail %dGB", $3/1024, $2/1024, $7/1024}')" >> sup16_$LABEL.log

for f in bank_train_n$NSKEL.jsonl bank_test_n$NSKEL.jsonl bank_eval_manifest.txt; do
  [ -s "$f" ] || { echo "[sup16] ✗ 缺 $f —— 先跑 supervisor15.sh" >> sup16_$LABEL.log; exit 1; }
done

$PY -u ../stage14_multiturn/train_mt.py \
    --base-ckpt ../stage11_datascale/ckpt_11_5_final.pt \
    --bpe ../stage11_datascale/cache_mm10g/bpe.json \
    --train-data bank_train_n$NSKEL.jsonl --test-data bank_test_n$NSKEL.jsonl \
    --steps $STEPS --max-len 320 --ckpt-out ckpt_16_bank_$LABEL.pt \
    > sft_bank_$LABEL.log 2>&1
echo "[sup16] train exit $? $(date)" >> sup16_$LABEL.log
grep -i "loss" sft_bank_$LABEL.log | tail -2 >> sup16_$LABEL.log
echo "[sup16] mem: $(free -m | awk 'NR==2{printf "avail %dGB", $7/1024}')" >> sup16_$LABEL.log

# 训过的 NSKEL 个(看有没有学会)+ ★ 未见 10 个(看迁移)—— 两次评测用同一批未见骨架
for T in $(head -$NSKEL bank_eval_manifest.txt) $(tail -10 bank_eval_manifest.txt); do
  $PY -u eval_arith.py --ckpt ckpt_16_bank_$LABEL.pt \
      --eval-file "bank_eval_$T.jsonl" --label "${LABEL}_$T" \
      --json-out "results/bank_${LABEL}_$T.json" > /dev/null 2>&1 \
      || echo "[sup16] ✗ eval $T" >> sup16_$LABEL.log
done
echo "[sup16] eval done $(date)" >> sup16_$LABEL.log

echo "[sup16] all done $(date)" >> sup16_$LABEL.log
echo DONE > 16_BANK_$LABEL.done
