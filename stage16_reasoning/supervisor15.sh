#!/bin/bash
# stage16 补充:骨架数量 → 未见骨架迁移率(曲线)
#
# 问题:从零方案的语料要放多少种骨架,没见过的骨架才迁移?
#   曲线上翘、K 小 → 手写骨架够用,不用起 LLM 批量生成
#   一直平         → 骨架绑定是硬约束,方案要换
#
# 设计:gen_arith_bank.py 的 BANK 是**有序**的,N=6/15/30 取前缀 —— 唯一变量是骨架数。
#   BANK[0:6] = 原来的 T1–T6(逐字相同)→ N=6 那一点是上一轮的**复现**
#   HELD(10 个)永远不训
#   每骨架固定 1 万条、步数固定 5000 → 总数据量随 N 涨,所以**必须看"见过的骨架"当对照**
#
# 用法(Spark): nohup bash supervisor15.sh > /dev/null 2>&1 &
PY=~/llm_study/.venv/bin/python
cd ~/llm_study/transformer_study/stage16_reasoning
rm -f 16_BANK.done
echo "[sup15] start $(date)" > sup15.log
echo "[sup15] mem before: $(free -g | awk 'NR==2{print $3"used/"$2"total, avail "$7}')" >> sup15.log

for N in 6 15 30; do
  echo "=========== N=$N ===========" >> sup15.log
  $PY -u gen_arith_bank.py --n-skel $N --per-skel 10000 --n-eval 100 \
      > gen_bank_n$N.log 2>&1
  echo "[sup15] gen N=$N exit $?" >> sup15.log

  $PY -u ../stage14_multiturn/train_mt.py \
      --base-ckpt ../stage11_datascale/ckpt_11_5_final.pt \
      --bpe ../stage11_datascale/cache_mm10g/bpe.json \
      --train-data bank_train_n$N.jsonl --test-data bank_test_n$N.jsonl \
      --steps 5000 --max-len 320 --ckpt-out ckpt_16_bank_n$N.pt \
      > sft_bank_n$N.log 2>&1
  echo "[sup15] train N=$N exit $? $(date)  mem: $(free -g | awk 'NR==2{print $3"used, avail "$7}')" >> sup15.log
  grep -i "loss" sft_bank_n$N.log | tail -2 >> sup15.log

  # 评测:★ 未见 10 个(每次都要)+ 这次训过的 N 个(对照)
  for T in $(head -$N bank_eval_manifest.txt); do
    $PY -u eval_arith.py --ckpt ckpt_16_bank_n$N.pt \
        --eval-file "bank_eval_$T.jsonl" --label "n${N}_$T" \
        --json-out "results/bank_n${N}_$T.json" > /dev/null 2>&1 \
        || echo "[sup15] ✗ eval $T" >> sup15.log
  done
  for T in $(tail -10 bank_eval_manifest.txt); do
    $PY -u eval_arith.py --ckpt ckpt_16_bank_n$N.pt \
        --eval-file "bank_eval_$T.jsonl" --label "n${N}_$T" \
        --json-out "results/bank_n${N}_$T.json" > /dev/null 2>&1 \
        || echo "[sup15] ✗ eval $T" >> sup15.log
  done
  echo "[sup15] eval N=$N done $(date)" >> sup15.log
done

echo "[sup15] all done $(date)" >> sup15.log
echo DONE > 16_BANK.done
