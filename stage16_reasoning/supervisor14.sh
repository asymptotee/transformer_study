#!/bin/bash
# stage16 补充:结构多样性能不能泛化?(见 gen_arith_struct.py 的说明)
#
# 上一轮:5 种格式混合 → 换词能过(U1 84%)、换结构过不去(U2 3.6% / U3 0%)。
#         但那一轮 5 种格式的**骨架相同**(都是 `X <算子> Y =`)。
#
# 这一轮:训练里放 **6 种骨架完全不同**的格式,考 **4 种没见过的骨架**。
#   能迁移 → 语料放十几种结构就够(方案简单)
#   不能   → 得枚举目标场景的说法(劳动密集,方案完全不同)
#
# 用法(Spark): nohup bash supervisor14.sh > /dev/null 2>&1 &
PY=~/llm_study/.venv/bin/python
cd ~/llm_study/transformer_study/stage16_reasoning
rm -f 16_STRUCT.done
echo "[sup14] start $(date)" > sup14.log

$PY -u ../stage14_multiturn/train_mt.py \
    --base-ckpt ../stage11_datascale/ckpt_11_5_final.pt \
    --bpe ../stage11_datascale/cache_mm10g/bpe.json \
    --train-data struct_train_nt.jsonl --test-data struct_test_nt.jsonl \
    --steps 5000 --max-len 320 --ckpt-out ckpt_16_struct.pt > sft_struct.log 2>&1
echo "[sup14] train exit $? $(date)" >> sup14.log

echo "--- 见过的 6 种(对照)---" >> sup14.log
for T in T1纯符号 T2无等号 T3求和 T4疑问 T5自然写法 T6应用题; do
  $PY -u eval_arith.py --ckpt ckpt_16_struct.pt \
      --eval-file "struct_eval_$T.jsonl" --label "struct_$T" \
      --json-out "results/eval_struct_$T.json" > "eval_struct_$T.log" 2>&1
  echo "[sup14] eval $T exit $?" >> sup14.log
done
echo "--- ★ 没见过的 4 种 ---" >> sup14.log
for T in V1新叙事 V2倒装 V3祈使 V4括号; do
  $PY -u eval_arith.py --ckpt ckpt_16_struct.pt \
      --eval-file "struct_eval_$T.jsonl" --label "struct_$T" \
      --json-out "results/eval_struct_$T.json" > "eval_struct_$T.log" 2>&1
  echo "[sup14] eval $T exit $?" >> sup14.log
done

echo "[sup14] all done $(date)" >> sup14.log
echo DONE > 16_STRUCT.done
