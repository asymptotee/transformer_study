#!/bin/bash
# 11.5-chat 剂量曲线:800(已有)/ 2000 / 5000 步
#   都从 ckpt_11_5_final.pt 起、同数据、同 seed、同 lr —— 唯一变量是步数。
#   动机:800 步只等于 0.4 epoch(32057/16=2004 步每 epoch),常规 SFT 跑 2-3
#   epoch,所以"是否欠训"是真问题。但加深的收益有上限:chat 的 53 条错题里
#   44 条基座 raw 也不会(知识缺口,SFT 无关),只有 9 条是"raw 会而 chat 不会"。
#   预登记预测:chat 在 2000 步附近见顶(78→80±3),raw 从 2000 步起单调下降。
# 用法(Spark): nohup bash supervisor6.sh > /dev/null 2>&1 &
PY=~/llm_study/.venv/bin/python
cd ~/llm_study/transformer_study/stage11_datascale
rm -f 11_5_CHATDOSE.done
echo "[sup6] start $(date)" > sup6.log

for N in 2000 5000; do
  echo "[sup6] --- ${N} 步开始 $(date)" >> sup6.log

  $PY -u ../stage10_preference/train_chat10.py \
      --base-ckpt ckpt_11_5_final.pt \
      --bpe cache_mm10g/bpe.json \
      --train-data ../stage10_preference/chat_train_nt.jsonl \
      --test-data ../stage10_preference/chat_test_nt.jsonl \
      --steps $N --ckpt-out ckpt_11_5_chat_${N}.pt > sft_11_5_${N}.log 2>&1
  echo "[sup6] sft ${N} exit $?" >> sup6.log

  $PY -u ../stage13_eval/eval_v2.py --ckpt ckpt_11_5_chat_${N}.pt \
      --bpe cache_mm10g/bpe.json --label 11_5_chat_${N} \
      --json-out ../stage13_eval/results_v2/11_5_chat_${N}.json \
      > eval_v2_11_5_chat_${N}.log 2>&1
  echo "[sup6] eval ${N} exit $?" >> sup6.log
done

echo "[sup6] all done $(date)" >> sup6.log
echo DONE > 11_5_CHATDOSE.done
