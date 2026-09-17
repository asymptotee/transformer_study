#!/bin/bash
# 11.5-chat:在 11.5 final(186k)基座上重跑 chat-SFT
#   数据/steps/lr/batch/seed 与 11.4-chat **逐字相同**(见 ckpt 元信息:step=800,
#   chat_of=ckpt_11_4_base.pt),唯一变量是基座 → 与 11_4_chat 严格可比。
#   数据用 nt(去 think)版:README 的 "12-chat-nt = 基座 + 800 步" + 时间戳佐证。
#   注意 train_chat10.py 的默认 --bpe 指 cache_mm/、默认数据是带 think 版,
#   这三项必须显式覆盖,否则不是同一个实验。
# 用法(Spark): nohup bash supervisor5.sh > /dev/null 2>&1 &
PY=~/llm_study/.venv/bin/python
cd ~/llm_study/transformer_study/stage11_datascale
rm -f 11_5_CHAT.done
echo "[sup5] start $(date)" > sup5.log

# 1) chat SFT
$PY -u ../stage10_preference/train_chat10.py \
    --base-ckpt ckpt_11_5_final.pt \
    --bpe cache_mm10g/bpe.json \
    --train-data ../stage10_preference/chat_train_nt.jsonl \
    --test-data ../stage10_preference/chat_test_nt.jsonl \
    --steps 800 --ckpt-out ckpt_11_5_chat.pt > sft_11_5.log 2>&1
echo "[sup5] sft exit $? $(date)" >> sup5.log

# 2) 131 题双格式:chat 是主指标,raw 用来看格式迁移代价
$PY -u ../stage13_eval/eval_v2.py --ckpt ckpt_11_5_chat.pt \
    --bpe cache_mm10g/bpe.json --label 11_5_chat \
    --json-out ../stage13_eval/results_v2/11_5_chat.json > eval_v2_11_5_chat.log 2>&1
echo "[sup5] eval_v2 exit $?" >> sup5.log

echo "[sup5] all done $(date)" >> sup5.log
echo DONE > 11_5_CHAT.done
