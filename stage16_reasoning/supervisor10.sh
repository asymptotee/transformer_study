#!/bin/bash
# stage16:合成算术 —— 分清"数据瓶颈"还是"容量瓶颈"
#
# 问题:我们的模型只会检索,不会处理输入(12×8=40 不是算错,是检索了一个
# 看起来合理的数)。要逼它*处理*,训练分布里答案必须**查不到** —— 所以用
# 程序化生成的算术(每条输入的组合都是新的)。
#
# 两组对照,只为回答一个问题:**分词不对齐扣了多少分**
#   A. 对齐版   1 2 3 4 + 5 6 7 8 = 6 9 1 2   (逐位加空格,token=位)
#   B. 对照组   1234 + 5678 = 6912             (现有乱切格式)
#
# 判读(预先定好):
#   · 低档会、高档卡在某个位数  → **容量/深度瓶颈** → 加大加深有价值
#   · 全档都学不会              → 数据/方法问题 → 加大模型也白搭
#   · 全档都学会                → 目标可以定更高(乘法/多步)
#   · B 明显差于 A              → 分词对齐本身就是个待修的坑(可能正是
#                                 现有模型"有外壳没内核"的原因之一)
#
# 窗口 256:算术序列很短(5位题约 30 token),基座原生就是 256,不用外推。
# 300,000 条 ÷ batch 16 = 18,750 步/epoch,跑 5,000 步 ≈ 0.27 epoch。
#
# 用法(Spark): nohup bash supervisor10.sh > /dev/null 2>&1 &
PY=~/llm_study/.venv/bin/python
cd ~/llm_study/transformer_study/stage16_reasoning
rm -f 16_ARITH.done
echo "[sup10] start $(date)" > sup10.log

# 0) 先量起点:基座对算术是完全不会(预期 ~0),这才是对照的零点
$PY -u eval_arith.py --ckpt ../stage11_datascale/ckpt_11_5_final.pt \
    --label base_11_5_final --json-out results/eval_base.json > eval_base.log 2>&1
echo "[sup10] baseline exit $?" >> sup10.log

# 1) A 组:对齐版
$PY -u ../stage14_multiturn/train_mt.py \
    --base-ckpt ../stage11_datascale/ckpt_11_5_final.pt \
    --bpe ../stage11_datascale/cache_mm10g/bpe.json \
    --train-data arith_train_nt.jsonl --test-data arith_test_nt.jsonl \
    --steps 5000 --max-len 256 --ckpt-out ckpt_16_arith.pt > sft_A.log 2>&1
echo "[sup10] train A exit $? $(date)" >> sup10.log
$PY -u eval_arith.py --ckpt ckpt_16_arith.pt --label 16_arith_aligned \
    --json-out results/eval_16_arith.json > eval_A.log 2>&1
echo "[sup10] eval A exit $?" >> sup10.log

# 2) B 组:对照组(不加逐位空格)
$PY -u ../stage14_multiturn/train_mt.py \
    --base-ckpt ../stage11_datascale/ckpt_11_5_final.pt \
    --bpe ../stage11_datascale/cache_mm10g/bpe.json \
    --train-data arith_train_nt_un.jsonl --test-data arith_test_nt_un.jsonl \
    --steps 5000 --max-len 256 --ckpt-out ckpt_16_arith_un.pt > sft_B.log 2>&1
echo "[sup10] train B exit $? $(date)" >> sup10.log
$PY -u eval_arith.py --ckpt ckpt_16_arith_un.pt --label 16_arith_unaligned \
    --eval-file arith_eval_un.jsonl \
    --json-out results/eval_16_arith_un.json > eval_B.log 2>&1
echo "[sup10] eval B exit $?" >> sup10.log

echo "[sup10] all done $(date)" >> sup10.log
echo DONE > 16_ARITH.done
