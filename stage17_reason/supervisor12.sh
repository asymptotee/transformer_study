#!/bin/bash
# stage17:能力边界扫描 —— 126M 除了算术,还能学会什么?
#
# **要回答的问题**:stage16 证明 126M 能用 14 分钟学会 5 位加法(0% → 83.5%)。
# 但加法是**一个窄算法**。"处理输入 / 逻辑思维"能不能也这么教出来?
#
# Holmes 的实测是**负面信号**(0.5B、4 倍参数、专为推理设计、100 倍数据,
# 关系推理 0/3、指令变换 2/4、形式逻辑 2/6)。所以先花 1 小时扫三个维度,
# **别在不知道边界的情况下开 30 小时的从零预训练。**
#
# 三个任务各训一个**独立**模型(不混),这样能分别回答"这个维度能不能学会":
#   ① relation   关系推理
#   ② transform  指令变换(倒序 / 删第 k 个)
#   ③ state      状态追踪(多步加减)
#
# 判读(预先定):
#   · 多个维度都能学会  → "处理输入"可训练 → 值得做正式模型(阶段 C)
#   · 只有部分能学会    → 边界画出来了,据此定范围
#   · 全都学不会        → 数据/格式还有问题,别急着训
#
# 每组:5,000 步(60,000 条 ÷ batch 16 = 3,750 步/epoch,≈1.3 epoch)
#       窗口 256(序列很短,基座原生),实测 ~14 分钟/组
#
# 用法(Spark): nohup bash supervisor12.sh > /dev/null 2>&1 &
PY=~/llm_study/.venv/bin/python
cd ~/llm_study/transformer_study/stage17_reason
rm -f 17_REASON.done
echo "[sup12] start $(date)" > sup12.log

# 0) 起点:基座对这三类任务(预期 ~0,这是对照零点)
for T in relation transform state; do
  $PY -u eval_reason.py --ckpt ../stage11_datascale/ckpt_11_5_final.pt \
      --task $T --label base_$T --json-out results/eval_base_$T.json \
      > eval_base_$T.log 2>&1
  echo "[sup12] baseline $T exit $?" >> sup12.log
done

# 1-3) 三个任务各训一个独立模型
for T in relation transform state; do
  $PY -u ../stage14_multiturn/train_mt.py \
      --base-ckpt ../stage11_datascale/ckpt_11_5_final.pt \
      --bpe ../stage11_datascale/cache_mm10g/bpe.json \
      --train-data ${T}_train_nt.jsonl --test-data ${T}_test_nt.jsonl \
      --steps 5000 --max-len 256 --ckpt-out ckpt_17_$T.pt > sft_$T.log 2>&1
  echo "[sup12] train $T exit $? $(date)" >> sup12.log
  $PY -u eval_reason.py --ckpt ckpt_17_$T.pt --task $T --label 17_$T \
      --json-out results/eval_17_$T.json > eval_$T.log 2>&1
  echo "[sup12] eval $T exit $?" >> sup12.log
done

echo "[sup12] all done $(date)" >> sup12.log
echo DONE > 17_REASON.done
