#!/bin/bash
# memwatch.sh —— 训练期间的内存监视器(每 30 秒一行)
#
# 为什么需要:2026-09-20 那次"机器卡死"事后才查出是内存耗尽 ——
# DGX Spark 是**统一内存**,加载进 GPU 的模型**不计入进程 RSS**,
# 所以 `ps` 只看得到 4.5 GB,实际被 llama-server 占着 62.5 GB。
# 事后翻 journal 才找到 OOM 记录,现场曲线没有。
#
# 现在每行记:时间 | 系统已用/可用 | 显存进程数与合计
# 撞车时能直接看出"是谁在涨、涨到多少"。
L=~/llm_study/transformer_study/stage15_toolcall/memwatch.log
echo "# $(date '+%F %T') 开始监视" > $L
while true; do
  read -r tot used avail < <(free -m | awk '/^内存|^Mem/{print $2/1024, $3/1024, $7/1024}')
  gpu=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1; n++} END{printf "%.1fG(%d个)", s/1024, n}')
  printf "%s 系统 用%.0fG/可用%.0fG | 显存 %s\n" "$(date '+%H:%M:%S')" "$used" "$avail" "$gpu" >> $L
  sleep 30
done
