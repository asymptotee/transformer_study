#!/bin/bash
# stage15:工具调用 SFT(从 14_mt 起,窗口 1024)
#
# 依据(侦察 + 裸测,2026-09-20):
#   · 76,520 条 tool 对话,100% 单个 user 请求,但**可有 1-3 轮工具调用链**
#     (92% 单轮),81% 只有 1 个可用工具,11 种人造工具
#   · 渲染后中位 761 token → **必须 1024**:768 只活 53.7%,512 存活 0.0%
#   · 零样本基线(14_mt):发标签 15% / 参数正确 5% / 收尾(读结果答对)48%
#   · 无污染:预训练语料里 tool_response/get_exchange_rate 命中 0,
#     14_mt 的训练数据里 tool 命中 0
#
# 1024 是新外推(基座 256 训练,9.2 实测 768 可用且"SFT 教位置")——
# 所以链里第一件事就是看实际存活率,别训完才发现掉一半。
#
# 用法(Spark): nohup bash supervisor9.sh > /dev/null 2>&1 &
PY=~/llm_study/.venv/bin/python
cd ~/llm_study/transformer_study/stage15_toolcall
rm -f 15_TOOL.done
echo "[sup9] start $(date)" > sup9.log

# 1) 训练:40,000 条 ÷ batch 16 = 2,500 步 ≈ 1 epoch
$PY -u ../stage14_multiturn/train_mt.py \
    --base-ckpt ../stage14_multiturn/ckpt_14_mt.pt \
    --bpe ../stage11_datascale/cache_mm10g/bpe.json \
    --train-data tool_train_nt.jsonl --test-data tool_test_nt.jsonl \
    --steps 2500 --max-len 1024 --ckpt-out ckpt_15_tool.pt > sft_15_tool.log 2>&1
echo "[sup9] train exit $? $(date)" >> sup9.log
# ↑ 日志第二行的「样本: 训练 N | 测试 M」就是 1024 存活率,对比 40000/500

# 2) 工具调用评测(两阶段:会不会调用 / 读不读得懂结果)
$PY -u eval_toolcall.py --ckpt ckpt_15_tool.pt --label 15_tool --n 100 \
    --json-out results/eval_15_tool.json > eval_tool_15.log 2>&1
echo "[sup9] eval_toolcall exit $?" >> sup9.log

# 3) 三把尺子回归 —— 别把对话/知识训坏
$PY -u ../stage13_eval/eval_v2.py --ckpt ckpt_15_tool.pt \
    --bpe ../stage11_datascale/cache_mm10g/bpe.json --label 15_tool \
    --json-out ../stage13_eval/results_v2/15_tool.json > eval_v2_15.log 2>&1
echo "[sup9] eval_v2 exit $?" >> sup9.log

$PY -u ../stage13_eval/eval_usability.py --ckpt ckpt_15_tool.pt \
    --bpe ../stage11_datascale/cache_mm10g/bpe.json --label 15_tool \
    --json-out ../stage13_eval/results_u/15_tool.json > eval_use_15.log 2>&1
echo "[sup9] usability exit $?" >> sup9.log

$PY -u ../stage14_multiturn/eval_contamination.py --mode contam \
    --ckpt ckpt_15_tool.pt --bpe ../stage11_datascale/cache_mm10g/bpe.json \
    --baseline ../stage13_eval/results_v2/15_tool.json --label 15_tool \
    --json-out results/contam_15_tool.json > eval_contam_15.log 2>&1
echo "[sup9] contam exit $?" >> sup9.log

echo "[sup9] all done $(date)" >> sup9.log
echo DONE > 15_TOOL.done
