#!/bin/bash
# stage16 补充:格式混合训练 —— 能力会不会跨格式迁移?
#
# **背景**:上一步测出 stage16 的 92.6% 完全是格式绑定的 ——
# 训练只见过 `1 2 3 4 + 5 6 7 8 =`,换成 `加` → 0/800,不逐位空格 → 0.6%,
# 三加数 → 0/800。(见 results/eval_arith_variant_*.json)
#
# **这一轮问**:把 5 种格式混着训,能力会不会跨到**没见过的格式**上?
#   · 会跨    → 格式多样性是可设计的抓手 → 从零方案的语料设计有了明确原则
#   · 不跨    → 泛化有上限,得用更激进的手段
#
# 训练集:5 种格式各 12,000 条等量混合(60,000)
# 评测集:见过的 5 种(对照)+ **没见过的 4 种**(关键数据),各 800 题
#
# 判读(预先定):
#   见过的 5 种都应接近满分 —— 否则混合本身有害
#   **没见过的 4 种**才是这轮的核心数字:
#     明显高于 0%  → 迁移成立,泛化半径可以靠格式多样性撑开
#     接近 0%      → 每种格式各学各的,没有共享的计算核心
#
# 用法(Spark): nohup bash supervisor13.sh > /dev/null 2>&1 &
PY=~/llm_study/.venv/bin/python
cd ~/llm_study/transformer_study/stage16_reasoning
rm -f 16_MIXED.done
echo "[sup13] start $(date)" > sup13.log

$PY -u ../stage14_multiturn/train_mt.py \
    --base-ckpt ../stage11_datascale/ckpt_11_5_final.pt \
    --bpe ../stage11_datascale/cache_mm10g/bpe.json \
    --train-data mixed_train_nt.jsonl --test-data mixed_test_nt.jsonl \
    --steps 5000 --max-len 256 --ckpt-out ckpt_16_mixed.pt > sft_mixed.log 2>&1
echo "[sup13] train exit $? $(date)" >> sup13.log

for T in S1同款 S2汉字算符 S3问句 S4自然写法 S5三加数 U1加上 U2求和 U3冒号无等号 U4自然加汉字; do
  $PY -u eval_arith.py --ckpt ckpt_16_mixed.pt \
      --eval-file "mixed_eval_$T.jsonl" --label "mixed_$T" \
      --json-out "results/eval_mixed_$T.json" > "eval_mixed_$T.log" 2>&1
  echo "[sup13] eval $T exit $?" >> sup13.log
done

echo "[sup13] all done $(date)" >> sup13.log
echo DONE > 16_MIXED.done
