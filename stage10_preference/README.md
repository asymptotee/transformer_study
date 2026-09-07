# stage10_preference —— 10.1 DPO:偏好学习的最小完整闭环

看齐 minimind 的第二阶段(后训练:偏好学习,见根目录 ROADMAP.md)。10.1 的目标:
在现代基座(94_full)上把 **SFT → 造偏好数据 → DPO** 整条链路自己走通——
从 Bradley-Terry 推到 DPO 闭式解、自己写 loss、并在数值上与 minimind
`train_dpo.py` 对齐。**机制是重点,知识不是**(模型只有 44M token 的知识底子,
20 题答不对是预期内的,DPO 也不该改变它)。

## 链路与关键数字

### ① 现代基座 SFT(`train_sft10.py`)—— policy 起点 = 参考模型

stage8 的配方(stage8 的 QA 数据 + loss 掩码 + 全量微调)搬到现代基座 94_full:

| | 旧架构(stage8) | 现代基座(10.1) |
|---|---|---|
| 微调前 test loss | ~4.53 | 4.33 |
| 1500 步后 test loss | 3.87 | **3.629** |
| 维基 val(灾难性遗忘) | — | 3.776 → 4.250(量化:SFT 让域外分布掉了 0.47) |

### ② 自造偏好数据(`make_pref.py`)—— 规则裁判,不请人工

对 2000 条 qa_train prompt,policy(温度 0.9)各采 6 个回答,三条**可解释规则**
打分(0~3):实质长度(≥8 token)/ 不绕圈(不复读 prompt 尾)/ 不重复(唯一 token
占比)。chosen=最高分、rejected=最低分,分差为 0 则弃。
**成对 934 / 弃 1066(47%)**——弱模型多数样本同分,弃得干脆(数据过滤本身
也是 RLHF 的一课)。偏好方向:"答得实质、不绕、不重复",与真实 RLHF 的
"有用、不糊弄"同构。

### ③ DPO(`dpo.py` + `train_dpo10.py`)—— 推导与实现见 dpo.py 文件头注释

四步推导(Bradley-Terry → RLHF 闭式解 → 反解隐式奖励 → 代回消掉配分函数)
全部写在 dpo.py 的 docstring 里,结论只剩一行:

    L = −E[ log σ( β·( log π(y_w|x)/π_ref(y_w|x) − log π(y_l|x)/π_ref(y_l|x) ) ) ]

实现约定与 minimind `train_dpo.py` 逐行对应:policy/ref 同权重初始化(ref
冻结)、chosen 前半 batch / rejected 后半、β=0.15、逐 token logp 按 mask 求和。

| 指标 | 温和(1 epoch, lr 1e-6) | 激进(3 epoch, lr 3e-6) |
|---|---|---|
| dpo loss 轨迹 | 0.90 → 0.55 → 0.69(噪声,batch 4) | 0.69 → 0.45 ~ 0.83(更噪) |
| dev 隐式 margin(起点≈0) | **+0.30** | **+0.50** |
| 生成行为:平均规则分 | 2.87 → 2.90 | 2.87 → 2.90 |
| 平均回答长度 | 20.2 → 19.2 | 20.2 → 21.7 |
| 20 常识题 | 0/20(不变,预期内) | 0/20 |

### 行为为什么不怎么动?—— 规则天花板,不是机制失败

SFT 按这三条规则已经 **2.87/3**:它本来就有实质长度、不绕、不重复。偏好对的
余量只有 ~0.1,DPO 把分布推过去也就 +0.03。而 DPO 的**直接优化目标
(dev margin:0 → +0.30 → +0.50)显著上升**——机制在完整工作,只是"行为轴"
选的尺子太钝。教训:**自造偏好数据要选 SFT 明显做不好的维度**(真实 RLHF 的
偏好来自人类都觉得难的任务),否则会复现这种"指标饱和"。这是造数据时就要
想清楚的,不是训练后能补救的。

### 两个诚实标注

1. **"KL~ −1.5"不是 KL**:dev 指标里我用 chosen 上的 (π−ref) logp 差当 KL 的
   单样本估计,符号为负说明 policy 对 chosen 的**绝对** logp 略降于 ref
   (~0.04/token)——DPO 保证的是**相对**偏好(chosen 压过 rejected),不保证
   绝对提升,这是该损失的已知性质;真 KL 要用 π 采样估计,此处只是粗略代理。
2. **20 题 0/20 依旧**:DPO 不该也不会上知识(偏好数据里没有知识信号)。
   stage8 的"记忆会、检索不会"在偏好阶段继续成立——知识与偏好在两套数据里。

## minimind 对照笔记(`trainer/train_dpo.py`)

| 它的实现 | 我们的 | 差异说明 |
|---|---|---|
| loss 公式与 batch 布局(chosen 前半/后半) | 逐行对应 | align_dpo.py 在**同一份 logps 上验证 Δ=0.00** |
| π=ref 起点 loss = log 2 | 同 | 机制自检:实测 0.69314718,精确命中 |
| β 默认 0.15 | 同 | — |
| lr 4e-8(注释:建议 ≤5e-8 防遗忘) | 1e-6 / 3e-6 | 真实偏好信号弱,必须极小 lr;我们的规则信号强、样本少,给大留余地;观察:激进档 margin 更大但 loss 更噪 |
| DPODataset:chat 模板 + 掩码含 eos 段 | build_batch:raw 格式 + 掩码自"答：" 起 | 对齐脚本用**它的数据约定**跑我们的公式,约定差异不构成分歧 |
| ref = init_model(同权重) 冻结 | 同 | — |
| 每步重算 ref logp | 同 | 数据/ref 固定时可预计算缓存(大语料优化点,未做) |

## 踩坑记录

1. **新脚本必须先 rsync 再远程启动**(make_pref 首跑找不到文件,白等一轮)
2. minimind 仓库同步时把 `dataset/` 目录整体 exclude 了——那是 `lm_dataset.py`
   的代码目录不是数据目录;排除规则要按文件类型不按目录名
3. **造数据前先看 SFT 的行为分布**:规则打分器做完才发现 SFT 已在天花板。
   流程上应该先采样 50 条看分分布,再决定规则与样本量
4. DPODataset 依赖 HF `datasets` 库;对齐脚本可用 `CUDA_VISIBLE_DEVICES=""`
   强制 CPU 跑(负载小),不干扰 GPU 上的采样
5. DPO loss 轨迹在 batch=4 时很噪(0.4~0.9 抖动),读趋势要看 dev margin,
   不看单步 loss

## 文件

| 文件 | 内容 |
|---|---|
| `train_sft10.py` | 现代基座 SFT(DPO 的起点/ref) |
| `make_pref.py` | 采样 + 规则裁判造偏好对(934 对) |
| `dpo.py` | 推导文档 + loss 实现 + build_batch |
| `train_dpo10.py` | DPO 训练(温和/激进两档) |
| `align_dpo.py` | **数值对齐:与 minimind dpo_loss Δ=0.00,π=ref 时 = log2** |
| `eval_pref10.py` | held-out 行为评估(规则分/长度/空答率) |
| `results/10-sft.json / 10-dpo.json` | harness 16-batch + 20 题明细 |
| ckpt(不入库) | `ckpt_10_sft.pt` / `ckpt_10_dpo.pt` / `ckpt_10_dpo_agg.pt` |

## 10.2 GRPO:在线策略 RL vs DPO 的同一规则对决

**目的**:10.1 学会了"从离线偏好对里反向推出奖励";10.2 学会"直接在线追逐
奖励,还要管住自己别跑飞"。机制推导(advantage 归一化 → PPO-clip → KL 刹车)
写在 `grpo.py` 文件头;**实验设计吸收 10.1 的教训**——先校准规则找余量,
再让 DPO 与 GRPO 在**同一份规则奖励**上各跑一遍,最后 held-out 同裁判打分。

### 规则校准(教训应用:这次有余量)

连续奖励 `rule_reward`:长度带 8~40(+1/过短 −1/过长 −0.5)+ EOS 收尾 +0.5
− 重复惩罚(连续)− 回声惩罚(连续)。校准采样 100 条:SFT 基线均值 ≈ +1.0
(上限 2.0),**30% 样本长度带外、~25% 有回声/重复尾部**——对比 10.1 造完
才发现 2.87/3 的天花板,这次先测后造。

### 训练与结果

- GRPO 温和(100 步, lr 1e-6):reward 前/后 +0.92→+0.98,长度 24.4→20.7
  (往带内收),KL 全程 ~0.09-0.13 受控
- GRPO-strong(450 步, lr 3e-6, K=8):reward 稳定 ~+1.1
- DPO 同规则对照:离线把同一 reward 做成偏好对(497 对/500),走 10.1 的 DPO

**held-out 对比(qa_test 120 题 × 3 采样,同一 rule_reward 裁判)**:

| 模型 | 规则分 | 重复率 | 解读 |
|---|---|---|---|
| SFT 基线 | +0.889 | 0.066 | 参照 |
| DPO(10.1 旧规则) | +0.857 | 0.069 | 旧规则不迁移到新裁判(规则即分布) |
| GRPO 温和 | +0.966 | 0.057 | 在线有效 |
| GRPO-strong | +0.979 | 0.059 | 更强剂量 ≈ 温和(收益饱和) |
| **DPO(同规则)** | **+1.010** | 0.059 | 离线同规则也有效 |

结论:
1. **同一奖励,两个优化器都能把行为推高 ~+0.1**(相对 SFT 约 2.5σ);
   DPO-rule 与 GRPO-strong 差 ~0.03,在采样噪声内——这个小规则上离线/在线
   打平,差别不在最终分,在**过程**(见下)
2. **无刷分翻车**:重复率没升反降(0.066→0.057~0.059)、长度未跑飞——KL 刹车
   + 规则里带重复惩罚的双保险在工作;RL 最经典的失败模式(奖励黑客:复读刷
   长度)在 117.8M 上也没有出现,值得记
3. 10.1 的 DPO 用旧规则(实质长度/不绕圈/不重复的离散版)在新连续裁判下
   **不迁移**(+0.857 ≈ 基线)——偏好规则是分布,换裁判要重新造对

### 机制观察(读 minimind `train_grpo.py` 的发现,已写入实现)

- **minimind 每步只对采样批更新一遍**:此时 π_new == π_old(采样与更新之间
  权重没变),ratio ≡ 1,clip 与 CISPO 的 clamp **结构上空转**——PPO 式 clip
  的价值只在**同一批数据多次复用**时体现。我们的 trainer 加 inner_epochs=2,
  让 ratio 真的偏离 1、clip 真正绑定;单遍时 loss 退化为无 clip 的策略梯度。
  (数值对齐脚本里两式都验证,不影响公式正确性)
- KL 项用 exp(δ)−δ−1(δ = log ref − log π),对"π 跑得离 ref 太远"双向惩罚;
  全程 KL ~0.1/token 说明 β=0.1 把策略摁在 SFT 附近
- 组内 advantage 归一化(减均值除 std)让不同 prompt 的绝对分数尺度对齐——
  K=8 组内 adv std ≈ 1.0,信号干净

### 对齐自检(align_grpo.py,结果同 10.1 风格)

同一批合成张量(覆盖:prompt 长不一、有/无 EOS 的行、padding、截断)喂我们
的 grpo 管线与 minimind train_grpo.py 的公式(逐字移植其 logp_pos /
completion_mask / kl_div / ratio-min-clip 管线):

    our grpo loss      = 38.69918823
    minimind grpo loss = 38.69918823
    Δ = 0.00e+00  一致 ✓   (掩码逐位 Δ = 0)

机制对齐完成:公式、掩码、advantage 布局三方一致(索引陷阱由合成批覆盖)。

### 文件

| 文件 | 内容 |
|---|---|
| `grpo.py` | 推导 + advantage/kl/loss + rule_reward(校准版) |
| `train_grpo10.py` | 在线训练(rollout → 奖励 → inner 更新) |
| `make_pref_rule.py` | 同规则的离线 DPO 对照对 |
| `eval_grpo10.py` | held-out 同裁判对比 |
| ckpt(不入库) | `ckpt_10_grpo.pt` / `ckpt_10_grpo_strong.pt` / `ckpt_10_dpo_rule.pt` |

### 踩坑记录

1. **奖励函数吃到了 pad token**:rollout 返回补零矩阵,直接用整行打分 →
   重复率爆炸、reward 从 +1.0 假摔到 −1.0。**规则打分永远用未 padding 的
   原始序列**(和 DPO 的 logp 掩码是两个不同的坑)
2. 每步 24~32 样本的 reward 单步噪声 ±0.8:看趋势要用 5-10 步的滑动窗口或
   分段均值,单步数字没有意义
3. pkill 远程训练进程时 ssh 会报 exit 255(它杀的是自己会话树里的进程),
   别误判成失败——用 ps + etime 确认

## 下一步

10.3 SFT 格式看齐(可选):chat template / system / think 那套,为把知识
SFT + 偏好链路搬到"问答格式"做准备——或直接进阶段 11 可选池(MoE /
蒸馏 / HF 生态化)。ROADMAP 主线到此为止的"看齐 minimind 后训练"已闭环:
SFT(行为)→ DPO/GRPO(偏好)两条路都通了,且都在数值上与 minimind 对齐过。
见 ROADMAP。
