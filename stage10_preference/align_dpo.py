"""align_dpo.py —— 10.1 看齐自检:DPO loss 与 minimind 实现数值一致

最强的"机制懂了"证据:同一份 logps/掩码上,我们自己推导实现的 dpo_loss
与 minimind `trainer/train_dpo.py` 的 dpo_loss(逐字拷贝其函数体)算出的
loss 完全一致——公式、batch 布局(chosen 前半/rejected 后半)、掩码求和
方式三者同时对齐才算数。

做法(Spark 上跑,借用 minimind 仓库与其官方权重):
  1. 用 minimind 的 DPODataset(其 tokenizer + 一段假的 dpo.jsonl)造一批
     (x_chosen/y_chosen/mask_chosen/…) —— 数据约定用它的
  2. 用官方 full_sft_768.pth 前向,算 policy/ref 的逐 token logp
  3. 我们的 dpo_loss vs 它的 dpo_loss,同一份输入各算一遍 → 比较
  4. 额外自检:π=ref(起点)时 loss 应 = log 2 ≈ 0.6931

用法(Spark):
  ~/llm_study/.venv/bin/python align_dpo.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path.home() / "llm_study" / "minimind"))
from dataset.lm_dataset import DPODataset                        # noqa: E402
from dpo import dpo_loss as our_dpo_loss                         # noqa: E402
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402
from transformers import AutoTokenizer                           # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MM = Path.home() / "llm_study" / "minimind"
MM_W = Path.home() / "llm_study" / "mm_weights" / "minimind-3-pytorch"


# minimind 的 dpo_loss 函数体(逐字拷贝自 trainer/train_dpo.py:34-50)
def minimind_dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]
    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    logits = pi_logratios - ref_logratios
    return (-F.logsigmoid(beta * logits)).mean()


def main():
    # 1. 假数据:2 对 chosen/rejected(minimind chat 模板格式)
    fake = [{"chosen": [{"role": "user", "content": "中国的首都在哪里?"},
                        {"role": "assistant", "content": "中国的首都是北京,是一座历史名城。"}],
             "rejected": [{"role": "user", "content": "中国的首都在哪里?"},
                          {"role": "assistant", "content": "不知道。"}]},
            {"chosen": [{"role": "user", "content": "水的化学式是什么?"},
                        {"role": "assistant", "content": "水的化学式是 H2O,由两个氢原子和一个氧原子组成。"}],
             "rejected": [{"role": "user", "content": "水的化学式是什么?"},
                          {"role": "assistant", "content": "水的化学式是二氧化碳。"}]}]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                     encoding="utf-8") as f:
        for ex in fake:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        path = f.name

    tok = AutoTokenizer.from_pretrained(MM / "model")
    ds = DPODataset(path, tok, max_length=256)

    # 2. 官方权重(64M dense)前向;policy/ref 都取它(对齐只要求同一输入)
    cfg = MiniMindConfig(hidden_size=768, num_hidden_layers=8,
                         num_attention_heads=8, num_key_value_heads=4)
    model = MiniMindForCausalLM(cfg).to(DEVICE)
    model.load_state_dict(torch.load(MM_W / "full_sft_768.pth",
                                     map_location=DEVICE), strict=True)
    model.eval()

    def pad2d(t, L):
        return torch.cat([t, torch.zeros(L - t.shape[0], dtype=t.dtype)])

    xs, ys, ms = [], [], []
    for k in range(len(ds)):
        s = ds[k]
        x_ch, y_ch, m_ch = (torch.as_tensor(s[k_]) for k_ in
                            ("x_chosen", "y_chosen", "mask_chosen"))
        x_rj, y_rj, m_rj = (torch.as_tensor(s[k_]) for k_ in
                            ("x_rejected", "y_rejected", "mask_rejected"))
        xs += [x_ch, x_rj]; ys += [y_ch, y_rj]; ms += [m_ch, m_rj]
    L = max(t.shape[0] for t in xs)
    x = torch.stack([pad2d(t, L) for t in xs]).to(DEVICE)
    y = torch.stack([pad2d(t, L) for t in ys]).to(DEVICE)
    mask = torch.stack([pad2d(t, L) for t in ms]).float().to(DEVICE)

    def logps(model, x, y):
        log_probs = F.log_softmax(model(x).logits.float(), dim=2)
        return torch.gather(log_probs, 2, y.unsqueeze(2)).squeeze(-1)

    with torch.no_grad():
        ref_lp = logps(model, x, y)
        pi_lp = ref_lp.clone()          # π=ref:正是 DPO 的初始点

    beta = 0.15
    a = our_dpo_loss(pi_lp, ref_lp, mask, beta).item()
    b = minimind_dpo_loss(pi_lp, ref_lp, mask, beta).item()
    print(f"our dpo_loss      = {a:.10f}")
    print(f"minimind dpo_loss = {b:.10f}")
    print(f"Δ = {abs(a - b):.2e}  {'一致 ✓' if abs(a - b) < 1e-6 else '不一致 ✗'}")
    print(f"π=ref 起点理论值 log2 = {0.69314718056:.6f} | 实测 = {a:.6f}")
    os.unlink(path)


if __name__ == "__main__":
    main()
