"""sampling.py —— 改进的解码策略：温度 + top-k + top-p(核采样) + 重复惩罚

阶段三/四的 generate 只有"温度 + top-k"，容易陷入重复循环
（"我们也不知道我们也不知道……"）。这里补上两个真实 LLM 常用的技巧：

  · top-p（核采样 / nucleus sampling）：
      把候选按概率从高到低排，只保留累积概率刚好达到 p 的最小集合，其余砍掉。
      和 top-k 的区别：top-k 固定保留 k 个；top-p 动态——分布集中时候选少，
      分布平坦时候选多，更自适应。

  · 重复惩罚（repetition penalty）：
      把已生成过的 token 的 logits 压低（正的除以 penalty，负的乘以 penalty），
      让模型不愿重复用词，专治循环。penalty 常用 1.1~1.3。
"""

import torch
import torch.nn.functional as F


def apply_repetition_penalty(logits, generated_ids, penalty):
    """对已生成的 token 施加惩罚（penalty>1 时压低其分数）。"""
    if penalty == 1.0:
        return logits
    for tok in set(generated_ids):
        if logits[tok] > 0:
            logits[tok] /= penalty
        else:
            logits[tok] *= penalty
    return logits


def top_k_top_p_filter(logits, top_k, top_p):
    """先 top-k 再 top-p，砍掉低概率长尾。logits 为一维 (vocab,)。"""
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[-1]] = float("-inf")
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        remove = cum > top_p
        remove[0] = False                    # 至少保留概率最高的一个
        sorted_logits[remove] = float("-inf")
        logits[sorted_idx] = sorted_logits   # 按原顺序放回
    return logits


def sample_next(logits, generated_ids, temperature=0.8, top_k=0, top_p=0.9,
                rep_penalty=1.2):
    """从最后一个位置的 logits 采样下一个 token。"""
    logits = logits.clone()
    apply_repetition_penalty(logits, generated_ids, rep_penalty)
    if temperature <= 0:
        return int(logits.argmax().item())
    logits = logits / temperature
    top_k_top_p_filter(logits, top_k, top_p)
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


def generate(model, tok, prompt, max_new, temperature=0.8, top_k=0, top_p=0.9,
             rep_penalty=1.2):
    """自回归续写，用上面的改进采样。"""
    generated = tok.encode(prompt)
    max_len = model.cfg.max_len
    device = next(model.parameters()).device   # ctx 建在模型所在设备
    for _ in range(max_new):
        ctx = torch.tensor([generated[-max_len:]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(ctx)[0, -1]       # 最后位置的 logits, (vocab,)
        nxt = sample_next(logits, generated, temperature, top_k, top_p, rep_penalty)
        generated.append(nxt)
    return tok.decode(generated)
