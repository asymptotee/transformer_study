"""generate_cached.py —— KV cache 增量解码生成(9.3)

对照旧版(no-cache):旧版每生成一个 token 都要对**整个前缀**重算一遍
注意力(9.1 前 stage5 sampling.generate;模型内 GPT.generate 同理)。
但注意力的一个关键性质(见 CONCEPTS 十二节):**旧 token 的 K/V 不会变**
——第 t 步算过的 k_0..k_{t-1},第 t+1 步再用时是一样的。所以生成时
只算最后一个 token 的 Q,并复用(拼接)历史上每层的 K/V,省掉的是
重复计算 O(序列²) 中"对旧 token 重算注意力"的部分。

本文件与旧版生成**共用同一个采样器**(stage5 sampling.sample_next)——
温度/top-k/top-p/重复惩罚逐 token 行为完全一致,唯一差别是算力省在哪。
这也让"两路生成文本一致"成为 KV cache 正确性的最强自检(见 bench_generate)。

注意:只支持 pos=rope 的模型(9.3 起 cache 落在 rope 路径)。
"""

import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "stage4_scaling_bpe"))
sys.path.insert(0, str(REPO / "stage5_capstone"))
from sampling import sample_next                       # noqa: E402


@torch.no_grad()
def generate_cached_ids(model, tok, prompt, max_new, temperature=0.8,
                        top_k=0, top_p=0.9, rep_penalty=1.2, eos_id=None):
    """KV cache 生成:返回 token id 列表(不含 prompt)。

    prefill:把 prompt 整段前向一次,拿到每层初始 (k, v);
    之后每步只喂 1 个新 token,层内把新 K/V 拼到缓存尾。
    eos_id 给定时遇 EOS 提前停(默认 None = 与旧版行为一致,不停)。
    """
    assert model.cfg.pos == "rope", "KV cache 只在 rope 路径实现"
    device = next(model.parameters()).device
    ctx = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    logits, past = model.forward_cached(ctx, None)     # prefill
    full_ids = tok.encode(prompt)     # 重复惩罚作用的全量上下文(与旧版同源)
    generated = []
    for _ in range(max_new):
        if len(generated) >= model.cfg.rope_ctx - 1:   # 别超出旋转表
            break
        nxt = sample_next(logits[0, -1], full_ids, temperature,
                          top_k, top_p, rep_penalty)
        if eos_id is not None and nxt == eos_id:
            break
        generated.append(nxt)
        full_ids.append(nxt)
        tk = torch.tensor([[nxt]], dtype=torch.long, device=device)
        logits, past = model.forward_cached(tk, past)  # 增量步:1 token
    return generated


def generate_cached(model, tok, prompt, max_new, **kw):
    """与旧版 sampling.generate 同接口:返回完整文本(prompt + 续写)。"""
    ids = tok.encode(prompt)
    return tok.decode(ids + generate_cached_ids(model, tok, prompt, max_new, **kw))
