"""hf_wrap.py —— stage12:把自研 GPT + BPE 包装成 HF 格式(供 lm_eval)

三件套:
  MiniGPTConfig(PretrainedConfig)      → config.json
  MiniGPTForCausalLM(PreTrainedModel)  → forward 输出 CausalLMOutputWithPast
  OurBPETokenizer(PreTrainedTokenizer) → 包装 bpe.py(慢速 tokenizer)+ chat_template

保存:`python hf_wrap.py --ckpt <ckpt.pt> --bpe <bpe.json> --out <dir>` →
权重 + config.json + tokenizer 文件 + bpe.json(原样拷贝,见下)。
加载:同进程先注册(见 check_parity.py / run_lm_eval.py),再 from_pretrained
——自带代码不需要 trust_remote_code。

已踩/预判的坑(包装期的关键决策):
  · 我们的 forward 不支持 padding → lm_eval 用 batch_size=1(attention_mask
    全 1 时忽略掩码)
  · 特殊 token:<pad>=0 <bos>=1 <eos>=2 <unk>=3;build_inputs_with_special_tokens
    原样返回(训练时没有 auto bos/eos,保持一致)
  · vocab.txt 逐行格式存不下控制字符 token(\n、\t、\x00…),所以用
    vocab_files_names 指向 bpe.json 原样拷贝 —— 这是本项目"含空格/控制符
    merge"与标准词表格式冲突的延续(见 stage11.2 的 Ġ 讨论)
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, PretrainedConfig,
                          PreTrainedModel, PreTrainedTokenizer)
from transformers.modeling_outputs import CausalLMOutputWithPast

HERE = Path(__file__).parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "stage9_modern_gpt"))
sys.path.insert(0, str(REPO / "stage4_scaling_bpe"))
from bpe import BPETokenizer, EOS_ID, PAD_ID, UNK_ID      # noqa: E402
from model_modern import GPT, GPTConfig                    # noqa: E402

DEFAULT_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


class MiniGPTConfig(PretrainedConfig):
    model_type = "minigpt"

    def __init__(self, vocab_size=26566, d_model=768, n_heads=12, n_layers=12,
                 d_ff=3072, dropout=0.1, max_len=512, pad_id=0, norm="rms",
                 ff="silu", pos="rope", rope_theta=1e6, rope_ctx=4096,
                 n_kv_heads=6, **kwargs):
        kwargs.setdefault("tie_word_embeddings", True)   # 我们的 GPT 是 tied
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.max_len = max_len
        self.pad_id = pad_id
        self.norm = norm
        self.ff = ff
        self.pos = pos
        self.rope_theta = rope_theta
        self.rope_ctx = rope_ctx
        self.n_kv_heads = n_kv_heads


class MiniGPTForCausalLM(PreTrainedModel):
    config_class = MiniGPTConfig
    # lm_head 与 embedding 绑定(同自研版):声明 tied key,保存时自动去重、
    # 加载时自动回绑;不声明的话 save_pretrained 会在 shared tensors 上报错
    # (transformers v5 要求 dict: {被绑定的 key: 源 key})
    _tied_weights_keys = {"model.lm_head.weight": "model.embedding.weight"}

    def _tie_weights(self):
        self.model.lm_head.weight = self.model.embedding.weight

    def __init__(self, config: MiniGPTConfig):
        super().__init__(config)
        gcfg = GPTConfig(vocab_size=config.vocab_size, d_model=config.d_model,
                         n_heads=config.n_heads, n_layers=config.n_layers,
                         d_ff=config.d_ff, dropout=config.dropout,
                         max_len=config.max_len, pad_id=config.pad_id,
                         norm=config.norm, ff=config.ff, pos=config.pos,
                         rope_theta=config.rope_theta, rope_ctx=config.rope_ctx,
                         n_kv_heads=config.n_kv_heads)
        self.model = GPT(gcfg)
        self.post_init()      # v5:post_init 负责设置 all_tied_weights_keys 等静态属性

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                **kwargs):
        logits = self.model(input_ids)
        loss = None
        if labels is not None:
            import torch.nn.functional as F
            x = logits[..., :-1, :].contiguous()
            y = labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1),
                                   ignore_index=-100)
        return CausalLMOutputWithPast(loss=loss, logits=logits)


class OurBPETokenizer(PreTrainedTokenizer):
    """包装自研 BPE 的慢速 tokenizer。词表原样用 bpe.json(见文件头说明)。"""

    vocab_files_names = {"bpe_file": "bpe.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, bpe_file=None, **kwargs):
        self._bpe = BPETokenizer.load(bpe_file)
        self._itos = self._bpe.itos
        self._stoi = self._bpe.stoi
        kwargs.setdefault("chat_template", DEFAULT_CHAT_TEMPLATE)
        # from_pretrained 会把 tokenizer_config.json 里的 special tokens 传进来,
        # 用 setdefault 避免与显式实参重复(v5 会报 multiple values)
        kwargs.setdefault("pad_token", "<pad>")
        kwargs.setdefault("bos_token", "<bos>")
        kwargs.setdefault("eos_token", "<eos>")
        kwargs.setdefault("unk_token", "<unk>")
        super().__init__(**kwargs)
        self.bpe_file = bpe_file

    @property
    def vocab_size(self):
        return len(self._itos)

    def get_vocab(self):
        return dict(self._stoi)

    def _tokenize(self, text):
        return [self._itos[i] for i in self._bpe.encode(text)]

    def _convert_token_to_id(self, token):
        return self._stoi.get(token, UNK_ID)

    def _convert_id_to_token(self, idx):
        return self._itos[idx]

    def convert_tokens_to_string(self, tokens):
        return self._bpe.decode([self._convert_token_to_id(t) for t in tokens])

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """训练未用 auto bos/eos —— 原样返回,保持推理与训练同分布。"""
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1

    def save_vocabulary(self, save_directory, filename_prefix=None):
        dst = os.path.join(save_directory, "bpe.json")
        shutil.copy(self.bpe_file, dst)
        return (dst,)


def register():
    """把自定义类注册进 Auto 工厂(同进程内 from_pretrained 即可)。"""
    from transformers import AutoConfig
    AutoConfig.register("minigpt", MiniGPTConfig)
    AutoModelForCausalLM.register(MiniGPTConfig, MiniGPTForCausalLM)
    AutoTokenizer.register(MiniGPTConfig, slow_tokenizer_class=OurBPETokenizer)


def save_pretrained_from_ckpt(ckpt_path, bpe_path, out_dir, half=False):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    gc = ckpt["config"]
    cfg = MiniGPTConfig(**{k: gc[k] for k in
                           ("vocab_size", "d_model", "n_heads", "n_layers",
                            "d_ff", "dropout", "max_len", "pad_id", "norm",
                            "ff", "pos", "rope_theta", "rope_ctx",
                            "n_kv_heads") if k in gc})
    model = MiniGPTForCausalLM(cfg)
    sd = {"model." + k: (v.half() if half else v)
          for k, v in ckpt["model"].items()}
    model.load_state_dict(sd)
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tok = OurBPETokenizer(bpe_file=bpe_path)
    tok.save_pretrained(out_dir)
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bpe", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = save_pretrained_from_ckpt(args.ckpt, args.bpe, args.out)
    files = sorted(os.listdir(out))
    print(f"saved → {out}")
    print("files:", files)


if __name__ == "__main__":
    main()
