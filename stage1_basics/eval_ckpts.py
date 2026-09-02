"""一次性评估脚本：对两个已训练 checkpoint 跑定量指标 + 生成样例"""
import torch

from data import (BOS_ID, EOS_ID, CharTokenizer, Seq2SeqDataset, collate,
                  load_corpus, make_pairs, random_lines, split_train_val)
from model import Transformer, TransformerConfig
from train import evaluate
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)


def load_model(path):
    ckpt = torch.load(path, map_location=DEVICE)
    cfg = TransformerConfig(**ckpt["config"])
    model = Transformer(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = CharTokenizer.from_vocab(ckpt["vocab"])
    return model, tok, ckpt["task"], ckpt["config"]


def generate(model, tok, prompt, max_new=64):
    src = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=DEVICE)
    out = model.generate(src, bos_id=BOS_ID, eos_id=EOS_ID,
                         max_new_tokens=max_new, temperature=0.0)
    return tok.decode(out[0].tolist())


def build_loader(pairs, tok, max_len, batch_size=64):
    ds = Seq2SeqDataset(pairs, tok, max_len)
    return DataLoader(ds, batch_size=batch_size, collate_fn=collate)


for path in ["ckpt.pt", "ckpt_complete.pt"]:
    print("=" * 70)
    print(f"checkpoint: {path}")
    model, tok, task, cfg = load_model(path)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"task={task} | vocab={len(tok)} | params={n_params:,} | device={DEVICE}")

    lines = load_corpus("corpus.txt")
    train_lines, val_lines = split_train_val(lines)
    max_len = cfg["max_len"]

    val_pairs = make_pairs(val_lines, task)
    val_loss, val_acc, val_exact = evaluate(model, build_loader(val_pairs, tok, max_len), DEVICE)
    print(f"[验证集 {len(val_pairs)} 条真实语料] loss={val_loss:.4f} "
          f"token_acc={val_acc:.3f} exact_match={val_exact:.3f}")

    if task == "complete":
        tr_pairs = make_pairs(train_lines, task)
        tr_loss, tr_acc, tr_exact = evaluate(model, build_loader(tr_pairs, tok, max_len), DEVICE)
        print(f"[训练集 {len(tr_pairs)} 条真实语料] loss={tr_loss:.4f} "
              f"token_acc={tr_acc:.3f} exact_match={tr_exact:.3f}")
        prompts = ["床前", "春眠", "白日", "红豆", "欲穷"]
        print("[生成样例 complete]")
        for p in prompts:
            ref = next((l for l in lines if l.startswith(p)), "")
            print(f"  {p!r} -> {generate(model, tok, p)!r}   (原句: {ref})")
    else:
        chars = tok.itos[4:]
        syn = random_lines(chars, 500, max_len=max_len - 2)
        s_loss, s_acc, s_exact = evaluate(model, build_loader(make_pairs(syn, task), tok, max_len), DEVICE)
        print(f"[合成随机串 500 条]          loss={s_loss:.4f} "
              f"token_acc={s_acc:.3f} exact_match={s_exact:.3f}")
        print("[生成样例 reverse]")
        for p in ["一二三四五", "床前明月光", "测试泛化新字符串", "abc"]:
            got = generate(model, tok, p)
            exp = p[::-1]
            mark = "OK" if got == exp else "MISS"
            print(f"  {p!r} -> {got!r}  期望 {exp!r}  [{mark}]")
    print()
