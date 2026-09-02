"""build_corpus.py —— 把 zhwiki dump 提取成干净的语料（JSONL，每行一篇文章）

管线：
  zhwiki dump.bz2 → wikiextractor（去 wiki 语法、模板）→ 每行一篇 JSON
  → opencc 繁转简 → 过滤短文/列表页 → 累积到 --max-mb 就停 → corpus_wiki.jsonl

输出是"一篇文章一行"的 JSONL，而不是一个大文本——这样 train_large.py
可以逐篇 encode 并在篇与篇之间插入 <eos>，保留文档边界。

用法（在 DGX Spark 上跑）：
  python -u build_corpus.py \
      --dump data/zhwiki-20260801-p1.bz2 --max-mb 30
"""

import argparse
import json
import multiprocessing as mp
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from opencc import OpenCC

HERE = Path(__file__).parent

# 过滤：太短的文章没内容；含这些词的多是列表/消歧/文件页，不是正常文章
MIN_CHARS = 100
BAD_PATTERNS = ("列表", "消歧义", "维基百科:文件", "分类:", "模板:")


def extract_with_wikiextractor(dump_path, out_dir):
    """调 wikiextractor 把 bz2 dump 提取成 --json 格式（AA/wiki_00 等文件）。

    用 python -m 模块调用（不依赖 PATH），--no-templates 跳过模板展开省时间。
    """
    cmd = [sys.executable, "-m", "wikiextractor.WikiExtractor",
           "-o", str(out_dir), "--json", "--no-templates",
           "--processes", str(min(mp.cpu_count(), 16)), str(dump_path)]
    print("运行:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    out = []
    for f in sorted(out_dir.rglob("wiki_*")):
        out.append(f)
    return out


def iter_articles(extracted_files):
    """逐个文件产出 (title, text)。"""
    for f in extracted_files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                    yield obj.get("title", ""), obj.get("text", "")
                except json.JSONDecodeError:
                    continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="zhwiki pages-articles bz2 路径")
    ap.add_argument("--out", default=str(HERE / "corpus_wiki.jsonl"))
    ap.add_argument("--max-mb", type=float, default=30.0,
                    help="提取到正文累计多少 MB 就停（探路用小切片）")
    ap.add_argument("--keep-extracted", action="store_true",
                    help="保留 wikiextractor 中间输出（默认用完即删）")
    args = ap.parse_args()

    cc = OpenCC("t2s")   # 繁 → 简
    tmp = Path(tempfile.mkdtemp(prefix="wiki_extract_"))
    try:
        files = extract_with_wikiextractor(args.dump, tmp)
        print(f"提取完成: {len(files)} 个文件，开始清洗切片...")

        limit = int(args.max_mb * 1024 * 1024)
        written = 0
        n_articles = 0
        n_skip = 0
        with open(args.out, "w", encoding="utf-8") as out:
            for title, text in iter_articles(files):
                text = cc.convert(text).strip()
                if len(text) < MIN_CHARS or any(p in title for p in BAD_PATTERNS):
                    n_skip += 1
                    continue
                out.write(json.dumps({"title": title, "text": text},
                                     ensure_ascii=False) + "\n")
                written += len(text.encode("utf-8"))
                n_articles += 1
                if written >= limit:
                    break

        print(f"\n语料: {args.out}")
        print(f"文章: {n_articles:,} | 跳过: {n_skip:,} | 正文: {written/1024/1024:.1f} MB")
    finally:
        if not args.keep_extracted:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
