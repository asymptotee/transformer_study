"""clean_luxun.py —— 清洗 luxun.txt

网上下的 txt 常带杂质，这里处理：
  1. BOM 头（用 utf-8-sig 读取自动去掉）
  2. Windows 换行 CRLF -> LF
  3. 小说网站广告行（含 xiaoshuotxt / 小说天堂 / http / www 等的行整行删除）
  4. 连续多个空行压缩成一个

鲁迅正文里不会出现这些现代字符串，所以按子串过滤整行是安全的。
输出 luxun_clean.txt（UTF-8，无 BOM）。
"""

import re
from pathlib import Path

HERE = Path(__file__).parent

raw = (HERE / "luxun.txt").read_text(encoding="utf-8-sig")   # 自动去 BOM
raw = raw.replace("\r\n", "\n").replace("\r", "\n")

AD_PATTERNS = ["xiaoshuotxt", "小说天堂", "http", "www", "TXT小说", "＿小＿说", "书评"]
clean = [ln for ln in raw.split("\n") if not any(p in ln for p in AD_PATTERNS)]

text = "\n".join(clean)
text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"        # 压缩多余空行

out = HERE / "luxun_clean.txt"
out.write_text(text, encoding="utf-8")
print(f"原始:   {len(raw):,} 字符")
print(f"清洗后: {len(text):,} 字符  ->  {out.name}")
print(f"删除:   {len(raw) - len(text):,} 字符（广告/空行）")
print(f"不同字符: {len(set(text))} 种")
