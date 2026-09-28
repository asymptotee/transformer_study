"""fix_eval_a.py —— 一次性修补:把评测文件里的答案字段去掉空格

**为什么**:`eval_arith.py` 的判据是

    re.sub(r"\\s+", "", 模型输出) == r["a"]

它**只归一化模型输出那一侧**,`r["a"]` 是原样比较的。而 `gen_arith_mixed.py`
第一版把答案写成了逐位空格的 `d(a+b)`,于是 `59+84=143` 这种**全对的答案**
被记成 0/800 —— 五个训练格式全部"归零",看起来像"混合训练把能力毁了"。

(生成器本身已经修好了,这个脚本只用来补救已经生成的评测文件。)
"""

import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
n = 0
for f in sorted(HERE.glob("mixed_eval_*.jsonl")):
    rows = [json.loads(l) for l in open(f, encoding="utf-8")]
    for r in rows:
        r["a"] = r["a"].replace(" ", "")
    with open(f, "w", encoding="utf-8") as g:
        for r in rows:
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
    n += 1
    print("  修好 %s" % f.name)
print("共 %d 个文件" % n)
r = json.loads(open(HERE / "mixed_eval_S1同款.jsonl", encoding="utf-8").readline())
print("抽查: q=%r a=%r  还有空格=%s" % (r["q"], r["a"], " " in r["a"]))
