"""make_corpus.py —— 生成 GPT 训练用的伪诗语料

为什么需要它：
  GPT（语言模型）要"学会写"，需要比阶段一 300 行大得多的连续文本。
  真实大语料不便离线获取，所以用程序生成上万行"伪诗"——有可学的结构
  （主题一致、前景→后情、五言/七言节奏），模型能从中学到字符搭配和行文节奏，
  续写时生成像样的句子。

  代码支持 --data 指向任何真实中文文本（见 train_lm.py），那会得到更地道的结果；
  本脚本提供的是一个**离线可复现、开箱即用**的默认语料。

语料结构：
  每行属一个主题（山/水/花/月/风/云），由"前景字"+"后情字"组成，
  随机为五言（3 前+2 后）或七言（4 前+3 后）。行与行用换行符隔开——
  换行符也是一个 token，GPT 借此学会"何时换行"。

运行：
  python make_corpus.py
生成 corpus.txt（约 12000 行）。
"""

import random
from pathlib import Path

HERE = Path(__file__).parent

# 6 主题，每个 8 前景字 + 8 后情字（沿用阶段二的字表，全部互不重复）
THEMES = {
    "山": {"前": "山峰岭崖岩峦嶂岫", "后": "高峻巍崇陡险峭拔"},
    "水": {"前": "水江河溪湖海涛波", "后": "清流深长阔浩渺茫"},
    "花": {"前": "花桃李杏梅兰菊荷", "后": "红香艳繁茂鲜娇嫩"},
    "月": {"前": "月轮魄影辉光晕芒", "后": "明圆皎洁亮晶莹闪"},
    "风": {"前": "风霜雨雪露电雷虹", "后": "急凉轻柔寒烈狂微"},
    "云": {"前": "云霞霭雾岚烟霏霙", "后": "白舒卷淡远飘浮散"},
}


def gen_line(rng: random.Random) -> str:
    """随机主题；五言（3前+2后）或七言（4前+3后）各占一半。"""
    name = rng.choice(list(THEMES))
    t = THEMES[name]
    if rng.random() < 0.5:
        front = "".join(rng.choice(t["前"]) for _ in range(3))
        back = "".join(rng.choice(t["后"]) for _ in range(2))
    else:
        front = "".join(rng.choice(t["前"]) for _ in range(4))
        back = "".join(rng.choice(t["后"]) for _ in range(3))
    return front + back


if __name__ == "__main__":
    rng = random.Random(2024)
    n = 12000
    lines = [gen_line(rng) for _ in range(n)]
    text = "\n".join(lines) + "\n"
    out = HERE / "corpus.txt"
    out.write_text(text, encoding="utf-8")

    n_chars = len(text)
    n_words = len({c for c in text})
    print(f"写入 {out.name}: {n} 行 | {n_chars:,} 字符（含换行）| 不同字符 {n_words} 种")
    print("\n样例:")
    for ln in lines[:10]:
        print(f"  {ln}")
