"""gen_corpus.py —— 生成阶段二的结构化伪诗语料

为什么需要合成语料：
  阶段一的 complete（续写）用 300 行真诗，模型只能背、不能泛化。
  本阶段要演示"数据多到背不动 → 被迫学规律 → 泛化"。为此语料需要：
    1. 有可学结构（泛化才可能发生）
    2. 组合空间远大于训练量（几千行背不完，且验证集能取到真·未见组合）
    3. 结构不太容易（小数才可能走"背诵"捷径，对比才鲜明）

伪诗的规则（可学的"语法"）：
  每行 8 字 = 前 4 字（前景）+ 后 4 字（后情），整行属于同一主题。
  后情字由前景字按两条规则推出：
    (a) 镜像：后情第 i 字 对应 前景第 3-i 字（位置倒序，像 reverse 的反对角线）
    (b) 对仗：每个前景字有固定的"对仗字"（山↔高、峰↔峻……）
  例：前景 山峰岭崖 → 后情 崇巍峻高（崖→崇、岭→巍、峰→峻、山→高，倒序+替换）

  组合空间：6 主题 × 8^4 = 24576 种前景，远大于 8000 行训练量，
  所以训练集只覆盖一小部分前景，验证集可以取到大量"没见过的前景"——
  在没见过的前景上续对，才算真泛化。

运行：
  python gen_corpus.py
生成：corpus_small.txt / corpus_large.txt / corpus_val.txt
"""

import random
from pathlib import Path

HERE = Path(__file__).parent

# 6 个主题，每个 8 个"前景字" + 8 个"后情字"，全部 96 字互不重复（有断言保证）。
THEMES = {
    "山": {"前": "山峰岭崖岩峦嶂岫", "后": "高峻巍崇陡险峭拔"},
    "水": {"前": "水江河溪湖海涛波", "后": "清流深长阔浩渺茫"},
    "花": {"前": "花桃李杏梅兰菊荷", "后": "红香艳繁茂鲜娇嫩"},
    "月": {"前": "月轮魄影辉光晕芒", "后": "明圆皎洁亮晶莹闪"},
    "风": {"前": "风霜雨雪露电雷虹", "后": "急凉轻柔寒烈狂微"},
    "云": {"前": "云霞霭雾岚烟霏霙", "后": "白舒卷淡远飘浮散"},
}

# ---------- 校验：所有字必须唯一 ----------
_all_front = "".join(t["前"] for t in THEMES.values())
_all_back = "".join(t["后"] for t in THEMES.values())
ALL_CHARS = _all_front + _all_back
assert len(_all_front) == len(set(_all_front)) == 48, f"前景字重复: {len(_all_front)}"
assert len(_all_back) == len(set(_all_back)) == 48, f"后情字重复: {len(_all_back)}"
assert not (set(_all_front) & set(_all_back)), "前景字和后情字有重叠"

# 对仗表：前景字 -> 后情字（按主题内序号一一对应）
PARTNER = {}
for t in THEMES.values():
    for f, b in zip(t["前"], t["后"]):
        PARTNER[f] = b


def gen_line(rng: random.Random) -> str:
    """前景 4 字随机选；后情 = 前景倒序后逐字替换成对仗字。"""
    name = rng.choice(list(THEMES))
    front = [rng.choice(THEMES[name]["前"]) for _ in range(4)]
    back = [PARTNER[front[3 - i]] for i in range(4)]   # 镜像 + 对仗
    return "".join(front) + "".join(back)


def gen_lines(n: int, seed: int) -> list:
    rng = random.Random(seed)
    return [gen_line(rng) for _ in range(n)]


def write(path: Path, lines: list):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  写入 {path.name}: {len(lines)} 行")


if __name__ == "__main__":
    n_front = len(THEMES) * 8 ** 4
    print(f"主题数: {len(THEMES)} | 总字数: {len(ALL_CHARS)} "
          f"(48 前景 + 48 后情) | 词表(含特殊符): {len(ALL_CHARS) + 4}")
    print(f"前景组合空间: {len(THEMES)}×8^4 = {n_front} 种（远大于训练量）")
    print("规则: 后情 = 前景倒序 + 逐字对仗替换\n")

    small = gen_lines(300, seed=1)     # 小语料：背得下 → 记忆
    large = gen_lines(8000, seed=2)    # 大语料：背不动 → 学规律 → 泛化

    # 验证集：严格按"没见过的前景"过滤（行由前景唯一确定，前景没见过的才是真·未见）
    train_fronts = {ln[:4] for ln in small} | {ln[:4] for ln in large}
    val_pool = gen_lines(5000, seed=3)
    val, seen = [], set()
    for ln in val_pool:
        f = ln[:4]
        if f not in train_fronts and f not in seen:
            val.append(ln); seen.add(f)
        if len(val) >= 600:
            break

    write(HERE / "corpus_small.txt", small)
    write(HERE / "corpus_large.txt", large)
    write(HERE / "corpus_val.txt", val)
    print(f"  验证集前景全部未在训练集出现 ✓")

    print("\n样例（前 4 字 = 前景，后 4 字 = 后情 = 前景倒序+对仗）:")
    for ln in small[:8]:
        print(f"  {ln[:4]} → {ln[4:]}")
