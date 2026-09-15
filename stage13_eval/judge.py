"""judge.py —— 13.1 的判分内核(纯 python,无 torch/模型依赖)

单独成文件的目的:**判分可以被离线复算**。历史评估的 JSON 里存了每题的
原始输出(out),判分规则一改,不用重跑 GPU,直接 rejudge.py 就能算出新口径
下的全套数字——这是"修正判分"这件事能被验证的前提。

三条规则(每条都对应一次真实踩坑):

  1. **归一化**:去空白/半全角逗号/顿号再匹配
     修 "8,848米" 这类"内容对、关键词没匹配上"的漏判(13.1)

  2. **回声截断**:截到第一个"自问自答"标记(问：/问题：/Q:/<|im_start|>…)之前
     修 "中国最长的城墙是秦始皇陵。问：长城的长度是多少？" 里
     "长城" 命中回声 —— 模型首答秦始皇陵(错),回声里才出现长城(假阳)
     同类:"中国的第一大城市是北京。…答：…是上海。"(11.5 审计发现)
     选它而不是"只看首句":首句规则会误伤真答案——重力加速度(先铺垫
     F=ma,下一句才说 9.8)、李白(首句生平,后文"被誉为诗仙")、
     《红楼梦》(后文"作者是清代作家曹雪芹")。实测首句规则 12 次评估掉
     15 个,审计后 8 个是误伤;回声截只掉 3 个,审计全是硬假阳。

  3. **题目回声降级**:命中片段±2 字的上下文若整体出现在题干里,记 suspect
     修 "秦始皇统一六国后…是秦" 里 "秦" 蹭中题干"秦始皇"这类(13.1)

**已知残留(记在案,未修)**:答案词只出现在"无标记的后续罗列"里时仍会
记真阳。样例:问"蝴蝶是由什么变成的",首答"由细胞和组织组成"(错),
后文罗列生命周期"卵、幼虫、蛹和成虫"里出现"幼虫"→ 记真阳。
量级:11_5_final/raw 的 76 个真阳里约 3 个属此类(含 2 个边缘)。
要修需引入"首答已给出不同答案则后续不算"的语义判断,收益小、误伤风险大。
"""

import re

NORM_RE = re.compile(r"[\s,，、·．。．]+")
# 自问自答/重播模板的开头:这之后的内容不是模型对本题的答案
ECHO_RE = re.compile(
    r"问：|问题：|问:|问题:|Q：|Q:|答：|答:|A：|A:"
    r"|<\|im_start\|>|<\|im_end\|>|User:|用户："
)


def norm(s):
    return NORM_RE.sub("", s).lower()


def answer_span(out):
    """模型真正作答的区间:截断到第一个回声标记之前(无标记则全文)。"""
    m = ECHO_RE.search(out)
    return out[:m.start()] if m else out


def judge(out, answers, question):
    """返回 (raw_hit, real_hit, suspect_hit)。

    raw    = 归一化后命中过答案词(不管在哪)
    real   = 命中且在作答区间内、且不是题目回声
    suspect= 只命中在题目回声里(不算真阳,也不单独计入)
    """
    o, q = norm(answer_span(out)), norm(question)
    raw = real = suspect = False
    for a in answers:
        a_n = norm(a)
        if not a_n:
            continue
        start = 0
        while True:
            i = o.find(a_n, start)
            if i < 0:
                break
            raw = True
            # 命中片段±2 字的上下文若整体出现在题干里 → 题目回声,算 suspect
            ctx = o[max(0, i - 2): i + len(a_n) + 2]
            if a_n in q and ctx in q:
                suspect = True
            else:
                real = True
            start = i + 1
    return raw, real, suspect
