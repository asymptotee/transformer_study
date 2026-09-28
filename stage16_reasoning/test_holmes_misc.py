"""test_holmes_misc.py —— 除算术外,Holmes v1 在其他维度上的表现

**为什么测这些**:算术已经量过了(3/60)。但"推理能力"不止算术 —— 我们想知道
它到底是"算术不行"还是"处理输入这件事整个不行"。

**题目的挑选标准**(和 stage16 的算术一致):**必须真的处理输入,不能靠背**。
所以优先选"变换/推理"类,而不是"事实检索"类。

六类:
  ① 形式逻辑    条件推理四形式 + 三段论 —— 答案有客观标准
  ② 关系推理    谁最高/最矮/最大
  ③ 状态追踪    一系列操作后的最终状态
  ④ 指令变换    改疑问句 / 倒序 / 删字 —— 最能区分"处理"和"检索"
  ⑤ 代码        Python 的求值结果
  ⑥ 中文理解    数数、指代

**判读方式**:大部分打印出来眼看(这类题的"对不对"人一看就知道,而它们
五花八门的表述不适合写死自动判据)。少数几个答案唯一的,我在题面里标了期望值。

⚠️ 它会复读,所以 max_new_tokens 给 256 就够,而且**只看开头的回答**。

用法(Spark):
  PYTHONPATH=/home/zhangxu/llm_study/tf450 ~/llm_study/.venv/bin/python test_holmes_misc.py
"""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = Path.home() / "llm_study" / "holmes_v1" / "model"

# (类别, 期望, 问题)。期望为 None 表示只看输出。
Q = [
    # ① 形式逻辑 —— 条件推理四形式。前两行的"有效/无效"是有客观答案的
    ("形式逻辑", "有效", "如果下雨，地面就会湿。现在下雨了。所以地面是湿的。这个推理有效吗？"),
    ("形式逻辑", "无效", "如果下雨，地面就会湿。地面湿了。所以下雨了。这个推理有效吗？"),
    ("形式逻辑", "无效", "如果下雨，地面就会湿。现在没下雨。所以地面不湿。这个推理有效吗？"),
    ("形式逻辑", "有效", "如果下雨，地面就会湿。地面没湿。所以没下雨。这个推理有效吗？"),
    ("形式逻辑", "有效", "所有的玫瑰都是花。所有的花都需要水。所以所有的玫瑰都需要水。这个推理有效吗？"),
    ("形式逻辑", "无效", "所有的狗都喜欢吃肉。这只猫喜欢吃肉。所以这只猫是狗。这个推理有效吗？"),
    # ② 关系推理
    ("关系推理", "王五", "张三比李四高，李四比王五高。三个人里谁最矮？"),
    ("关系推理", "C", "A 比 B 大，C 比 A 大。谁最大？"),
    ("关系推理", "小美", "小美站在小刚前面，小刚站在小强前面。谁在最前面？"),
    # ③ 状态追踪
    ("状态追踪", "10", "篮子里有 5 个苹果，拿走 2 个，又放进去 7 个。现在篮子里有几个苹果？"),
    ("状态追踪", "2 1", "有一串数字 1 2 3。先把它倒过来，再删掉第一个数。剩下什么？"),
    ("状态追踪", "红", "灯一开始是绿的。按一次开关变黄，再按一次变红。按两次后是什么颜色？"),
    # ④ 指令变换 —— 这类最能区分"处理"和"检索"
    ("指令变换", "今天天气很好吗？", "把这句话改成疑问句：今天天气很好。"),
    ("指令变换", "橘子 香蕉 苹果", "把这三个词倒过来说：苹果 香蕉 橘子"),
    ("指令变换", "这个苹果大甜美。", "把下面句子里的所有“很”字去掉：这个苹果很大很甜很美。"),
    ("指令变换", "我不喜欢吃苹果。", "把“我喜欢吃苹果”改成否定句。"),
    # ⑤ 代码
    ("代码", "3", "Python 里 print(len([1,2,3])) 的输出是什么？"),
    ("代码", "8", "Python 里 print(2 ** 3) 的输出是什么？"),
    # ⑥ 中文理解
    ("中文理解", "4", "“我买了苹果、香蕉、橘子和葡萄”这句话里一共提到了几种水果？"),
    ("中文理解", "2", "“三加二等于几”这句话里有几个汉字数字？"),
]


def main():
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), torch_dtype="auto", trust_remote_code=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev).eval()

    cur = None
    for cat, want, q in Q:
        if cat != cur:
            cur = cat
            print("\n" + "#" * 74 + f"\n### {cat}\n" + "#" * 74, flush=True)
        text = tok.apply_chat_template([{"role": "user", "content": q}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=256, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id,
                                 eos_token_id=tok.eos_token_id)
        ans = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        ans = ans[:420].replace("\n", " ")
        print(f"\n问: {q}")
        if want:
            print(f"期望: {want}")
        print(f"答: {ans}" + ("…" if len(ans) >= 420 else ""), flush=True)


if __name__ == "__main__":
    main()
