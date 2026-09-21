"""chat15.py —— 工具调用对话工具

和 `chat11.py`(单轮)、`chat14.py`(多轮)一个路子,区别是**带工具定义前言**
和**工具结果模拟** —— 没有后者链条走不下去(模型发完调用就等着,没人回它)。

为什么需要:测 `ckpt_15_tool.pt` 不能用 chat14.py,它不渲染工具前言,而
**没有前言的输入对这个模型是分布外的**(实测会起手输出 `<tool` 然后崩成下划线)。

**工具结果是模拟的,不是真的** —— 那 11 个工具本就是人造的
(`generate_image` / `web_search` 之类没法真跑)。能真算的(`calculate_math`、
`text_length`、`unit_converter`、`random_number`、`get_current_time`、
`get_exchange_rate`)就真算,其余的返回一个占位符。

用法(Spark):
  ~/llm_study/.venv/bin/python chat15.py                      # 交互
  ~/llm_study/.venv/bin/python chat15.py --ask "100千克等于多少磅？"
  ~/llm_study/.venv/bin/python chat15.py --ask "北京天气怎么样？" --ask "那适合出门吗？"
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "stage4_scaling_bpe"))
sys.path.insert(0, str(REPO / "stage9_modern_gpt"))
sys.path.insert(0, str(Path.home() / "llm_study" / "minimind"))
from bpe import BPETokenizer, EOS_ID                # noqa: E402
from model_modern import GPT, GPTConfig             # noqa: E402
from transformers import AutoTokenizer              # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IM_END = "<|im_end|>"
THINK = re.compile(r"<think>.*?</think>", re.S)
TOOLCALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)

# 11 个工具的定义(与语料一致;系统前言由 chat_template 的 tools= 参数渲染)
TOOLS = [
    ("get_exchange_rate", "查询两种货币之间的实时汇率",
     {"from_currency": "string", "to_currency": "string"}),
    ("random_number", "生成指定范围内的随机数", {"min": "integer", "max": "integer"}),
    ("unit_converter", "进行单位换算,支持长度、重量、温度等",
     {"value": "number", "from_unit": "string", "to_unit": "string"}),
    ("calculate_math", "计算数学表达式", {"expression": "string"}),
    ("get_news", "获取新闻", {"topic": "string", "count": "integer"}),
    ("get_current_time", "获取当前时间", {"timezone": "string"}),
    ("text_length", "计算文本长度", {"text": "string"}),
    ("translate_text", "翻译文本", {"text": "string", "target_language": "string"}),
    ("get_current_weather", "查询天气", {"location": "string", "unit": "string"}),
    ("web_search", "搜索网络", {"query": "string", "num_results": "integer"}),
    ("generate_image", "根据描述生成图片", {"prompt": "string"}),
]
TOOLS_SPEC = [{"type": "function", "function": {
    "name": n, "description": d,
    "parameters": {"type": "object",
                   "properties": {k: {"type": v} for k, v in p.items()}}}}
    for n, d, p in TOOLS]

# 按问题猜"该给哪几个工具"。**不给全部 11 个是有原因的**:训练数据里
# **81% 的对话只带 1 个工具**(16% 带 2 个),一次给 11 个是分布外 ——
# 实测会跨工具串参数(把 unit_converter 的参数写成 get_exchange_rate 的
# from_currency/to_currency)。要测多工具就显式传 --tools。
KEYWORD = [
    ("unit_converter", ("换算", "等于多少磅", "多少公里", "多少英里", "摄氏度", "华氏")),
    ("get_exchange_rate", ("汇率", "美元", "人民币", "欧元", "日元", "兑换")),
    ("get_current_weather", ("天气", "气温", "下雨", "温度")),
    ("get_current_time", ("几点", "现在时间", "当前时间", "日期")),
    ("calculate_math", ("计算", "算一下", "等于多少", "+", "*", "×")),
    ("text_length", ("几个字", "字数", "多长", "长度")),
    ("random_number", ("随机", "随机数")),
    ("translate_text", ("翻译", "英文怎么说")),
    ("generate_image", ("生成图", "画一", "图片", "插图")),
    ("get_news", ("新闻", "资讯")),
    ("web_search", ("搜索", "查一下", "搜一下")),
]


def guess_tools(q, k=2):
    """按关键词猜该给哪几个工具;猜不到就给一个通用集。"""
    hit = [n for n, kws in KEYWORD if any(w in q for w in kws)]
    if not hit:
        hit = ["web_search", "get_current_time"]
    return hit[:k]


RATES = {("USD", "CNY"): 7.21, ("CNY", "USD"): 0.139, ("EUR", "CNY"): 7.85,
         ("USD", "JPY"): 151.2, ("USD", "EUR"): 0.92, ("CNY", "JPY"): 20.9}
UNITS = {("kg", "pounds"): 2.20462, ("pounds", "kg"): 0.453592,
         ("km", "miles"): 0.621371, ("miles", "km"): 1.60934,
         ("celsius", "fahrenheit"): lambda x: x * 9 / 5 + 32,
         ("fahrenheit", "celsius"): lambda x: (x - 32) * 5 / 9,
         ("m", "cm"): 100, ("cm", "m"): 0.01}


def run_tool(name, args):
    """模拟工具执行。能真算的真算,其余返回占位符。"""
    try:
        if name == "calculate_math":
            expr = str(args.get("expression", ""))
            if not re.fullmatch(r"[\d\s+\-*/().%*]+", expr):
                return {"error": "只支持算术表达式"}
            return {"result": eval(expr, {"__builtins__": {}}, {})}
        if name == "text_length":
            return {"length": len(str(args.get("text", "")))}
        if name == "random_number":
            return {"result": random.randint(int(args.get("min", 0)),
                                             int(args.get("max", 100)))}
        if name == "unit_converter":
            v = float(args.get("value", 0))
            f, t = str(args.get("from_unit", "")), str(args.get("to_unit", ""))
            k = UNITS.get((f, t))
            if k is None:
                return {"error": "不支持该换算"}
            r = k(v) if callable(k) else v * k
            return {"result": round(r, 4), "from": "%s %s" % (v, f),
                    "to": "%.4f %s" % (r, t)}
        if name == "get_exchange_rate":
            a, b = str(args.get("from_currency", "")).upper(), str(args.get("to_currency", "")).upper()
            r = RATES.get((a, b))
            return ({"from_currency": a, "to_currency": b, "rate": r,
                     "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")}
                    if r else {"error": "无该货币对"})
        if name == "get_current_time":
            return {"timezone": args.get("timezone", "UTC"),
                    "datetime": time.strftime("%Y-%m-%d %H:%M:%S")}
        # 剩下的是人造工具,没法真跑 —— 明确标注是模拟的
        return {"result": "(模拟结果,chat15.py 无法真正执行 %s)" % name}
    except Exception as e:
        return {"error": str(e)}


@torch.no_grad()
def gen(model, tok, msgs, mm, max_new, temperature, tools):
    prompt = THINK.sub("", mm.apply_chat_template(
        msgs, tools=tools, tokenize=False, add_generation_prompt=True))
    ids = tok.encode(prompt)
    ctx = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    logits, past = model.forward_cached(ctx, None)
    out, stopped = [], False
    for _ in range(max_new):
        if temperature <= 0:
            nxt = int(logits[0, -1].argmax().item())
        else:
            p = torch.softmax(logits[0, -1] / temperature, dim=-1)
            nxt = int(torch.multinomial(p, 1).item())
        out.append(nxt)
        if nxt == EOS_ID or IM_END in tok.decode(out):
            stopped = True
            break
        logits, past = model.forward_cached(
            torch.tensor([[nxt]], dtype=torch.long, device=DEVICE), past)
    return tok.decode(out).split(IM_END)[0].strip(), stopped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "ckpt_15_tool.pt"))
    ap.add_argument("--bpe", default=str(REPO / "stage11_datascale/cache_mm10g/bpe.json"))
    ap.add_argument("--ask", action="append", default=[],
                    help="同一段对话的连续轮次(可重复)")
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-rounds", type=int, default=3, help="工具链最多几轮")
    ap.add_argument("--tools", default="", help="逗号分隔的工具名;留空则按问题自动猜(默认最多 2 个)")
    ap.add_argument("--n-tools", type=int, default=2, help="自动猜时最多给几个")
    args = ap.parse_args()

    tok = BPETokenizer.load(args.bpe)
    ck = torch.load(args.ckpt)
    model = GPT(GPTConfig(**ck["config"])).to(DEVICE)
    model.load_state_dict(ck["model"])
    model.eval()
    mm = AutoTokenizer.from_pretrained(Path.home() / "llm_study" / "minimind" / "model")
    print(f"[chat15] {Path(args.ckpt).name} | {sum(p.numel() for p in model.parameters())/1e6:.1f}M "
          f"| 温度 {args.temperature} | **带工具**(11 个,结果模拟)", flush=True)

    msgs = []

    def turn(q):
        msgs.append({"role": "user", "content": q})
        names = ([x.strip() for x in args.tools.split(",") if x.strip()]
                 if args.tools else guess_tools(q, args.n_tools))
        tools = [t for t in TOOLS_SPEC if t["function"]["name"] in names]
        print("  (工具: %s)" % ", ".join(t["function"]["name"] for t in tools), flush=True)
        for rnd in range(args.max_rounds):
            txt, stopped = gen(model, tok, msgs, mm, args.max_new, args.temperature, tools)
            if not stopped:
                print("⚠️  被 max_new 截断,没收尾标记 —— 后续轮次不可信,请调大 --max-new",
                      flush=True)
            m = TOOLCALL.search(txt)
            if not m:
                print("答: %s" % txt)
                msgs.append({"role": "assistant", "content": txt})
                return
            try:
                call = json.loads(m.group(1))
                fn = call.get("function", call)         # 两种嵌套都容忍
                name = fn.get("name")
                a = fn.get("arguments")
                a = json.loads(a) if isinstance(a, str) else (a or {})
            except Exception as e:
                print("答: %s\n  (调用解析失败: %s)" % (txt, e))
                msgs.append({"role": "assistant", "content": txt})
                return
            res = run_tool(name, a)
            print("🔧 调用 %s(%s)" % (name, json.dumps(a, ensure_ascii=False)), flush=True)
            print("   ← %s" % json.dumps(res, ensure_ascii=False), flush=True)
            msgs.append({"role": "assistant", "content": "",
                         "tool_calls": [{"type": "function",
                                         "function": {"name": name,
                                                      "arguments": json.dumps(a)}}]})
            msgs.append({"role": "tool", "content": json.dumps(res, ensure_ascii=False)})
        print("  (达到最多 %d 轮工具链,停止)" % args.max_rounds)

    if args.ask:
        for i, q in enumerate(args.ask, 1):
            print("\n--- 第 %d 轮 ---\n问:%s" % (i, q), flush=True)
            turn(q)
    else:
        print("\n(交互模式,直接打字;Ctrl-D 退出)\n")
        while True:
            try:
                q = input("问: ")
            except EOFError:
                break
            if not q.strip():
                continue
            turn(q)
            print()


if __name__ == "__main__":
    main()
