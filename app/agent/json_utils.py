"""
LLM JSON 输出解析工具

分析和图表节点的提示词都要求模型输出纯 JSON，
但实际调用时模型经常包裹 ```json 代码块或额外说明文字。
这里统一做一层容错解析，解析失败返回 None，由调用方决定降级策略。
"""

import json


def parse_json_object(text: str) -> dict | None:
    """从模型输出文本中解析出 JSON 对象，失败时返回 None"""

    if not text:
        return None

    candidate = text.strip()

    # 剥离 Markdown 代码块包裹
    if candidate.startswith("```"):
        candidate = candidate.strip("`").strip()
        if candidate.lower().startswith("json"):
            candidate = candidate[4:].strip()

    # 优先尝试整体解析，再截取首个大括号到最后一个大括号之间的片段
    for fragment in (candidate, _extract_braces(candidate)):
        if not fragment:
            continue
        try:
            parsed = json.loads(fragment)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    return None


def _extract_braces(text: str) -> str | None:
    """截取文本中第一个 { 到最后一个 } 之间的片段"""

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    return text[start : end + 1]
