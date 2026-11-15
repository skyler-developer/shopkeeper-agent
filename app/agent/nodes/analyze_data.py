"""
数据分析节点

在 SQL 执行拿到结果行之后，基于用户问题 执行的 SQL 和查询结果生成结构化分析结论。
分析能力沉淀在 prompts/analyze_data.prompt 中，相当于可独立替换的分析 skill。
分析失败时只降级不中断：查询结果已经流式返回给用户，不能因为分析异常把整次问答标记为失败。
"""

import yaml
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.json_utils import parse_json_object
from app.agent.llm import llm
from app.agent.state import AnalysisState, DataAgentState
from app.core.log import logger
from app.prompt.prompt_loader import load_prompt

# 送入提示词的最大行数，避免大结果集撑爆上下文
MAX_ROWS_IN_PROMPT = 50


def _trim_rows(rows: list[dict]) -> list[dict]:
    """截断结果行并压缩超长单元格，控制提示词体积"""

    trimmed = []
    for row in rows[:MAX_ROWS_IN_PROMPT]:
        trimmed.append(
            {
                key: value if len(str(value)) <= 200 else str(value)[:200]
                for key, value in row.items()
            }
        )
    return trimmed


def _normalize(parsed: dict, raw_text: str) -> AnalysisState:
    """把模型输出归一化成 AnalysisState，缺字段时兜底"""

    findings = []
    for item in parsed.get("findings") or []:
        if isinstance(item, dict) and item.get("title"):
            findings.append(
                {"title": str(item["title"]), "detail": str(item.get("detail", ""))}
            )

    suggestions = [
        str(item) for item in parsed.get("suggestions") or [] if str(item).strip()
    ]

    summary = parsed.get("summary")
    if not summary:
        # 结构化解析失败时，把原始输出整体作为摘要，保证结论不丢
        summary = raw_text.strip() or "本次查询未产出分析结论。"

    return {"summary": str(summary), "findings": findings, "suggestions": suggestions}


async def analyze_data(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    """基于查询结果生成结构化分析结论"""

    writer = runtime.stream_writer
    step = "数据分析"
    writer({"type": "progress", "step": step, "status": "running"})

    try:
        query = state["query"]
        sql = state["sql"]
        result_rows = state.get("result_rows") or []

        # 结果为空时没有可分析的数据，直接给出占位结论
        if not result_rows:
            writer({"type": "progress", "step": step, "status": "success"})
            analysis = {
                "summary": "查询结果为空，没有可分析的数据。",
                "findings": [],
                "suggestions": ["请调整筛选条件或时间范围后重试。"],
            }
            writer({"type": "analysis", "data": analysis})
            return {"analysis": analysis}

        prompt = PromptTemplate(
            template=load_prompt("analyze_data"),
            input_variables=["query", "sql", "result_rows"],
        )
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser

        result = await chain.ainvoke(
            {
                "query": query,
                "sql": sql,
                # YAML 序列化保留中文字段名和行结构，模型更容易对齐列语义
                "result_rows": yaml.dump(
                    _trim_rows(result_rows), allow_unicode=True, sort_keys=False
                ),
            }
        )
        logger.info(f"分析结论原文：{result}")

        parsed = parse_json_object(result)
        analysis = _normalize(parsed or {}, result)
        writer({"type": "progress", "step": step, "status": "success"})
        writer({"type": "analysis", "data": analysis})
        return {"analysis": analysis}

    except Exception as e:
        # 查询结果已经交付给用户，分析失败只降级不抛出，后续图表节点继续执行
        logger.error(f"{step} failed: {e}")
        writer({"type": "progress", "step": step, "status": "error"})
        return {}
