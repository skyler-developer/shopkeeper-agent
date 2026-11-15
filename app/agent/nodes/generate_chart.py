"""
图表生成节点

基于查询结果和已产出的分析结论，生成一份前端可直接 setOption 渲染的 ECharts option。
图表选型规则沉淀在 prompts/generate_chart.prompt 中，相当于可独立替换的可视化 skill。
生成失败时只降级不中断：缺少图表不影响已经交付的查询结果和分析结论。
"""

import yaml
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.json_utils import parse_json_object
from app.agent.llm import llm
from app.agent.state import DataAgentState
from app.core.log import logger
from app.prompt.prompt_loader import load_prompt

MAX_ROWS_IN_PROMPT = 50


async def generate_chart(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    """基于查询结果生成 ECharts option 配置"""

    writer = runtime.stream_writer
    step = "生成图表"
    writer({"type": "progress", "step": step, "status": "running"})

    try:
        query = state["query"]
        result_rows = state.get("result_rows") or []
        analysis = state.get("analysis") or {}

        # 空结果或单行纯数值结果画不出有意义的图，直接跳过
        if len(result_rows) < 2:
            writer({"type": "progress", "step": step, "status": "success"})
            return {}

        prompt = PromptTemplate(
            template=load_prompt("generate_chart"),
            input_variables=["query", "result_rows", "analysis"],
        )
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser

        result = await chain.ainvoke(
            {
                "query": query,
                "result_rows": yaml.dump(
                    result_rows[:MAX_ROWS_IN_PROMPT],
                    allow_unicode=True,
                    sort_keys=False,
                ),
                # 图表标题和结论呼应，让可视化成为分析结论的载体
                "analysis": yaml.dump(analysis, allow_unicode=True, sort_keys=False),
            }
        )
        logger.info(f"图表配置原文：{result}")

        option = parse_json_object(result)

        # 轻量校验：必须是含 series 的对象，避免把说明文字当 option 渲染
        if not option or not option.get("series"):
            logger.warning("图表配置解析失败或缺少 series，跳过图表渲染")
            writer({"type": "progress", "step": step, "status": "success"})
            return {}

        writer({"type": "progress", "step": step, "status": "success"})
        writer({"type": "chart", "data": option})
        return {"chart_option": option}

    except Exception as e:
        # 图表是增强信息，失败时降级为不展示图表，不影响已交付的结果
        logger.error(f"{step} failed: {e}")
        writer({"type": "progress", "step": step, "status": "error"})
        return {}
