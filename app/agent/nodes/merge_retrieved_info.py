"""
召回信息合并节点

负责把字段 字段取值和指标三路召回结果聚合成统一上下文
这一层会补齐指标依赖字段 字段真实取值 主外键字段和表信息
后续过滤节点不再关心信息来自哪个检索分支，只处理合并后的表上下文和指标上下文
"""

from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.state import (
    ColumnInfoState,
    DataAgentState,
    MetricInfoState,
    TableInfoState,
)
from app.core.log import logger
from app.entities.column_info import ColumnInfo
from app.entities.metric_info import MetricInfo
from app.entities.table_info import TableInfo
from app.entities.value_info import ValueInfo


async def merge_retrieved_info(
    state: DataAgentState, runtime: Runtime[DataAgentContext]
):
    """合并召回结果，并输出 SQL 生成前的候选表信息和指标信息"""

    writer = runtime.stream_writer
    step = "合并召回信息"
    writer({"type": "progress", "step": step, "status": "running"})

    try:
        retrieved_column_infos: list[ColumnInfo] = state["retrieved_column_infos"]
        retrieved_metric_infos: list[MetricInfo] = state["retrieved_metric_infos"]
        retrieved_value_infos: list[ValueInfo] = state["retrieved_value_infos"]

        meta_mysql_repository = runtime.context["meta_mysql_repository"]

        # 本节点的主线是：
        # 字段召回 + 指标依赖字段 + 字段真实取值 + 主外键补齐
        # -> table_infos / metric_infos，交给后续过滤和 SQL 生成节点继续使用。

        # 1. 以 column_id 为 key 合并字段信息
        # 三路召回可能命中同一个字段，用 dict 可以自然去重；
        # 后续指标依赖字段和字段取值也都通过 column_id 合并进来。
        retrieved_column_infos_map: dict[str, ColumnInfo] = {
            retrieved_column_info.id: retrieved_column_info
            for retrieved_column_info in retrieved_column_infos
        }

        # 2. 补齐指标依赖字段
        # 指标告诉模型“怎么算”，但生成 SQL 还需要知道指标依赖哪些真实字段。
        # 如果相关字段没有被字段召回命中，就从 Meta MySQL 里按 id 查出来补进上下文。
        for retrieved_metric_info in retrieved_metric_infos:
            for relevant_column in retrieved_metric_info.relevant_columns:
                if relevant_column not in retrieved_column_infos_map:
                    column_info: ColumnInfo = (
                        await meta_mysql_repository.get_column_info_by_id(
                            relevant_column
                        )
                    )
                    retrieved_column_infos_map[relevant_column] = column_info

        # 3. 把字段取值合并回字段 examples
        # 字段取值召回命中的是 column_id + value，例如 dim_region.region_name.华北。
        # 把真实 value 放进字段 examples，可以帮助模型写出更接近真实数据的 where 条件。
        for retrieved_value_info in retrieved_value_infos:
            value = retrieved_value_info.value
            column_id = retrieved_value_info.column_id
            if column_id not in retrieved_column_infos_map:
                column_info: ColumnInfo = (
                    await meta_mysql_repository.get_column_info_by_id(column_id)
                )
                retrieved_column_infos_map[column_id] = column_info
            if value not in retrieved_column_infos_map[column_id].examples:
                retrieved_column_infos_map[column_id].examples.append(value)

        # 4. 按表组织字段上下文
        # SQL 生成提示词通常按“表 -> 字段列表”的方式描述结构，
        # 所以这里先把分散的字段按 table_id 归到各自所属表下面。
        # 示例：
        # table_to_columns_map = {
        #     "orders": [
        #         ColumnInfo(
        #             id="orders.pay_amount",
        #             name="pay_amount",
        #             type="decimal",
        #             role="measure",
        #             examples=[99.9, 200.0],
        #             description="订单实际支付金额",
        #             alias=["销售额", "支付金额"],
        #             table_id="orders",
        #         ),
        #         ColumnInfo(
        #             id="orders.region",
        #             name="region",
        #             type="varchar",
        #             role="dimension",
        #             examples=["华北", "华南", "华东"],
        #             description="订单所属地区",
        #             alias=["地区"],
        #             table_id="orders",
        #         ),
        #     ],
        #     "shops": [
        #         ColumnInfo(
        #             id="shops.shop_name",
        #             name="shop_name",
        #             type="varchar",
        #             role="dimension",
        #             examples=["星巴克", "瑞幸咖啡"],
        #             description="店铺名称",
        #             alias=["门店名称"],
        #             table_id="shops",
        #         ),
        #     ],
        # }
        table_to_columns_map: dict[str, list[ColumnInfo]] = {}
        for column_info in retrieved_column_infos_map.values():
            table_id = column_info.table_id
            if table_id not in table_to_columns_map:
                table_to_columns_map[table_id] = []
            table_to_columns_map[table_id].append(column_info)

        # 5. 补齐主外键字段
        # 主外键通常不会出现在用户问题里，单靠向量召回容易漏掉；
        # 但多表查询的 Join 路径必须依赖它们，所以每张候选表都要兜底补齐。
        # get_key_columns_by_table_id方法查询表的主外键，它接收一个 table_id，查询 Meta MySQL 中属于该表、且 role 为 primary_key 或 foreign_key 的字段，返回 list[ColumnInfo]。
        # [
            # ColumnInfo(id="orders.order_id", name="order_id", role="primary_key", ...),
            # ColumnInfo(id="orders.shop_id", name="shop_id", role="foreign_key", ...),
        # ]
        for table_id in table_to_columns_map.keys():
            key_columns: list[
                ColumnInfo
            ] = await meta_mysql_repository.get_key_columns_by_table_id(table_id)
            column_ids = [
                column_info.id for column_info in table_to_columns_map[table_id]
            ]
            for key_column in key_columns:
                if key_column.id not in column_ids:
                    table_to_columns_map[table_id].append(key_column)

        # 在上一步示例的基础上，补齐主外键后的最终结果示例：
        # table_to_columns_map = {
        #     "orders": [
        #         ColumnInfo(
        #             id="orders.pay_amount",
        #             name="pay_amount",
        #             type="decimal",
        #             role="measure",
        #             examples=[99.9, 200.0],
        #             description="订单实际支付金额",
        #             alias=["销售额", "支付金额"],
        #             table_id="orders",
        #         ),
        #         ColumnInfo(
        #             id="orders.region",
        #             name="region",
        #             type="varchar",
        #             role="dimension",
        #             examples=["华北", "华南", "华东"],
        #             description="订单所属地区",
        #             alias=["地区"],
        #             table_id="orders",
        #         ),
        #         ColumnInfo(
        #             id="orders.order_id",
        #             name="order_id",
        #             type="bigint",
        #             role="primary_key",
        #             examples=[10001, 10002],
        #             description="订单ID",
        #             alias=["订单编号"],
        #             table_id="orders",
        #         ),
        #         ColumnInfo(
        #             id="orders.shop_id",
        #             name="shop_id",
        #             type="bigint",
        #             role="foreign_key",
        #             examples=[201, 202],
        #             description="订单所属店铺ID",
        #             alias=["店铺ID"],
        #             table_id="orders",
        #         ),
        #     ],
        #     "shops": [
        #         ColumnInfo(
        #             id="shops.shop_name",
        #             name="shop_name",
        #             type="varchar",
        #             role="dimension",
        #             examples=["星巴克", "瑞幸咖啡"],
        #             description="店铺名称",
        #             alias=["门店名称"],
        #             table_id="shops",
        #         ),
        #         ColumnInfo(
        #             id="shops.shop_id",
        #             name="shop_id",
        #             type="bigint",
        #             role="primary_key",
        #             examples=[201, 202],
        #             description="店铺ID",
        #             alias=["门店ID"],
        #             table_id="shops",
        #         ),
        #     ],
        # }

        # 6. 生成表结构上下文
        # 数据库实体里可能包含入库和索引用字段，传给模型前只保留必要信息，
        # 让后续过滤和 SQL 生成节点面对的是更稳定的 TableInfoState 结构。
        table_infos: list[TableInfoState] = []
        for table_id, column_infos in table_to_columns_map.items():
            # 根据表 ID 查询表名、表角色和表描述，与已经分组的字段组装成完整表上下文。
            table_info: TableInfo = await meta_mysql_repository.get_table_info_by_id(
                table_id
            )
            # 精简字段信息：去掉 id 和 table_id，只保留模型生成 SQL 所需的字段属性。
            columns = [
                ColumnInfoState(
                    name=column_info.name,
                    type=column_info.type,
                    role=column_info.role,
                    examples=column_info.examples,
                    description=column_info.description,
                    alias=column_info.alias,
                )
                for column_info in column_infos
            ]
            table_info_state = TableInfoState(
                name=table_info.name,
                role=table_info.role,
                description=table_info.description,
                columns=columns,
            )
            table_infos.append(table_info_state)

        # 最终结果示例（TableInfoState 和 ColumnInfoState 运行时是字典）：
        # table_infos = [
        #     {
        #         "name": "orders",
        #         "role": "fact",
        #         "description": "订单事实表",
        #         "columns": [
        #             {
        #                 "name": "pay_amount",
        #                 "type": "decimal",
        #                 "role": "measure",
        #                 "examples": [99.9, 200.0],
        #                 "description": "订单实际支付金额",
        #                 "alias": ["销售额", "支付金额"],
        #             },
        #             {
        #                 "name": "order_id",
        #                 "type": "bigint",
        #                 "role": "primary_key",
        #                 "examples": [10001, 10002],
        #                 "description": "订单ID",
        #                 "alias": ["订单编号"],
        #             },
        #         ],
        #     },
        #     {
        #         "name": "shops",
        #         "role": "dimension",
        #         "description": "店铺维度表",
        #         "columns": [
        #             {
        #                 "name": "shop_name",
        #                 "type": "varchar",
        #                 "role": "dimension",
        #                 "examples": ["星巴克", "瑞幸咖啡"],
        #                 "description": "店铺名称",
        #                 "alias": ["门店名称"],
        #             },
        #         ],
        #     },
        # ]

        # 7. 生成指标上下文
        # 遍历召回的 MetricInfo，去掉内部 id，保留模型理解业务口径所需的名称、描述、依赖字段和别名。
        # MetricInfoState 是 TypedDict，运行时生成的是字典；这里只整理指标定义，不负责计算指标。
        metric_infos: list[MetricInfoState] = [
            MetricInfoState(
                name=retrieved_metric_info.name,
                description=retrieved_metric_info.description,
                relevant_columns=retrieved_metric_info.relevant_columns,
                alias=retrieved_metric_info.alias,
            )
            for retrieved_metric_info in retrieved_metric_infos
        ]
        # 最终结果示例：
        # metric_infos = [
        #     {
        #         "name": "GMV",
        #         "description": "订单支付总金额",
        #         "relevant_columns": [
        #             "orders.pay_amount",
        #             "orders.pay_status",
        #         ],
        #         "alias": ["销售额", "成交总额"],
        #     },
        #     {
        #         "name": "支付订单数",
        #         "description": "完成支付的订单数量",
        #         "relevant_columns": [
        #             "orders.order_id",
        #             "orders.pay_status",
        #         ],
        #         "alias": ["支付单量"],
        #     },
        # ]

        logger.info(
            f"合并后的表信息：{[table_info['name'] for table_info in table_infos]}"
        )
        logger.info(
            f"合并后的指标信息：{[metric_info['name'] for metric_info in metric_infos]}"
        )

        writer({"type": "progress", "step": step, "status": "success"})
        return {
            "table_infos": table_infos,
            "metric_infos": metric_infos,
        }
    except Exception as e:
        logger.error(f"{step} failed: {e}")
        writer({"type": "progress", "step": step, "status": "error"})
        raise
