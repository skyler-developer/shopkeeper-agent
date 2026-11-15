/**
 * ECharts 图表组件
 * 接收后端图表 skill 生成的 ECharts option 配置并渲染
 */
import * as echarts from "echarts";
import { useEffect, useRef } from "react";
import { LineChart } from "lucide-react";

function isEchartsOption(value: Record<string, unknown>) {
  return Array.isArray(value.series) && value.series.length > 0;
}

export function ChartPanel({ option }: { option: Record<string, unknown> }) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    if (!containerRef.current || !isEchartsOption(option)) return;

    // 图表实例只在挂载时创建，后续统一走 setOption 更新
    if (!chartRef.current) {
      chartRef.current = echarts.init(containerRef.current);
    }
    chartRef.current.setOption(option as echarts.EChartsOption, { notMerge: true });

    const handleResize = () => chartRef.current?.resize();
    window.addEventListener("resize", handleResize);
    return () => {
      window.removeEventListener("resize", handleResize);
    };
  }, [option]);

  useEffect(() => {
    // 卸载时销毁实例，避免连续消息累积图表句柄
    return () => {
      chartRef.current?.dispose();
      chartRef.current = null;
    };
  }, []);

  if (!isEchartsOption(option)) {
    return null;
  }

  return (
    <section className="mt-4 overflow-hidden border border-ink/10 bg-white/70 shadow-line">
      <div className="flex items-center gap-2 border-b border-ink/10 px-4 py-3 text-sm font-semibold text-ink">
        <LineChart className="h-4 w-4 text-moss" aria-hidden="true" />
        图表可视化
      </div>
      <div ref={containerRef} className="h-[340px] w-full bg-white/60" />
    </section>
  );
}
