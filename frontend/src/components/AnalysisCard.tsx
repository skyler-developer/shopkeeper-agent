/**
 * 分析结论卡片组件
 * 展示分析 skill 产出的总体结论、关键发现和业务建议
 */
import { ClipboardList, Lightbulb, Sparkles } from "lucide-react";
import type { AnalysisData } from "../types/agent";

export function AnalysisCard({ analysis }: { analysis: AnalysisData }) {
  const findings = analysis.findings ?? [];
  const suggestions = analysis.suggestions ?? [];

  return (
    <section className="mt-4 border border-ink/10 bg-[#fffaf1]/78 shadow-line">
      <div className="flex items-center gap-2 border-b border-ink/10 px-4 py-3 text-sm font-semibold text-ink">
        <Sparkles className="h-4 w-4 text-moss" aria-hidden="true" />
        分析结论
      </div>

      <div className="space-y-4 px-4 py-4">
        <p className="text-[15px] leading-7 text-ink">{analysis.summary}</p>

        {findings.length > 0 && (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.16em] text-ink/45">
              <ClipboardList className="h-3.5 w-3.5" aria-hidden="true" />
              关键发现
            </div>
            <ul className="space-y-2">
              {findings.map((finding, index) => (
                <li
                  key={index}
                  className="border border-ink/10 bg-white/55 px-3 py-2.5 text-sm leading-6"
                >
                  <div className="font-semibold text-ink">{finding.title}</div>
                  {finding.detail && (
                    <div className="mt-1 text-ink/70">{finding.detail}</div>
                  )}
                </li>
              ))}
            </ul>
          </div>
        )}

        {suggestions.length > 0 && (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.16em] text-ink/45">
              <Lightbulb className="h-3.5 w-3.5" aria-hidden="true" />
              业务建议
            </div>
            <ul className="space-y-1.5">
              {suggestions.map((suggestion, index) => (
                <li key={index} className="flex gap-2 text-sm leading-6 text-ink/75">
                  <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-moss" />
                  {suggestion}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </section>
  );
}
