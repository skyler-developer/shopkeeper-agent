/**
 * 智能体类型定义
 * 定义问数智能体前端使用的 SSE 事件、流程步骤和聊天消息类型
 */
export type ProgressStatus = "running" | "success" | "error";

export type ProgressEvent = {
  type: "progress";
  step: string;
  status: ProgressStatus;
};

export type ResultEvent = {
  type: "result";
  data: unknown;
};

export type AnalysisFinding = {
  title: string;
  detail: string;
};

export type AnalysisData = {
  summary: string;
  findings?: AnalysisFinding[];
  suggestions?: string[];
};

export type AnalysisEvent = {
  type: "analysis";
  data: AnalysisData;
};

export type ChartEvent = {
  type: "chart";
  data: Record<string, unknown>;
};

export type ErrorEvent = {
  type: "error";
  message: string;
};

export type AgentEvent =
  | ProgressEvent
  | ResultEvent
  | AnalysisEvent
  | ChartEvent
  | ErrorEvent;

export type StepState = {
  step: string;
  status: ProgressStatus;
  updatedAt: number;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: number;
  status?: "streaming" | "done" | "error";
  steps?: StepState[];
  result?: unknown;
  analysis?: AnalysisData;
  chart?: Record<string, unknown>;
  error?: string;
};
