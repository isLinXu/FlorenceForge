/**
 * Agentic 类型定义
 * 与后端 FastAPI Python 数据结构严格对齐
 */

// ── 核心数据结构 ─────────────────────────────────────

export interface ToolCall {
  intent: string;
  task_name: string;
  text_input?: string | null;
}

export interface StepRecord {
  sub_task_index: number;
  intent: string;
  tool_call: ToolCall;
  raw_output: string;
  parsed: Record<string, unknown>;
  verified: boolean;
  issues: string[];
  attempts: number;
  transcript: string;
  visualization?: string;
}

export interface AgentState {
  detected_objects: Array<{ box: number[]; via: string }>;
  extracted_text: string[];
  located_regions: Array<{ box: number[]; via: string }>;
  counts: Record<string, number>;
  descriptions: string[];
  pending_issues: string[];
}

export interface PlanStep {
  intent: string;
  tool: string;
  description?: string;
}

export interface PlanResult {
  steps: PlanStep[];
}

export interface OrchestratorResult {
  goal: string;
  plan: PlanResult;
  steps: StepRecord[];
  state: AgentState;
  final_answer: string;
  transcript: string;
  success: boolean;
}

// ── 工具列表 ──────────────────────────────────────────

export interface ToolInfo {
  name: string;
  description: string;
  parameters?: Record<string, unknown>;
}

export interface ToolsResponse {
  tools: ToolInfo[];
}

// ── SSE 流式事件 ──────────────────────────────────────

export type SseEventType = 'plan' | 'step' | 'done' | 'error';

export interface SseEvent<T = unknown> {
  type: SseEventType;
  payload: T;
}

export interface PlanEventPayload {
  steps: PlanStep[];
}

export interface StepEventPayload {
  sub_task_index: number;
  step: StepRecord;
}

export interface DoneEventPayload {
  result: OrchestratorResult;
}

export interface ErrorEventPayload {
  message: string;
  code?: string;
}

// ── 请求参数 ──────────────────────────────────────────

export interface RunAgentRequest {
  image: File;
  goal: string;
  max_steps?: number;
  max_retries?: number;
}
