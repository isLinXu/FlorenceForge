/**
 * API 调用工具
 * 封装与后端 FastAPI Agentic 端点的交互
 */

/// <reference types="vite/client" />

import type {
  ToolsResponse,
  OrchestratorResult,
  RunAgentRequest,
  SseEvent,
  SseEventType,
  PlanEventPayload,
  StepEventPayload,
  DoneEventPayload,
  ErrorEventPayload,
} from '../types/agentic';

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';

// ── 同步 API ──────────────────────────────────────────

export async function getTools(): Promise<ToolsResponse> {
  const response = await fetch(`${API_BASE}/agentic/tools`, {
    headers: { Accept: 'application/json' },
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => 'Unknown error');
    throw new Error(`HTTP ${response.status}: ${errorText}`);
  }

  return response.json() as Promise<ToolsResponse>;
}

export async function runAgent(request: RunAgentRequest): Promise<OrchestratorResult> {
  const formData = buildFormData(request);

  const response = await fetch(`${API_BASE}/agentic/run`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => 'Unknown error');
    throw new Error(`HTTP ${response.status}: ${errorText}`);
  }

  return response.json() as Promise<OrchestratorResult>;
}

// ── 流式 API (POST + SSE) ───────────────────────────

export interface StreamOptions {
  onEvent?: (event: SseEvent) => void;
  onError?: (error: Error) => void;
  onClose?: () => void;
}

interface SseMessage {
  event?: string;
  data?: string;
  id?: string;
}

/**
 * 发送 POST 请求并解析 SSE 流式响应。
 * 返回一个取消函数，调用即可中断连接。
 */
export async function streamAgent(
  request: RunAgentRequest,
  options: StreamOptions = {},
): Promise<() => void> {
  const { onEvent, onError, onClose } = options;

  const abortController = new AbortController();
  const formData = buildFormData(request);

  try {
    const response = await fetch(`${API_BASE}/agentic/stream`, {
      method: 'POST',
      body: formData,
      headers: { Accept: 'text/event-stream' },
      signal: abortController.signal,
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('Response body is not readable');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    const processChunk = async (): Promise<void> => {
      let currentEvent: SseMessage = {};

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() ?? '';

          for (const line of lines) {
            if (line.trim() === '') {
              if (currentEvent.data) {
                handleSseMessage(currentEvent, onEvent, onError);
              }
              currentEvent = {};
              continue;
            }
            const parsed = parseSseLine(line);
            currentEvent = { ...currentEvent, ...parsed };
          }
        }

        // 处理缓冲区中剩余内容
        if (buffer.trim()) {
          const lines = buffer.split('\n');
          for (const line of lines) {
            if (line.trim() === '') {
              if (currentEvent.data) {
                handleSseMessage(currentEvent, onEvent, onError);
              }
              currentEvent = {};
              continue;
            }
            const parsed = parseSseLine(line);
            currentEvent = { ...currentEvent, ...parsed };
          }
          if (currentEvent.data) {
            handleSseMessage(currentEvent, onEvent, onError);
          }
        }

        onClose?.();
      } catch (err) {
        if (err instanceof Error && err.name === 'AbortError') {
          return;
        }
        onError?.(err instanceof Error ? err : new Error(String(err)));
      }
    };

    // 启动读取循环（不 await，避免阻塞返回取消函数）
    void processChunk();
  } catch (err) {
    onError?.(err instanceof Error ? err : new Error(String(err)));
  }

  return () => abortController.abort();
}

// ── 内部工具函数 ──────────────────────────────────────

function buildFormData(request: RunAgentRequest): FormData {
  const formData = new FormData();
  formData.append('image', request.image);
  formData.append('goal', request.goal);
  if (request.max_steps !== undefined) {
    formData.append('max_steps', String(request.max_steps));
  }
  if (request.max_retries !== undefined) {
    formData.append('max_retries', String(request.max_retries));
  }
  return formData;
}

function parseSseLine(line: string): Partial<SseMessage> {
  if (line.startsWith('event:')) return { event: line.slice(6).trim() };
  if (line.startsWith('data:')) return { data: line.slice(5).trim() };
  if (line.startsWith('id:')) return { id: line.slice(3).trim() };
  return {};
}

function handleSseMessage(
  message: SseMessage,
  onEvent?: (event: SseEvent) => void,
  onError?: (error: Error) => void,
): void {
  if (!message.event || !message.data) return;

  try {
    const payload = JSON.parse(message.data) as unknown;

    switch (message.event) {
      case 'plan':
        onEvent?.({ type: 'plan', payload: payload as PlanEventPayload });
        break;
      case 'step':
        onEvent?.({ type: 'step', payload: payload as StepEventPayload });
        break;
      case 'done':
        onEvent?.({ type: 'done', payload: payload as DoneEventPayload });
        break;
      case 'error':
        onEvent?.({ type: 'error', payload: payload as ErrorEventPayload });
        break;
      default:
        onEvent?.({ type: message.event as SseEventType, payload });
    }
  } catch (err) {
    onError?.(new Error(`Failed to parse SSE event data: ${String(err)}`));
  }
}
