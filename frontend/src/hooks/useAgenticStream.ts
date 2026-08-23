/**
 * useAgenticStream
 * SSE 流式连接 Hook，支持自动重连与错误处理
 */

import { useState, useRef, useCallback, useEffect } from 'react';
import { streamAgent } from '../utils/api';
import type {
  RunAgentRequest,
  SseEvent,
  OrchestratorResult,
  DoneEventPayload,
} from '../types/agentic';

export interface UseAgenticStreamReturn {
  /** 是否已建立 SSE 连接 */
  connected: boolean;
  /** 是否正在加载（连接中或重试中） */
  loading: boolean;
  /** 错误消息，null 表示无错误 */
  error: string | null;
  /** 已接收的全部 SSE 事件 */
  events: SseEvent[];
  /** 当收到 done 事件时，解析出的最终结果 */
  currentResult: OrchestratorResult | null;
  /** 启动流式请求 */
  startStream: (request: RunAgentRequest) => void;
  /** 立即中断当前流并重置重试 */
  stopStream: () => void;
  /** 完全重置所有状态 */
  reset: () => void;
}

/**
 * @param maxRetries  连接失败后的最大重试次数（默认 3）
 * @param retryDelay  每次重试的间隔毫秒数（默认 2000）
 */
export function useAgenticStream(
  maxRetries = 3,
  retryDelay = 2000,
): UseAgenticStreamReturn {
  const [connected, setConnected] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [events, setEvents] = useState<SseEvent[]>([]);
  const [currentResult, setCurrentResult] = useState<OrchestratorResult | null>(null);

  const abortRef = useRef<(() => void) | null>(null);
  const retryCountRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const requestRef = useRef<RunAgentRequest | null>(null);

  /** 停止当前流并重置重试计时器 */
  const stopStream = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    if (abortRef.current) {
      abortRef.current();
      abortRef.current = null;
    }
    requestRef.current = null;
    setConnected(false);
    setLoading(false);
  }, []);

  /** 重置所有状态到初始值 */
  const reset = useCallback(() => {
    stopStream();
    setEvents([]);
    setError(null);
    setCurrentResult(null);
    retryCountRef.current = 0;
  }, [stopStream]);

  /** 启动流式连接（会自动中断之前的连接） */
  const startStream = useCallback(
    (request: RunAgentRequest) => {
      // 1. 先清理旧连接
      stopStream();

      // 2. 初始化状态
      setLoading(true);
      setError(null);
      setEvents([]);
      setCurrentResult(null);
      retryCountRef.current = 0;
      requestRef.current = request;

      const attemptConnection = async (): Promise<void> => {
        const currentRequest = requestRef.current;
        if (!currentRequest) return;

        try {
          const abort = await streamAgent(currentRequest, {
            onEvent: (event) => {
              setEvents((prev) => [...prev, event]);
              if (event.type === 'done') {
                setCurrentResult((event.payload as DoneEventPayload).result);
                setConnected(false);
                setLoading(false);
              }
            },
            onError: (err) => {
              if (
                retryCountRef.current < maxRetries &&
                requestRef.current === currentRequest
              ) {
                retryCountRef.current += 1;
                reconnectTimerRef.current = setTimeout(() => {
                  if (requestRef.current === currentRequest) {
                    void attemptConnection();
                  }
                }, retryDelay);
              } else {
                setError(err.message);
                setConnected(false);
                setLoading(false);
              }
            },
            onClose: () => {
              setConnected(false);
              setLoading(false);
            },
          });

          abortRef.current = abort;
          setConnected(true);
        } catch (err) {
          const message = err instanceof Error ? err.message : String(err);
          if (
            retryCountRef.current < maxRetries &&
            requestRef.current === currentRequest
          ) {
            retryCountRef.current += 1;
            reconnectTimerRef.current = setTimeout(() => {
              if (requestRef.current === currentRequest) {
                void attemptConnection();
              }
            }, retryDelay);
          } else {
            setError(message);
            setConnected(false);
            setLoading(false);
          }
        }
      };

      void attemptConnection();
    },
    [stopStream, maxRetries, retryDelay],
  );

  // 组件卸载时清理所有副作用
  useEffect(() => {
    return () => {
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
      }
      if (abortRef.current) {
        abortRef.current();
      }
    };
  }, []);

  return {
    connected,
    loading,
    error,
    events,
    currentResult,
    startStream,
    stopStream,
    reset,
  };
}
