import { useMemo } from 'react';
import type { AgentState } from '../types/agentic';

interface FinalAnswerProps {
  answer: string;
  success: boolean;
  state: AgentState;
}

export function FinalAnswer({ answer, success, state }: FinalAnswerProps) {
  const stats = useMemo(() => {
    const detected = state.detected_objects.length;
    const text = state.extracted_text.length;
    const located = state.located_regions.length;
    const countSum = Object.values(state.counts).reduce((a, b) => a + b, 0);
    const descriptions = state.descriptions.length;
    const issues = state.pending_issues.length;
    return { detected, text, located, countSum, descriptions, issues };
  }, [state]);

  return (
    <div className="panel flex flex-col gap-4">
      <div className="flex items-center gap-3">
        <div
          className={`flex items-center justify-center w-10 h-10 rounded-full ${
            success ? 'bg-emerald-50' : 'bg-red-50'
          }`}
        >
          {success ? (
            <svg className="w-5 h-5 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          ) : (
            <svg className="w-5 h-5 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          )}
        </div>
        <div>
          <h2 className="text-lg font-bold text-warm-800">
            {success ? '任务成功完成' : '任务执行失败'}
          </h2>
          <p className="text-xs text-warm-500 mt-0.5">
            {success ? '所有步骤已验证通过' : `${state.pending_issues.length} 个待处理问题`}
          </p>
        </div>
      </div>

      <div className={`rounded-xl border p-4 ${success ? 'bg-emerald-50/50 border-emerald-200' : 'bg-red-50/50 border-red-200'}`}>
        <div className="text-xs font-semibold uppercase tracking-wider text-warm-500 mb-2">
          最终答案
        </div>
        <div className={`text-sm leading-relaxed font-medium ${success ? 'text-emerald-800' : 'text-red-800'}`}>
          {answer}
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        <div className="flex flex-col gap-1 rounded-lg border border-warm-200 bg-warm-50 px-3 py-2.5">
          <span className="text-[10px] uppercase tracking-wider text-warm-400 font-medium">检测对象</span>
          <span className="text-lg font-bold text-warm-800 tabular-nums">{stats.detected}</span>
        </div>
        <div className="flex flex-col gap-1 rounded-lg border border-warm-200 bg-warm-50 px-3 py-2.5">
          <span className="text-[10px] uppercase tracking-wider text-warm-400 font-medium">文字片段</span>
          <span className="text-lg font-bold text-warm-800 tabular-nums">{stats.text}</span>
        </div>
        <div className="flex flex-col gap-1 rounded-lg border border-warm-200 bg-warm-50 px-3 py-2.5">
          <span className="text-[10px] uppercase tracking-wider text-warm-400 font-medium">定位区域</span>
          <span className="text-lg font-bold text-warm-800 tabular-nums">{stats.located}</span>
        </div>
        <div className="flex flex-col gap-1 rounded-lg border border-warm-200 bg-warm-50 px-3 py-2.5">
          <span className="text-[10px] uppercase tracking-wider text-warm-400 font-medium">计数合计</span>
          <span className="text-lg font-bold text-warm-800 tabular-nums">{stats.countSum}</span>
        </div>
        <div className="flex flex-col gap-1 rounded-lg border border-warm-200 bg-warm-50 px-3 py-2.5">
          <span className="text-[10px] uppercase tracking-wider text-warm-400 font-medium">描述</span>
          <span className="text-lg font-bold text-warm-800 tabular-nums">{stats.descriptions}</span>
        </div>
        <div className="flex flex-col gap-1 rounded-lg border border-warm-200 bg-warm-50 px-3 py-2.5">
          <span className="text-[10px] uppercase tracking-wider text-warm-400 font-medium">待处理 issues</span>
          <span className={`text-lg font-bold tabular-nums ${stats.issues > 0 ? 'text-red-600' : 'text-warm-800'}`}>
            {stats.issues}
          </span>
        </div>
      </div>
    </div>
  );
}
