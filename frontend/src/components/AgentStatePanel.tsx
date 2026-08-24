import type { ReactNode } from 'react';
import type { AgentState } from '../types/agentic';

interface AgentStatePanelProps {
  state: AgentState;
}

const COLORS: Record<string, string> = {
  detect: 'bg-red-50 text-red-700 border-red-200',
  read_text: 'bg-emerald-50 text-emerald-700 border-emerald-200',
  locate: 'bg-sky-50 text-sky-700 border-sky-200',
  describe: 'bg-purple-50 text-purple-700 border-purple-200',
  count: 'bg-amber-50 text-amber-700 border-amber-200',
  default: 'bg-warm-50 text-warm-700 border-warm-200',
};

function getViaStyle(via: string) {
  return COLORS[via] ?? COLORS.default;
}

function Card({ title, count, children }: { title: string; count: number; children: ReactNode }) {
  return (
    <div className="panel flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-warm-800 tracking-wide">{title}</h3>
        <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-primary-50 text-primary-700">
          {count}
        </span>
      </div>
      <div className="flex-1 min-h-0">
        {children}
      </div>
    </div>
  );
}

export function AgentStatePanel({ state }: AgentStatePanelProps) {
  const detectedCount = state.detected_objects.length;
  const textCount = state.extracted_text.length;
  const locatedCount = state.located_regions.length;
  const countEntries = Object.entries(state.counts);
  const descriptionCount = state.descriptions.length;
  const issueCount = state.pending_issues.length;

  const isEmpty =
    detectedCount === 0 &&
    textCount === 0 &&
    locatedCount === 0 &&
    countEntries.length === 0 &&
    descriptionCount === 0 &&
    issueCount === 0;

  if (isEmpty) {
    return (
      <div className="panel flex flex-col items-center justify-center py-12 text-warm-400">
        <svg className="w-10 h-10 mb-3 opacity-40" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
        </svg>
        <p className="text-sm font-medium">等待执行...</p>
        <p className="text-xs mt-1 opacity-60">当前没有检测或分析结果</p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
      {detectedCount > 0 && (
        <Card title="检测到对象" count={detectedCount}>
          <div className="flex flex-col gap-2 max-h-48 overflow-y-auto pr-1">
            {state.detected_objects.map((obj, i) => (
              <div key={i} className={`rounded-lg border px-3 py-2 text-xs ${getViaStyle(obj.via)}`}>
                <div className="flex items-center justify-between mb-1">
                  <span className="font-medium">对象 #{i + 1}</span>
                  <span className="opacity-70">{obj.via}</span>
                </div>
                <div className="font-mono opacity-80">
                  [{obj.box.map((n) => n.toFixed(0)).join(', ')}]
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {textCount > 0 && (
        <Card title="提取文字" count={textCount}>
          <div className="flex flex-col gap-2 max-h-48 overflow-y-auto pr-1">
            {state.extracted_text.map((text, i) => (
              <div key={i} className="rounded-lg border border-emerald-100 bg-emerald-50/50 px-3 py-2 text-xs text-emerald-800">
                <div className="font-medium mb-0.5">片段 #{i + 1}</div>
                <div className="font-mono opacity-90 leading-relaxed break-all">{text}</div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {locatedCount > 0 && (
        <Card title="定位区域" count={locatedCount}>
          <div className="flex flex-col gap-2 max-h-48 overflow-y-auto pr-1">
            {state.located_regions.map((reg, i) => (
              <div key={i} className={`rounded-lg border px-3 py-2 text-xs ${getViaStyle(reg.via)}`}>
                <div className="flex items-center justify-between mb-1">
                  <span className="font-medium">区域 #{i + 1}</span>
                  <span className="opacity-70">{reg.via}</span>
                </div>
                <div className="font-mono opacity-80">
                  [{reg.box.map((n) => n.toFixed(0)).join(', ')}]
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {countEntries.length > 0 && (
        <Card title="计数" count={countEntries.length}>
          <div className="flex flex-col gap-2 max-h-48 overflow-y-auto pr-1">
            {countEntries.map(([key, value]) => (
              <div key={key} className="flex items-center justify-between rounded-lg border border-amber-100 bg-amber-50/50 px-3 py-2 text-xs">
                <span className="font-medium text-amber-800">{key}</span>
                <span className="text-lg font-bold text-amber-700 tabular-nums">{value}</span>
              </div>
            ))}
          </div>
        </Card>
      )}

      {descriptionCount > 0 && (
        <Card title="描述" count={descriptionCount}>
          <div className="flex flex-col gap-2 max-h-48 overflow-y-auto pr-1">
            {state.descriptions.map((desc, i) => (
              <div key={i} className="rounded-lg border border-purple-100 bg-purple-50/50 px-3 py-2 text-xs text-purple-800 leading-relaxed">
                <span className="font-medium mr-1">#{i + 1}</span>
                {desc}
              </div>
            ))}
          </div>
        </Card>
      )}

      {issueCount > 0 && (
        <Card title="待处理 issues" count={issueCount}>
          <div className="flex flex-col gap-2 max-h-48 overflow-y-auto pr-1">
            {state.pending_issues.map((issue, i) => (
              <div key={i} className="flex items-start gap-2 rounded-lg border border-red-100 bg-red-50/50 px-3 py-2 text-xs text-red-700">
                <svg className="w-4 h-4 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
                <span className="leading-relaxed">{issue}</span>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
