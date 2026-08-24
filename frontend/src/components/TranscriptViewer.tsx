import { useState, useMemo } from 'react';

interface TranscriptViewerProps {
  transcript: string;
}

interface TokenBlock {
  tag: string;
  content: string;
  start: number;
  end: number;
}

const TAG_COLORS: Record<string, string> = {
  PLAN: 'bg-blue-50 text-blue-700 border-blue-200',
  ACT: 'bg-emerald-50 text-emerald-700 border-emerald-200',
  VERIFY: 'bg-amber-50 text-amber-700 border-amber-200',
  REFLECT: 'bg-purple-50 text-purple-700 border-purple-200',
  DECIDE: 'bg-red-50 text-red-700 border-red-200',
  SUMMARIZE_STATE: 'bg-teal-50 text-teal-700 border-teal-200',
  DONE: 'bg-warm-100 text-warm-700 border-warm-300',
};

const TAG_LABELS: Record<string, string> = {
  PLAN: '计划',
  ACT: '行动',
  VERIFY: '验证',
  REFLECT: '反思',
  DECIDE: '决策',
  SUMMARIZE_STATE: '状态汇总',
  DONE: '完成',
};

function parseTranscript(transcript: string): TokenBlock[] {
  const blocks: TokenBlock[] = [];
  const tagRegex = /<(PLAN|ACT|VERIFY|REFLECT|DECIDE|SUMMARIZE_STATE|DONE)>([\s\S]*?)<\/\1>/g;
  let lastEnd = 0;
  let match: RegExpExecArray | null;

  while ((match = tagRegex.exec(transcript)) !== null) {
    if (match.index > lastEnd) {
      blocks.push({
        tag: 'TEXT',
        content: transcript.slice(lastEnd, match.index),
        start: lastEnd,
        end: match.index,
      });
    }
    blocks.push({
      tag: match[1],
      content: match[2].trim(),
      start: match.index,
      end: match.index + match[0].length,
    });
    lastEnd = match.index + match[0].length;
  }

  if (lastEnd < transcript.length) {
    blocks.push({
      tag: 'TEXT',
      content: transcript.slice(lastEnd),
      start: lastEnd,
      end: transcript.length,
    });
  }

  return blocks;
}

function TagBlock({ block }: { block: TokenBlock }) {
  const [expanded, setExpanded] = useState(true);
  const colorClass = TAG_COLORS[block.tag] ?? 'bg-warm-50 text-warm-700 border-warm-200';
  const label = TAG_LABELS[block.tag] ?? block.tag;

  return (
    <div className={`rounded-lg border ${colorClass} overflow-hidden transition-all duration-200`}>
      <button
        onClick={() => setExpanded((v) => !v)}
        className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-black/5 transition-colors"
      >
        <svg
          className={`w-3.5 h-3.5 flex-shrink-0 transition-transform duration-200 ${expanded ? 'rotate-90' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
        <span className="text-xs font-bold uppercase tracking-wider">{label}</span>
        <span className="text-xs opacity-50 font-mono ml-auto">
          {block.content.length} chars
        </span>
      </button>
      {expanded && (
        <div className="px-3 pb-2">
          <pre className="text-xs leading-relaxed whitespace-pre-wrap font-mono opacity-90">{block.content}</pre>
        </div>
      )}
    </div>
  );
}

export function TranscriptViewer({ transcript }: TranscriptViewerProps) {
  const blocks = useMemo(() => parseTranscript(transcript), [transcript]);
  const [allExpanded, setAllExpanded] = useState(true);

  const tagCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const block of blocks) {
      if (block.tag !== 'TEXT') {
        counts[block.tag] = (counts[block.tag] ?? 0) + 1;
      }
    }
    return counts;
  }, [blocks]);

  if (!transcript.trim()) {
    return (
      <div className="panel flex flex-col items-center justify-center py-12 text-warm-400">
        <svg className="w-8 h-8 mb-2 opacity-40" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
        </svg>
        <p className="text-sm font-medium">暂无 transcript</p>
      </div>
    );
  }

  return (
    <div className="panel flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold text-warm-800 tracking-wide">元认知 Transcript</h3>
          <div className="flex items-center gap-1.5">
            {Object.entries(tagCounts).map(([tag, count]) => (
              <span
                key={tag}
                className={`text-[10px] px-1.5 py-0.5 rounded-full border font-medium ${TAG_COLORS[tag] ?? 'bg-warm-50 text-warm-600 border-warm-200'}`}
              >
                {TAG_LABELS[tag] ?? tag} {count}
              </span>
            ))}
          </div>
        </div>
        <button
          onClick={() => setAllExpanded((v) => !v)}
          className="text-xs px-2.5 py-1 rounded-md border border-warm-200 bg-warm-50 text-warm-600 hover:bg-warm-100 transition-colors"
        >
          {allExpanded ? '全部折叠' : '全部展开'}
        </button>
      </div>
      <div className="flex flex-col gap-2 max-h-[28rem] overflow-y-auto pr-1">
        {blocks.map((block, i) =>
          block.tag === 'TEXT' ? (
            <div key={i} className="text-xs text-warm-500 whitespace-pre-wrap px-1 py-0.5">
              {block.content}
            </div>
          ) : (
            <TagBlock key={i} block={block} />
          )
        )}
      </div>
    </div>
  );
}
