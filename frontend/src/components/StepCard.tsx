import { useState } from "react";
import type { StepRecord } from "../types/agentic";

interface StepCardProps {
  step: StepRecord;
  visualization?: string;
}

export function StepCard({ step, visualization }: StepCardProps) {
  const [expanded, setExpanded] = useState(false);

  const statusConfig = step.verified
    ? {
        label: "Verified",
        dotColor: "bg-emerald-500",
        badgeClass:
          "bg-emerald-50 text-emerald-700 border-emerald-200",
      }
    : step.issues.length > 0
    ? {
        label: "Failed",
        dotColor: "bg-rose-500",
        badgeClass: "bg-rose-50 text-rose-700 border-rose-200",
      }
    : {
        label: "Pending",
        dotColor: "bg-amber-400",
        badgeClass: "bg-amber-50 text-amber-700 border-amber-200",
      };

  return (
    <div className="w-full rounded-2xl border border-warm-200 bg-white overflow-hidden transition-all duration-200 hover:shadow-md hover:shadow-warm-200/30">
      {/* Header — always visible */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-start gap-4 p-5 text-left"
      >
        <div className="flex-shrink-0 mt-0.5">
          <div
            className={`
              w-3 h-3 rounded-full ${statusConfig.dotColor}
              ${step.verified ? "" : step.issues.length > 0 ? "" : "animate-pulse"}
            `}
          />
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-xs font-bold uppercase tracking-wider text-warm-500">
                {step.tool_call.task_name}
              </span>
              <span
                className={`
                  inline-flex items-center px-2 py-0.5 rounded-md text-[10px] font-semibold border
                  ${statusConfig.badgeClass}
                `}
              >
                {statusConfig.label}
              </span>
              {step.attempts > 1 && (
                <span className="
                  inline-flex items-center px-2 py-0.5 rounded-md text-[10px] font-semibold border
                  bg-amber-50 text-amber-700 border-amber-200
                ">
                  {step.attempts} tries
                </span>
              )}
            </div>

            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className={`
                text-warm-400 flex-shrink-0 transition-transform duration-200
                ${expanded ? "rotate-180" : ""}
              `}
            >
              <path d="m6 9 6 6 6-6" />
            </svg>
          </div>

          <p className="text-sm font-medium text-warm-800 mt-1">
            {step.intent}
          </p>

          <p className="text-xs text-warm-500 mt-1 truncate">
            {step.raw_output.slice(0, 120)}
            {step.raw_output.length > 120 ? "..." : ""}
          </p>
        </div>
      </button>

      {/* Expanded body */}
      {expanded && (
        <div className="px-5 pb-5 space-y-4">
          <div className="border-t border-warm-100 pt-4 space-y-4">
            {/* Intent & Tool Call */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div className="p-3 rounded-xl bg-warm-50 border border-warm-100">
                <span className="text-[10px] font-bold uppercase tracking-wider text-warm-500 block mb-1">
                  Intent
                </span>
                <span className="text-sm text-warm-800">{step.intent}</span>
              </div>
              <div className="p-3 rounded-xl bg-warm-50 border border-warm-100">
                <span className="text-[10px] font-bold uppercase tracking-wider text-warm-500 block mb-1">
                  Task Name
                </span>
                <span className="text-sm text-warm-800">
                  {step.tool_call.task_name}
                </span>
              </div>
            </div>

            {/* Raw Output */}
            <div>
              <span className="text-[10px] font-bold uppercase tracking-wider text-warm-500 block mb-2">
                Raw Output
              </span>
              <pre className="text-xs text-warm-700 bg-warm-50 p-4 rounded-xl border border-warm-200 overflow-x-auto whitespace-pre-wrap break-words font-mono leading-relaxed">
                {step.raw_output}
              </pre>
            </div>

            {/* Parsed Result */}
            {Object.keys(step.parsed).length > 0 && (
              <div>
                <span className="text-[10px] font-bold uppercase tracking-wider text-warm-500 block mb-2">
                  Parsed Result
                </span>
                <pre className="text-xs text-warm-700 bg-warm-50 p-4 rounded-xl border border-warm-200 overflow-x-auto whitespace-pre-wrap break-words font-mono leading-relaxed">
                  {JSON.stringify(step.parsed, null, 2)}
                </pre>
              </div>
            )}

            {/* Issues */}
            {step.issues.length > 0 && (
              <div>
                <span className="text-[10px] font-bold uppercase tracking-wider text-rose-500 block mb-2">
                  Issues ({step.issues.length})
                </span>
                <div className="space-y-2">
                  {step.issues.map((issue, i) => (
                    <div
                      key={i}
                      className="flex items-start gap-2 p-3 rounded-xl bg-rose-50 border border-rose-200"
                    >
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="14"
                        height="14"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        className="text-rose-500 flex-shrink-0 mt-0.5"
                      >
                        <circle cx="12" cy="12" r="10" />
                        <line x1="12" x2="12" y1="8" y2="12" />
                        <line x1="12" x2="12.01" y1="16" y2="16" />
                      </svg>
                      <span className="text-xs text-rose-700 leading-relaxed">
                        {issue}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Visualization overlay */}
            {(visualization || step.visualization) && (
              <div>
                <span className="text-[10px] font-bold uppercase tracking-wider text-warm-500 block mb-2">
                  Visualization
                </span>
                <div className="relative rounded-xl border border-warm-200 overflow-hidden bg-warm-50">
                  <img
                    src={visualization ?? step.visualization}
                    alt="Step visualization"
                    className="w-full max-h-80 object-contain"
                  />
                </div>
              </div>
            )}

            {/* Transcript */}
            {step.transcript && (
              <div>
                <span className="text-[10px] font-bold uppercase tracking-wider text-warm-500 block mb-2">
                  Transcript
                </span>
                <p className="text-xs text-warm-600 leading-relaxed bg-warm-50 p-4 rounded-xl border border-warm-100">
                  {step.transcript}
                </p>
              </div>
            )}

            {/* Metadata footer */}
            <div className="flex items-center gap-4 pt-2 border-t border-warm-100">
              <span className="text-[10px] text-warm-400">
                Sub-task index: <span className="text-warm-600 font-medium">{step.sub_task_index}</span>
              </span>
              <span className="text-[10px] text-warm-400">
                Attempts: <span className="text-warm-600 font-medium">{step.attempts}</span>
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
