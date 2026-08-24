import { useState } from "react";
import type { StepRecord } from "../types/agentic";

interface OrchestratorTimelineProps {
  steps: StepRecord[];
  activeStepIndex: number;
  onStepClick?: (index: number) => void;
}

const STEP_LABELS = [
  "DECOMPOSE",
  "DETECT",
  "COUNT",
  "READ_TEXT",
  "VERIFY",
  "DONE",
];

function getStepIcon(step: StepRecord, isActive: boolean) {
  if (step.verified) {
    return (
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M20 6 9 17l-5-5" />
      </svg>
    );
  }
  if (step.issues.length > 0) {
    return (
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M18 6 6 18" />
        <path d="m6 6 12 12" />
      </svg>
    );
  }
  if (isActive) {
    return (
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M21 12a9 9 0 1 1-6.219-8.56" />
      </svg>
    );
  }
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="12" r="10" />
    </svg>
  );
}

function getStepIconColors(step: StepRecord, isActive: boolean) {
  if (step.verified) {
    return "bg-emerald-50 text-emerald-600 border-emerald-200";
  }
  if (step.issues.length > 0) {
    return "bg-rose-50 text-rose-600 border-rose-200";
  }
  if (isActive) {
    return "bg-primary-50 text-primary-600 border-primary-300 animate-pulse";
  }
  return "bg-warm-100 text-warm-400 border-warm-200";
}

function getConnectorColor(step: StepRecord, _nextStep?: StepRecord) {
  if (step.verified) {
    return "bg-emerald-300";
  }
  if (step.issues.length > 0) {
    return "bg-rose-200";
  }
  return "bg-warm-200";
}

export function OrchestratorTimeline({
  steps,
  activeStepIndex,
  onStepClick,
}: OrchestratorTimelineProps) {
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);

  const handleStepClick = (index: number) => {
    setExpandedIndex(expandedIndex === index ? null : index);
    onStepClick?.(index);
  };

  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-sm font-semibold text-warm-800 tracking-wide uppercase">
          Execution Timeline
        </h3>
        <span className="text-xs text-warm-500 font-medium">
          {steps.filter((s) => s.verified).length} / {steps.length} completed
        </span>
      </div>

      <div className="relative">
        {steps.map((step, index) => {
          const isActive = index === activeStepIndex;
          const isExpanded = expandedIndex === index;
          const label = STEP_LABELS[index] ?? step.tool_call.task_name.toUpperCase();
          const hasDetails =
            step.raw_output || step.issues.length > 0 || step.attempts > 1;

          return (
            <div key={index} className="relative">
              {/* Connector line */}
              {index < steps.length - 1 && (
                <div
                  className={`
                    absolute left-5 top-10 w-px h-6
                    ${getConnectorColor(step, steps[index + 1])}
                    transition-colors duration-300
                  `}
                />
              )}

              <div
                onClick={() => handleStepClick(index)}
                className={`
                  group flex items-start gap-4 p-4 rounded-xl
                  cursor-pointer transition-all duration-200
                  ${isActive ? "bg-warm-50 border border-warm-200" : "hover:bg-warm-50/50 border border-transparent"}
                  ${isExpanded ? "mb-2" : ""}
                `}
              >
                {/* Status icon */}
                <div
                  className={`
                    flex-shrink-0 w-10 h-10 rounded-full border-2
                    flex items-center justify-center
                    transition-all duration-300
                    ${getStepIconColors(step, isActive)}
                  `}
                >
                  {getStepIcon(step, isActive)}
                </div>

                {/* Content */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span
                        className={`
                          text-xs font-bold uppercase tracking-wider
                          ${isActive ? "text-primary-700" : "text-warm-600"}
                        `}
                      >
                        {label}
                      </span>
                      {step.attempts > 1 && (
                        <span className="
                          px-1.5 py-0.5 rounded text-[10px] font-medium
                          bg-amber-50 text-amber-700 border border-amber-200
                        ">
                          {step.attempts} attempts
                        </span>
                      )}
                    </div>
                    {hasDetails && (
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
                        className={`
                          text-warm-400 transition-transform duration-200
                          ${isExpanded ? "rotate-180" : ""}
                        `}
                      >
                        <path d="m6 9 6 6 6-6" />
                      </svg>
                    )}
                  </div>

                  <p className="text-sm text-warm-700 mt-1 truncate">
                    {step.intent}
                  </p>

                  <div className="flex items-center gap-3 mt-2">
                    <span className="text-xs text-warm-500">
                      Tool: <span className="text-warm-700 font-medium">{step.tool_call.task_name}</span>
                    </span>
                    <span className="text-warm-300">·</span>
                    <span
                      className={`
                        text-xs font-medium
                        ${step.verified ? "text-emerald-600" : step.issues.length > 0 ? "text-rose-600" : "text-warm-500"}
                      `}
                    >
                      {step.verified
                        ? "Verified"
                        : step.issues.length > 0
                        ? `${step.issues.length} issue${step.issues.length > 1 ? "s" : ""}`
                        : "Pending"}
                    </span>
                  </div>
                </div>
              </div>

              {/* Expanded details */}
              {isExpanded && hasDetails && (
                <div className="ml-14 mb-4 p-4 rounded-xl bg-warm-50 border border-warm-200 space-y-3">
                  {step.raw_output && (
                    <div>
                      <span className="text-[10px] font-bold uppercase tracking-wider text-warm-500 block mb-1">
                        Raw Output
                      </span>
                      <pre className="text-xs text-warm-700 bg-white p-3 rounded-lg border border-warm-200 overflow-x-auto whitespace-pre-wrap break-words font-mono leading-relaxed">
                        {step.raw_output}
                      </pre>
                    </div>
                  )}

                  {step.issues.length > 0 && (
                    <div>
                      <span className="text-[10px] font-bold uppercase tracking-wider text-rose-500 block mb-1">
                        Issues
                      </span>
                      <ul className="space-y-1">
                        {step.issues.map((issue, i) => (
                          <li
                            key={i}
                            className="text-xs text-rose-700 bg-rose-50 px-3 py-2 rounded-lg border border-rose-200"
                          >
                            {issue}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {step.attempts > 1 && (
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] font-bold uppercase tracking-wider text-warm-500">
                        Retry Count:
                      </span>
                      <span className="text-xs text-warm-700 font-medium">
                        {step.attempts}
                      </span>
                    </div>
                  )}

                  {step.transcript && (
                    <div>
                      <span className="text-[10px] font-bold uppercase tracking-wider text-warm-500 block mb-1">
                        Transcript
                      </span>
                      <p className="text-xs text-warm-600 leading-relaxed">
                        {step.transcript}
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
