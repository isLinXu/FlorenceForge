import { useState } from "react";

interface GoalInputProps {
  value: string;
  onChange: (value: string) => void;
}

const EXAMPLE_GOALS = [
  "detect all objects and count the cars",
  "read text and describe the scene",
  "locate the red box",
  "find all people and identify their actions",
  "count the number of tables in the room",
  "describe the overall mood of the image",
  "find the nearest exit sign",
  "identify all brands visible in the image",
];

export function GoalInput({ value, onChange }: GoalInputProps) {
  const [showExamples, setShowExamples] = useState(false);

  const handleSelectExample = (example: string) => {
    onChange(example);
    setShowExamples(false);
  };

  return (
    <div className="w-full space-y-2">
      <div className="relative">
        <textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="Describe what you want the agent to do..."
          rows={3}
          className="
            w-full px-4 py-3 rounded-xl
            bg-white border border-warm-200
            text-warm-800 placeholder-warm-400
            text-sm leading-relaxed
            focus:outline-none focus:ring-2 focus:ring-primary-300 focus:border-primary-400
            transition-all duration-200
            resize-none
          "
        />
        {value.length > 0 && (
          <button
            onClick={() => onChange("")}
            className="absolute top-3 right-3 p-1 rounded-md text-warm-400 hover:text-warm-600 hover:bg-warm-100 transition-colors"
            title="Clear"
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
            >
              <path d="M18 6 6 18" />
              <path d="m6 6 12 12" />
            </svg>
          </button>
        )}
      </div>

      <div className="relative">
        <button
          onClick={() => setShowExamples(!showExamples)}
          className="
            inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg
            text-xs font-medium text-warm-600
            bg-warm-100 hover:bg-warm-200
            border border-warm-200
            transition-colors duration-200
          "
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="13"
            height="13"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M12 5v14" />
            <path d="m19 12-7 7-7-7" />
          </svg>
          Example goals
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="13"
            height="13"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className={`
              transition-transform duration-200
              ${showExamples ? "rotate-180" : ""}
            `}
          >
            <path d="m6 9 6 6 6-6" />
          </svg>
        </button>

        {showExamples && (
          <div className="
            absolute z-10 mt-2 w-full max-w-md
            bg-white rounded-xl border border-warm-200 shadow-lg shadow-warm-200/50
            py-2
          ">
            {EXAMPLE_GOALS.map((example, idx) => (
              <button
                key={idx}
                onClick={() => handleSelectExample(example)}
                className="
                  w-full text-left px-4 py-2
                  text-sm text-warm-700
                  hover:bg-warm-50
                  transition-colors duration-150
                "
              >
                {example}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
