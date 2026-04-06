"use client";

import { useRef, useEffect } from "react";

interface Props {
  value: string;
  onChange: (v: string) => void;
  onSubmit: () => void;
  disabled: boolean;
}

export default function ChatInput({ value, onChange, onSubmit, disabled }: Props) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [value]);

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!disabled && value.trim()) onSubmit();
    }
  }

  return (
    <div className="border-t border-[#2d3148] bg-[#0f1117] px-4 py-4">
      <div className="max-w-3xl mx-auto flex items-end gap-3">
        <div className="flex-1 flex items-end rounded-xl border border-[#2d3148] bg-[#1e2130] focus-within:border-indigo-500 transition-colors px-4 py-3">
          <textarea
            ref={textareaRef}
            rows={1}
            className="flex-1 resize-none bg-transparent text-sm text-slate-200 placeholder-slate-600 outline-none leading-relaxed"
            placeholder="Ask a research question… (Shift+Enter for newline)"
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={disabled}
          />
        </div>

        <button
          onClick={onSubmit}
          disabled={disabled || !value.trim()}
          className="flex-shrink-0 w-10 h-10 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:bg-[#2d3148] disabled:cursor-not-allowed transition-colors flex items-center justify-center"
          aria-label="Send"
        >
          {disabled ? (
            // Spinner
            <svg className="w-4 h-4 animate-spin text-slate-400" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
          ) : (
            // Send arrow
            <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 19V5m-7 7 7-7 7 7" />
            </svg>
          )}
        </button>
      </div>

      <p className="mt-2 text-center text-xs text-slate-700">
        ReSearch uses Groq + Pinecone &mdash; answers grounded in retrieved papers
      </p>
    </div>
  );
}
