"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import ConfidenceBadge from "./ConfidenceBadge";
import type { Message } from "@/types/chat";

interface Props {
  message: Message;
}

// ---------------------------------------------------------------------------
// Action label mapping
// ---------------------------------------------------------------------------

const ACTION_LABELS: Record<string, string> = {
  used_existing_knowledge:             "answered using existing knowledge",
  used_existing_knowledge_borderline:  "answered with limited confidence",
  triggered_ingestion_low_docs:        "searched for additional sources",
  triggered_ingestion_low_relevance:   "searched for additional sources",
  triggered_ingestion_low_confidence:  "searched for additional sources",
  fallback_no_answer:                  "answered using general knowledge",
};

function actionLabel(action: string): string {
  return ACTION_LABELS[action] ?? action.replace(/_/g, " ");
}

// ---------------------------------------------------------------------------
// Single summary line
// ---------------------------------------------------------------------------

function confidenceLevel(v: number): string {
  if (v >= 0.85) return "Very high confidence";
  if (v >= 0.70) return "High confidence";
  if (v >= 0.40) return "Moderate confidence";
  return "Low confidence";
}

function formatSummary(message: Message): string {
  const parts: string[] = [];

  if (typeof message.confidence === "number") {
    parts.push(confidenceLevel(message.confidence));
  }

  // Dominant quality signal overrides action when it indicates a problem
  if (message.critiqueType === "incomplete") {
    parts.push("answer may be incomplete");
  } else if (message.critiqueType === "not_grounded") {
    parts.push("answer may not be well grounded in sources");
  } else if (message.decisionStrength === "weak") {
    parts.push("low confidence in this answer");
  } else if (message.decisionTrace?.action) {
    parts.push(actionLabel(message.decisionTrace.action));
  }

  return parts.join(" — ");
}

// ---------------------------------------------------------------------------
// Subcomponents
// ---------------------------------------------------------------------------

function UserMessage({ content }: { content: string }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[75%] rounded-2xl rounded-tr-sm bg-indigo-600 px-4 py-3 text-sm text-white leading-relaxed">
        {content}
      </div>
    </div>
  );
}

function SourceChips({ sources }: { sources: NonNullable<Message["sources"]> }) {
  if (sources.length === 0) return null;
  return (
    <div className="mt-3 pt-3 border-t border-white/5">
      <p className="text-xs text-slate-500 uppercase tracking-wider mb-2">Sources</p>
      <div className="flex flex-wrap gap-1.5">
        {sources.map((s, i) =>
          s.url ? (
            <a
              key={i}
              href={s.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs px-2 py-1 rounded-md border border-indigo-700/60 text-indigo-300 hover:border-indigo-500 hover:text-indigo-200 transition-colors truncate max-w-[220px]"
              title={s.title}
            >
              📄 {s.title}
            </a>
          ) : (
            <span
              key={i}
              className="text-xs px-2 py-1 rounded-md border border-[#2d3148] text-slate-400 truncate max-w-[220px] opacity-60 cursor-default"
              title={s.title}
            >
              {s.title}
            </span>
          )
        )}
      </div>
    </div>
  );
}

function StatusBanner({ status }: { status: NonNullable<Message["status"]> }) {
  if (status === "success") return null;

  const config = {
    low_confidence: {
      icon: "⚠",
      text: "Low confidence — the retrieved papers may not fully cover this topic.",
      cls: "bg-amber-950/60 border-amber-700/50 text-amber-300",
    },
    fallback: {
      icon: "⊘",
      text: "Retrieved sources were insufficient — this answer draws on general knowledge and may be less precise.",
      cls: "bg-orange-950/60 border-orange-700/50 text-orange-300",
    },
    error: {
      icon: "✕",
      text: "The pipeline encountered an error.",
      cls: "bg-red-950/60 border-red-800/50 text-red-300",
    },
  }[status];

  if (!config) return null;

  return (
    <div className="mt-3 pt-3 border-t border-white/5">
      <div className={`flex items-start gap-2 text-xs rounded-lg border px-3 py-2 ${config.cls}`}>
        <span className="flex-shrink-0 font-bold">{config.icon}</span>
        <span>{config.text}</span>
      </div>
    </div>
  );
}

/** Contextual warnings derived from critique_type and decision_strength */
function QualityWarnings({
  critiqueType,
  decisionStrength,
}: {
  critiqueType?: string;
  decisionStrength?: string;
}) {
  const warnings: string[] = [];

  if (critiqueType === "incomplete")
    warnings.push("Answer may be incomplete");
  if (critiqueType === "not_grounded")
    warnings.push("Answer may not be well grounded in sources");
  if (decisionStrength === "weak")
    warnings.push("Low confidence in this answer");

  if (warnings.length === 0) return null;

  return (
    <div className="mt-2 flex flex-col gap-1">
      {warnings.map((w) => (
        <div
          key={w}
          className="flex items-start gap-1.5 text-xs text-amber-400/80 bg-amber-950/30 border border-amber-700/30 rounded-md px-2.5 py-1.5"
        >
          <span className="flex-shrink-0">⚠️</span>
          <span>{w}</span>
        </div>
      ))}
    </div>
  );
}

/** Shows the decision action taken by the pipeline */
function ActionRow({ action }: { action: string }) {
  if (!action) return null;
  const isIngestion = action.startsWith("triggered_ingestion");
  return (
    <p className={`text-xs mt-1.5 ${isIngestion ? "text-indigo-400/70" : "text-slate-600"}`}>
      {isIngestion ? "⟳ " : ""}
      {actionLabel(action)}
    </p>
  );
}

/** Self-correction badge — only shown when retry ran */
function RetryBadge({ retried }: { retried?: boolean }) {
  if (!retried) return null;
  return (
    <div className="mt-2 flex items-center gap-1.5 text-xs text-emerald-400/80 bg-emerald-950/30 border border-emerald-700/30 rounded-md px-2.5 py-1.5 w-fit">
      <span>🔄</span>
      <span>Answer improved after self-correction</span>
    </div>
  );
}

/** Refined badge — shown when post-stream critique replaced a low-quality answer */
function RefinedBadge({ refined }: { refined?: boolean }) {
  if (!refined) return null;
  return (
    <div className="mt-2 flex items-center gap-1.5 text-xs text-amber-400/80 bg-amber-950/30 border border-amber-700/30 rounded-md px-2.5 py-1.5 w-fit">
      <span>✏️</span>
      <span>Answer refined — initial response had low source grounding</span>
    </div>
  );
}

/** Debug panel — raw signals, shown only when toggle is ON */
function DebugPanel({ message }: { message: Message }) {
  const rows: { label: string; value: string }[] = [
    {
      label: "Action",
      value: message.decisionTrace?.action ?? "—",
    },
    {
      label: "Retrieval quality",
      value: message.decisionTrace?.retrieval_quality ?? "—",
    },
    {
      label: "Confidence reasoning",
      value: message.decisionTrace?.confidence_reasoning ?? "—",
    },
    {
      label: "Decision strength",
      value: message.decisionStrength ?? "—",
    },
    {
      label: "Critique type",
      value: message.critiqueType || "—",
    },
    {
      label: "Critique score",
      value:
        typeof message.critiqueScore === "number"
          ? message.critiqueScore.toFixed(2)
          : "—",
    },
    {
      label: "Retried",
      value: message.retried ? "yes" : "no",
    },
    ...(message.latency
      ? [
          { label: "Retrieve", value: `${message.latency.retrieve_ms}ms` },
          { label: "Rerank",   value: `${message.latency.rerank_ms}ms` },
          { label: "LLM",      value: `${message.latency.llm_ms}ms` },
        ]
      : []),
    ...(typeof message.latencyMs === "number"
      ? [{ label: "Total", value: `${(message.latencyMs / 1000).toFixed(1)}s` }]
      : []),
  ];

  return (
    <div className="mt-3 pt-3 border-t border-white/5">
      <p className="text-xs text-slate-500 uppercase tracking-wider mb-2">Debug</p>
      <div className="rounded-lg border border-[#2d3148] bg-[#151722] divide-y divide-white/5 text-xs font-mono">
        {rows.map(({ label, value }) => (
          <div key={label} className="flex gap-3 px-3 py-1.5">
            <span className="w-36 flex-shrink-0 text-slate-600">{label}</span>
            <span className="text-slate-400 break-words">{value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function LatencyRow({ latency }: { latency: NonNullable<Message["latency"]> }) {
  const parts: { label: string; ms: number }[] = [
    { label: "Retrieve", ms: latency.retrieve_ms },
    { label: "Rerank",   ms: latency.rerank_ms },
    { label: "LLM",      ms: latency.llm_ms },
  ].filter((p) => p.ms > 0);

  if (parts.length === 0) return null;

  return (
    <div className="mt-1 ml-1 flex gap-3 flex-wrap">
      {parts.map(({ label, ms }) => (
        <span key={label} className="text-xs text-slate-600">
          {label}: {ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`}
        </span>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main assistant message
// ---------------------------------------------------------------------------

function AssistantMessage({ message }: { message: Message }) {
  const [showDebug, setShowDebug] = useState(false);

  const isThinking = message.streaming && message.content.length === 0;
  const isDone = !message.streaming && !message.error;

  return (
    <div className="flex justify-start">
      <div className="mr-3 mt-1 flex-shrink-0 w-7 h-7 rounded-full bg-indigo-900 border border-indigo-700 flex items-center justify-center">
        <span className="text-indigo-300 text-xs font-bold">R</span>
      </div>

      <div className="max-w-[80%]">
        <div
          className={`rounded-2xl rounded-tl-sm px-5 py-4 text-sm leading-relaxed ${
            message.error
              ? "bg-red-950 border border-red-800 text-red-300"
              : "bg-[#1e2130] border border-[#2d3148] text-slate-300"
          }`}
        >
          {isThinking ? (
            <div className="flex flex-col gap-2 py-0.5 min-w-[5rem]">
              <span className="flex items-end gap-1.5">
                <span className="typing-dot w-2 h-2 rounded-full bg-indigo-400 inline-block" />
                <span className="typing-dot w-2 h-2 rounded-full bg-indigo-400 inline-block" />
                <span className="typing-dot w-2 h-2 rounded-full bg-indigo-400 inline-block" />
              </span>
              <span className="text-xs text-slate-500 tracking-wide">Thinking...</span>
            </div>
          ) : (
            <div className="content-fade-in">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  strong: ({ children }) => (
                    <strong className="text-slate-100 font-semibold">{children}</strong>
                  ),
                  ul: ({ children }) => (
                    <ul className="mt-2 space-y-1 list-disc list-inside">{children}</ul>
                  ),
                  li: ({ children }) => <li className="text-slate-300">{children}</li>,
                  p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                }}
              >
                {message.content}
              </ReactMarkdown>

              {/* Blinking cursor while streaming */}
              {message.streaming && (
                <span className="inline-block w-0.5 h-4 bg-indigo-400 ml-0.5 animate-pulse align-text-bottom" />
              )}

              {isDone && (
                <>
                  {/* Confidence bar */}
                  {typeof message.confidence === "number" && (
                    <ConfidenceBadge value={message.confidence} />
                  )}

                  {/* Single summary line — replaces separate reasoning / action / warnings */}
                  {formatSummary(message) && (
                    <p className="mt-1.5 text-xs text-slate-500 leading-relaxed">
                      {formatSummary(message)}
                    </p>
                  )}

                  {/* Self-correction and post-stream refinement badges */}
                  <RetryBadge retried={message.retried} />
                  <RefinedBadge refined={message.refined} />

                  {/* Sources */}
                  {message.sources && message.sources.length > 0 && (
                    <SourceChips sources={message.sources} />
                  )}

                  {/* Pipeline status banner */}
                  {message.status && message.status !== "success" && (
                    <StatusBanner status={message.status} />
                  )}

                  {/* Debug toggle — dev only */}
                  {process.env.NODE_ENV === "development" && (
                    <>
                      <div className="mt-3 pt-3 border-t border-white/5">
                        <button
                          onClick={() => setShowDebug((v) => !v)}
                          className="text-xs text-slate-600 hover:text-slate-400 transition-colors"
                        >
                          {showDebug ? "Hide debug info" : "Show debug info"}
                        </button>
                      </div>

                      {showDebug && <DebugPanel message={message} />}
                    </>
                  )}
                </>
              )}
            </div>
          )}
        </div>

        {/* Per-stage latency — dev only */}
        {process.env.NODE_ENV === "development" && isDone && !showDebug && message.latency && (
          <LatencyRow latency={message.latency} />
        )}

        {/* Generation time — always visible */}
        {isDone && typeof message.latencyMs === "number" && (
          <p className="mt-1 ml-1 text-xs text-slate-600">
            Generated in {(message.latencyMs / 1000).toFixed(1)}s
          </p>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

export default function ChatMessage({ message }: Props) {
  return message.role === "user" ? (
    <UserMessage content={message.content} />
  ) : (
    <AssistantMessage message={message} />
  );
}
