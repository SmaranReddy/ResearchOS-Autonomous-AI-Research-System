"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import ChatMessage from "@/components/ChatMessage";
import ChatInput from "@/components/ChatInput";
import type { Message, HistoryEntry } from "@/types/chat";

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center px-4 py-16">
      <div className="w-14 h-14 rounded-2xl bg-indigo-900 border border-indigo-700 flex items-center justify-center mb-6">
        <span className="text-indigo-300 text-2xl font-bold">R</span>
      </div>
      <h1 className="text-2xl font-semibold text-slate-200 mb-2">ReSearch</h1>
      <p className="text-slate-500 max-w-sm text-sm leading-relaxed">
        Ask any research question. I&apos;ll retrieve relevant papers, rank the
        evidence, and synthesise a grounded answer.
      </p>
      <p className="mt-6 text-xs text-slate-600 max-w-xs">
        ⚡ First query may take a few extra seconds to warm up — subsequent queries are faster.
      </p>
      <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-md">
        {[
          "Explain transformers in deep learning",
          "Compare BERT and GPT architectures",
          "What is retrieval-augmented generation?",
          "How does reinforcement learning from human feedback work?",
        ].map((q) => (
          <button
            key={q}
            className="text-left text-xs text-slate-400 bg-[#1e2130] border border-[#2d3148] rounded-lg px-3 py-2.5 hover:border-indigo-600 hover:text-slate-200 transition-colors"
            onClick={() => {
              window.dispatchEvent(new CustomEvent("suggestion", { detail: q }));
            }}
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState<HistoryEntry[]>([]);

  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    const handler = (e: Event) => {
      setInput((e as CustomEvent<string>).detail);
    };
    window.addEventListener("suggestion", handler);
    return () => window.removeEventListener("suggestion", handler);
  }, []);

  const submit = useCallback(async () => {
    const query = input.trim();
    if (!query || loading) return;

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: query,
    };

    const assistantId = crypto.randomUUID();
    const assistantPlaceholder: Message = {
      id: assistantId,
      role: "assistant",
      content: "",
      streaming: true,
    };

    setMessages((prev) => [...prev, userMsg, assistantPlaceholder]);
    setInput("");
    setLoading(true);

    const start = Date.now();
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 300_000);

    try {
      const res = await fetch("/api/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, history }),
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      if (!res.ok || !res.body) {
        const err = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, streaming: false, content: err.error ?? "Unexpected error", error: true }
              : m
          )
        );
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          let event: Record<string, unknown>;
          try {
            event = JSON.parse(line.slice(6));
          } catch {
            continue;
          }

          if (event.type === "token") {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? { ...m, content: m.content + (event.content as string) }
                  : m
              )
            );
          } else if (event.type === "refine") {
            // Post-stream critique: backend replaced a low-quality answer.
            // Swap in the refined content and mark the message so the UI
            // can show a "refined" badge.
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? { ...m, content: event.content as string, refined: true }
                  : m
              )
            );
          } else if (event.type === "done") {
            const latencyMs = Date.now() - start;
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? {
                      ...m,
                      streaming: false,
                      confidence: event.confidence as number,
                      status: event.status as Message["status"],
                      sources: event.sources as Message["sources"],
                      latency: event.latency as Message["latency"],
                      latencyMs,
                      decisionTrace: event.decision_trace as Message["decisionTrace"],
                      critiqueType: event.critique_type as string | undefined,
                      critiqueScore: event.critique_score as number | undefined,
                      retried: event.retried as boolean | undefined,
                      decisionStrength: event.decision_strength as string | undefined,
                    }
                  : m
              )
            );
            setHistory(event.history as HistoryEntry[]);
          } else if (event.type === "error") {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? { ...m, streaming: false, content: event.message as string, error: true }
                  : m
              )
            );
          }
        }
      }
    } catch (err) {
      clearTimeout(timeoutId);
      const isTimeout = err instanceof DOMException && err.name === "AbortError";
      const content = isTimeout
        ? "Request timed out after 300s — the backend may be overloaded."
        : err instanceof Error
        ? err.message
        : "Network error — is the backend running?";
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? { ...m, streaming: false, content, error: true }
            : m
        )
      );
    } finally {
      setLoading(false);
    }
  }, [input, history, loading]);

  const hasMessages = messages.length > 0;

  return (
    <div className="flex flex-col h-screen bg-[#0f1117]">
      <header className="flex-shrink-0 flex items-center justify-between px-6 py-4 border-b border-[#2d3148]">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-indigo-900 border border-indigo-700 flex items-center justify-center">
            <span className="text-indigo-300 text-xs font-bold">R</span>
          </div>
          <span className="text-slate-200 font-semibold text-sm">ReSearch</span>
        </div>
        {hasMessages && (
          <button
            onClick={() => {
              setMessages([]);
              setHistory([]);
            }}
            className="text-xs text-slate-600 hover:text-slate-400 transition-colors"
          >
            Clear chat
          </button>
        )}
      </header>

      <main className="flex-1 overflow-y-auto">
        {!hasMessages ? (
          <EmptyState />
        ) : (
          <div className="max-w-3xl mx-auto px-4 py-6 space-y-6">
            {messages.map((msg) => (
              <ChatMessage key={msg.id} message={msg} />
            ))}
            <div ref={bottomRef} />
          </div>
        )}
      </main>

      <ChatInput
        value={input}
        onChange={setInput}
        onSubmit={submit}
        disabled={loading}
      />
    </div>
  );
}
