export type Role = "user" | "assistant";

export interface Source {
  title: string;
  url: string;
}

export interface LatencyInfo {
  retrieve_ms: number;
  rerank_ms: number;
  llm_ms: number;
}

export type ResponseStatus = "success" | "low_confidence" | "fallback" | "error";

export interface DecisionTrace {
  retrieval_quality: string;
  action: string;
  confidence_reasoning: string;
}

export interface Message {
  id: string;
  role: Role;
  content: string;
  confidence?: number;
  status?: ResponseStatus;
  sources?: Source[];
  latency?: LatencyInfo;
  latencyMs?: number;         // total round-trip wall-clock time
  streaming?: boolean;        // true while tokens are still arriving
  error?: boolean;
  decisionTrace?: DecisionTrace;
  critiqueType?: string;      // "incomplete" | "incorrect" | "not_grounded" | "good" | ""
  critiqueScore?: number;     // 0–1
  retried?: boolean;          // true if self-correction retry ran
  decisionStrength?: string;  // "strong" | "moderate" | "weak"
}

export interface HistoryEntry {
  query: string;
  answer: string;
}

export interface QueryRequest {
  query: string;
  history: HistoryEntry[] | null;
}

export interface QueryResponse {
  answer: string;
  confidence: number;
  status: ResponseStatus;
  history: HistoryEntry[];
  sources: Source[];
  latency: LatencyInfo;
  decision_trace: DecisionTrace;
}
