// ============================================================================
// Type definitions matching backend Pydantic schemas (src/domain/schemas/)
// Update when backend schemas change.
// ============================================================================

// ── Chat ────────────────────────────────────────────────────────────────────

export interface SourceItem {
  subject:     string;
  jenis_ujian: string;
  id_soal:     string;
  source:      string;
}

export interface ChatRequest {
  user_id:     string;
  session_id?: string | null;
  query:       string;
}

export interface ResponseMeta {
  latency_ms: number;
}

export interface ChatResponse {
  session_id: string;
  answer:     string;
  sources:    SourceItem[];
  meta?:      ResponseMeta | null;
}


// ── Health ──────────────────────────────────────────────────────────────────

export interface HealthResponse {
  status:    "ok" | "degraded" | "down";
  timestamp: string;
}


// ── Error ───────────────────────────────────────────────────────────────────

export interface ErrorResponse {
  error:   string;
  message: string;
  detail?: unknown;
}


// ── Frontend-local types ────────────────────────────────────────────────────

export type MessageRole = "user" | "assistant";

export interface Message {
  id:        string;
  role:      MessageRole;
  content:   string;
  sources?:  SourceItem[];
  latency?:  number;
  error?:    string;
  pending?:  boolean;
}

export interface AppSettings {
  baseUrl: string;
  apiKey:  string;
  userId:  string;
}