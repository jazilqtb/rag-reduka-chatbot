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


// ── Documents ────────────────────────────────────────────────────────────────

export type DocType = "soal" | "jawaban";
export type DocStatus = "uploaded" | "ingested" | "failed";

export interface DocumentItem {
  file_id:      string;
  filename:     string;
  doc_type:     DocType;
  jenis_ujian:  string;
  size_bytes:   number;
  status:       DocStatus;
  chunk_count?: number | null;
  uploaded_at:  string;
  ingested_at?: string | null;
}

export interface DocumentListResponse {
  total: number;
  page:  number;
  limit: number;
  items: DocumentItem[];
}

export interface UploadResponse {
  file_id:     string;
  filename:    string;
  doc_type:    DocType;
  jenis_ujian: string;
  size_bytes:  number;
}

export interface IngestTriggerResponse {
  job_id:       string;
  files_queued: number;
}

export interface IngestJobStatus {
  job_id:          string;
  status:          "processing" | "done" | "failed";
  files_queued:    number;
  files_processed: number;
  files_failed:    number;
  errors:          string[];
  created_at:      string;
  completed_at?:   string | null;
}

export interface IngestJobSummary {
  job_id:          string;
  status:          string;
  files_queued:    number;
  files_processed: number;
  files_failed:    number;
  errors:          string[];
  created_at:      string;
  completed_at?:   string | null;
}

export interface IngestJobListResponse {
  jobs:  IngestJobSummary[];
  total: number;
}

export interface DeleteDocumentResponse {
  file_id:               string;
  deleted_from_storage:  boolean;
  deleted_from_vectordb: boolean;
  chunks_removed:        number;
  message:               string;
}

// ── Health Detail ────────────────────────────────────────────────────────────

export interface HealthComponent {
  status:      "ok" | "error";
  latency_ms?: number;
  detail:      string;
}

export interface HealthDetailResponse {
  status:     "ok" | "degraded" | "down";
  timestamp:  string;
  components: {
    postgres?: HealthComponent;
    redis?:    HealthComponent;
    chromadb?: HealthComponent;
    gemini?:   HealthComponent;
    storage?:  HealthComponent;
    [key: string]: HealthComponent | undefined;
  };
}

// ── Navigation ───────────────────────────────────────────────────────────────

export type AppPage = "chat" | "documents" | "getting-started";