// ============================================================================
// Document management API wrappers
// Semua fungsi wrap ApiClient dari client.ts.
// ============================================================================
import type { ApiClient } from "./client";
import type {
  DocumentListResponse,
  DocumentItem,
  UploadResponse,
  IngestTriggerResponse,
  IngestJobStatus,
  IngestJobListResponse,
  DeleteDocumentResponse,
  HealthDetailResponse,
} from "@/types/api";

// ── Documents ────────────────────────────────────────────────────────────────

export async function listDocuments(
  client:       ApiClient,
  page:         number = 1,
  limit:        number = 50,
  jenis_ujian?: string,
): Promise<DocumentListResponse> {
  const params = new URLSearchParams({
    page:  String(page),
    limit: String(limit),
  });
  if (jenis_ujian) params.set("jenis_ujian", jenis_ujian);

  const res = await client.get<{
    total: number;
    page:  number;
    limit: number;
    items: Array<{ ingested?: boolean; status?: string } & Record<string, unknown>>;
  }>(`/v1/documents?${params}`);

  return {
    ...res,
    items: res.items.map((item) => ({
      ...item,
      status: (item.status as DocumentItem["status"]) ??
              (item.ingested ? "ingested" : "uploaded"),
    })) as DocumentItem[],
  };
}

export async function uploadDocument(
  client:      ApiClient,
  file:        File,
  doc_type:    "soal" | "jawaban",
  jenis_ujian: string,
): Promise<UploadResponse> {
  // Pakai fetch langsung karena ApiClient.post hanya handle JSON.
  // Untuk multipart/form-data perlu manual.
  const cfg = (client as unknown as { cfg: { baseUrl: string; apiKey: string } }).cfg;
  const url  = `${cfg.baseUrl.replace(/\/$/, "")}/v1/documents/upload`;

  const form = new FormData();
  form.append("file", file);
  form.append("doc_type", doc_type);
  form.append("jenis_ujian", jenis_ujian);

  const response = await fetch(url, {
    method:  "POST",
    headers: { "X-API-Key": cfg.apiKey },
    body:    form,
  });

  let payload: unknown;
  try { payload = await response.json(); } catch { payload = {}; }

  if (!response.ok) {
    const err = (payload as { detail?: { message?: string; error?: string } }).detail ?? payload;
    const msg = (err as { message?: string })?.message ?? `HTTP ${response.status}`;
    throw new Error(msg);
  }
  return payload as UploadResponse;
}

export async function deleteDocument(
  client:  ApiClient,
  file_id: string,
): Promise<DeleteDocumentResponse> {
  return client.delete<DeleteDocumentResponse>(`/v1/documents/${file_id}`);
}

// ── Ingestion ─────────────────────────────────────────────────────────────────

export async function triggerIngest(
  client:    ApiClient,
  file_ids?: string[],
): Promise<IngestTriggerResponse> {
  const body = file_ids?.length
    ? { file_ids, ingest_all_pending: false }
    : { ingest_all_pending: true };
  return client.post<IngestTriggerResponse>("/v1/documents/ingest", body);
}

export async function getIngestJobStatus(
  client: ApiClient,
  job_id: string,
): Promise<IngestJobStatus> {
  return client.get<IngestJobStatus>(`/v1/documents/ingest/${job_id}`);
}

export async function listIngestJobs(
  client: ApiClient,
  limit:  number = 20,
): Promise<IngestJobListResponse> {
  return client.get<IngestJobListResponse>(`/v1/documents/ingest?limit=${limit}`);
}

// ── Health ────────────────────────────────────────────────────────────────────

export async function getHealthDetail(
  client: ApiClient,
): Promise<HealthDetailResponse> {
  return client.get<HealthDetailResponse>("/v1/health/detailed");
}
