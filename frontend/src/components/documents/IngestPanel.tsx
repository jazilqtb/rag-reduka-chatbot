import { cn } from "@/lib/utils";
import type { IngestJobStatus } from "@/types/api";

interface Props {
  isRunning:    boolean;
  activeJob:    IngestJobStatus | null;
  error:        string | null;
  onTrigger:    () => void;
  pendingCount: number;
}

export function IngestPanel({ isRunning, activeJob, error, onTrigger, pendingCount }: Props) {
  const progress = activeJob
    ? Math.round((activeJob.files_processed / Math.max(activeJob.files_queued, 1)) * 100)
    : 0;

  return (
    <div className="rounded-xl border border-line p-5 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-medium text-ink">Ingestion ke Vector Store</h3>
          <p className="text-xs text-muted mt-0.5">
            Proses PDF menjadi embedding yang bisa di-query chatbot.
          </p>
        </div>
        <button
          type="button"
          onClick={onTrigger}
          disabled={isRunning || pendingCount === 0}
          className={cn(
            "px-4 py-2 text-sm rounded-md font-medium focus-ring transition-colors",
            isRunning || pendingCount === 0
              ? "bg-line text-muted cursor-not-allowed"
              : "bg-ai text-canvas hover:bg-ai/85",
          )}
        >
          {isRunning
            ? "Berjalan…"
            : pendingCount > 0
            ? `Ingest ${pendingCount} Dokumen Pending`
            : "Tidak Ada Pending"}
        </button>
      </div>

      {error && (
        <div className="text-sm text-error border border-error/20 bg-error/5 rounded-md p-3">
          {error}
        </div>
      )}

      {activeJob && (
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs text-muted font-mono">
            <span>
              {activeJob.status === "processing"
                ? `Memproses: ${activeJob.files_processed}/${activeJob.files_queued} file`
                : activeJob.status === "done"
                ? `Selesai: ${activeJob.files_processed} file`
                : `Gagal setelah ${activeJob.files_processed} file`}
            </span>
            <span
              className={cn(
                "font-medium",
                activeJob.status === "done"    ? "text-ai"
                : activeJob.status === "failed" ? "text-error"
                : "text-marker2",
              )}
            >
              {activeJob.status.toUpperCase()}
            </span>
          </div>

          <div className="h-1.5 w-full rounded-full bg-line overflow-hidden">
            <div
              className={cn(
                "h-full rounded-full transition-all duration-500",
                activeJob.status === "done"    ? "bg-ai"
                : activeJob.status === "failed" ? "bg-error"
                : "bg-marker2",
              )}
              style={{ width: `${progress}%` }}
            />
          </div>

          <div className="text-[10px] text-muted font-mono">
            Job ID: {activeJob.job_id}
            {activeJob.files_failed > 0 && (
              <span className="ml-2 text-error">
                · {activeJob.files_failed} file gagal
              </span>
            )}
          </div>

          {activeJob.errors.length > 0 && (
            <ul className="text-xs text-error space-y-0.5 mt-1">
              {activeJob.errors.map((e, i) => (
                <li key={i}>{e}</li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
