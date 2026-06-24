import { cn } from "@/lib/utils";
import type { IngestJobSummary } from "@/types/api";

interface Props {
  jobs:      IngestJobSummary[];
  onRefresh: () => void;
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleString("id-ID", {
    day:    "2-digit",
    month:  "short",
    hour:   "2-digit",
    minute: "2-digit",
  });
}

function durationStr(job: IngestJobSummary): string {
  if (!job.completed_at || !job.created_at) return "–";
  const secs = (new Date(job.completed_at).getTime() - new Date(job.created_at).getTime()) / 1000;
  return secs < 60 ? `${Math.round(secs)}s` : `${Math.round(secs / 60)}m ${Math.round(secs % 60)}s`;
}

export function IngestJobLog({ jobs, onRefresh }: Props) {
  if (jobs.length === 0) {
    return (
      <div className="rounded-xl border border-line p-5">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium text-ink">Riwayat Ingestion</h3>
          <button type="button" onClick={onRefresh} className="text-xs text-muted hover:text-ink focus-ring rounded">
            Refresh
          </button>
        </div>
        <p className="text-sm text-muted">Belum ada riwayat ingestion.</p>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-line p-5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-ink">Riwayat Ingestion</h3>
        <button type="button" onClick={onRefresh} className="text-xs text-muted hover:text-ink focus-ring rounded transition-colors">
          ↻ Refresh
        </button>
      </div>

      <div className="space-y-3">
        {jobs.map((job, i) => (
          <div key={job.job_id} className="relative pl-5">
            {i < jobs.length - 1 && (
              <div className="absolute left-1.5 top-4 h-full w-px bg-line" />
            )}
            <div
              className={cn(
                "absolute left-0 top-1 h-3 w-3 rounded-full border-2 border-canvas",
                job.status === "completed" || job.status === "done"
                  ? "bg-ai"
                  : job.status === "failed"
                  ? "bg-error"
                  : "bg-marker2",
              )}
            />

            <div className="pb-3">
              <div className="flex items-start justify-between gap-2">
                <div>
                  <span className="text-xs font-mono text-muted">
                    {job.job_id.slice(0, 16)}…
                  </span>
                  <span
                    className={cn(
                      "ml-2 text-[10px] font-medium uppercase font-mono",
                      job.status === "completed" || job.status === "done"
                        ? "text-ai"
                        : job.status === "failed"
                        ? "text-error"
                        : "text-marker2",
                    )}
                  >
                    {job.status}
                  </span>
                </div>
                <span className="text-[10px] text-muted shrink-0">{formatDate(job.created_at)}</span>
              </div>

              <div className="mt-1 text-xs text-muted flex gap-3 font-mono">
                <span>{job.files_processed}/{job.files_queued} file</span>
                {job.files_failed > 0 && (
                  <span className="text-error">{job.files_failed} gagal</span>
                )}
                <span>{durationStr(job)}</span>
              </div>

              {job.errors.length > 0 && (
                <ul className="mt-1 text-[10px] text-error space-y-0.5">
                  {job.errors.slice(0, 2).map((e, idx) => (
                    <li key={idx} className="truncate">{e}</li>
                  ))}
                  {job.errors.length > 2 && (
                    <li className="text-muted">+{job.errors.length - 2} error lainnya</li>
                  )}
                </ul>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
