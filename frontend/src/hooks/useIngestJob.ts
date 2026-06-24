import { useCallback, useMemo, useRef, useState } from "react";
import { ApiClient } from "@/api/client";
import {
  triggerIngest,
  getIngestJobStatus,
  listIngestJobs,
} from "@/api/documents";
import type {
  AppSettings,
  IngestJobStatus,
  IngestJobSummary,
} from "@/types/api";

const POLL_INTERVAL_MS = 3000;

export function useIngestJob(settings: AppSettings) {
  const [activeJob,  setActiveJob]  = useState<IngestJobStatus | null>(null);
  const [jobHistory, setJobHistory] = useState<IngestJobSummary[]>([]);
  const [isRunning,  setIsRunning]  = useState(false);
  const [error,      setError]      = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const client = useMemo(
    () => new ApiClient({ baseUrl: settings.baseUrl, apiKey: settings.apiKey }),
    [settings.baseUrl, settings.apiKey],
  );

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const fetchJobHistory = useCallback(async () => {
    try {
      const res = await listIngestJobs(client, 10);
      setJobHistory(res.jobs);
    } catch {
      // Silently ignore — history bukan critical path
    }
  }, [client]);

  const startPolling = useCallback(
    (job_id: string) => {
      stopPolling();
      pollRef.current = setInterval(async () => {
        try {
          const statusRes = await getIngestJobStatus(client, job_id);
          setActiveJob(statusRes);
          if (statusRes.status === "done" || statusRes.status === "failed") {
            stopPolling();
            setIsRunning(false);
            void fetchJobHistory();
          }
        } catch {
          stopPolling();
          setIsRunning(false);
        }
      }, POLL_INTERVAL_MS);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [client, stopPolling],
  );

  const triggerAll = useCallback(async () => {
    if (isRunning) return;
    setError(null);
    setIsRunning(true);
    setActiveJob(null);
    try {
      const res = await triggerIngest(client);
      setActiveJob({
        job_id:          res.job_id,
        status:          "processing",
        files_queued:    res.files_queued,
        files_processed: 0,
        files_failed:    0,
        errors:          [],
        created_at:      new Date().toISOString(),
      });
      startPolling(res.job_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Gagal memulai ingestion.");
      setIsRunning(false);
    }
  }, [client, isRunning, startPolling]);

  return {
    activeJob, jobHistory, isRunning, error,
    triggerAll, fetchJobHistory,
  };
}
