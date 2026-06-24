import { useCallback, useEffect, useMemo, useState } from "react";
import { ApiClient }    from "@/api/client";
import {
  listDocuments,
  uploadDocument,
  deleteDocument,
} from "@/api/documents";
import type { AppSettings, DocumentItem } from "@/types/api";

interface UploadState {
  filename: string;
  status:   "pending" | "uploading" | "done" | "error";
  error?:   string;
}

export function useDocuments(settings: AppSettings) {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [total,     setTotal]     = useState(0);
  const [loading,   setLoading]   = useState(false);
  const [error,     setError]     = useState<string | null>(null);
  const [uploads,   setUploads]   = useState<UploadState[]>([]);
  const [deleting,  setDeleting]  = useState<Set<string>>(new Set());

  const client = useMemo(
    () => new ApiClient({ baseUrl: settings.baseUrl, apiKey: settings.apiKey }),
    [settings.baseUrl, settings.apiKey],
  );

  const fetchDocuments = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await listDocuments(client);
      setDocuments(res.items);
      setTotal(res.total);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Gagal mengambil daftar dokumen.");
    } finally {
      setLoading(false);
    }
  }, [client]);

  useEffect(() => { fetchDocuments(); }, [fetchDocuments]);

  const upload = useCallback(
    async (files: File[], jenis_ujian: string) => {
      const initial: UploadState[] = files.map((f) => ({
        filename: f.name,
        status:   "pending",
      }));
      setUploads(initial);

      for (let i = 0; i < files.length; i++) {
        const file     = files[i];
        const doc_type = file.name.startsWith("soal_") ? "soal" : "jawaban";

        setUploads((prev) =>
          prev.map((u, idx) => (idx === i ? { ...u, status: "uploading" } : u)),
        );

        try {
          await uploadDocument(client, file, doc_type, jenis_ujian);
          setUploads((prev) =>
            prev.map((u, idx) => (idx === i ? { ...u, status: "done" } : u)),
          );
        } catch (e) {
          const msg = e instanceof Error ? e.message : "Upload gagal.";
          setUploads((prev) =>
            prev.map((u, idx) =>
              idx === i ? { ...u, status: "error", error: msg } : u,
            ),
          );
        }
      }
      await fetchDocuments();
    },
    [client, fetchDocuments],
  );

  const remove = useCallback(
    async (file_id: string) => {
      setDeleting((prev) => new Set(prev).add(file_id));
      try {
        await deleteDocument(client, file_id);
        setDocuments((prev) => prev.filter((d) => d.file_id !== file_id));
        setTotal((prev) => Math.max(0, prev - 1));
      } catch (e) {
        setError(e instanceof Error ? e.message : "Hapus gagal.");
      } finally {
        setDeleting((prev) => {
          const next = new Set(prev);
          next.delete(file_id);
          return next;
        });
      }
    },
    [client],
  );

  const clearUploadState = useCallback(() => setUploads([]), []);

  return {
    documents, total, loading, error,
    uploads, deleting,
    fetchDocuments, upload, remove, clearUploadState,
  };
}
