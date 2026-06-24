import { useState } from "react";
import { cn } from "@/lib/utils";
import { DeleteConfirmModal } from "./DeleteConfirmModal";
import type { DocumentItem } from "@/types/api";

interface Props {
  documents: DocumentItem[];
  loading:   boolean;
  error:     string | null;
  deleting:  Set<string>;
  onDelete:  (file_id: string) => void;
  onRefresh: () => void;
}

const STATUS_LABEL: Record<string, string> = {
  uploaded: "Belum diingest",
  ingested: "Ter-ingest",
  failed:   "Gagal",
};

const STATUS_CLASS: Record<string, string> = {
  uploaded: "bg-marker/40 text-ink",
  ingested: "bg-ai/15 text-ai",
  failed:   "bg-error/15 text-error",
};

function formatBytes(bytes: number): string {
  if (bytes < 1024)         return `${bytes} B`;
  if (bytes < 1024 * 1024)  return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString("id-ID", {
    day: "2-digit", month: "short", year: "numeric",
  });
}

export function DocumentTable({ documents, loading, error, deleting, onDelete, onRefresh }: Props) {
  const [toDelete, setToDelete] = useState<DocumentItem | null>(null);

  const handleConfirmDelete = () => {
    if (toDelete) {
      onDelete(toDelete.file_id);
      setToDelete(null);
    }
  };

  return (
    <>
      <div className="rounded-xl border border-line overflow-hidden">
        <div className="flex items-center justify-between px-5 py-3 border-b border-line">
          <h3 className="text-sm font-medium text-ink">
            Daftar Dokumen
            {documents.length > 0 && (
              <span className="ml-1.5 text-xs text-muted font-normal">({documents.length})</span>
            )}
          </h3>
          <button
            type="button"
            onClick={onRefresh}
            className="text-xs text-muted hover:text-ink focus-ring rounded transition-colors"
          >
            ↻ Refresh
          </button>
        </div>

        {loading && documents.length === 0 ? (
          <div className="px-5 py-8 text-sm text-muted text-center animate-pulse">
            Memuat dokumen…
          </div>
        ) : error ? (
          <div className="px-5 py-4 text-sm text-error">{error}</div>
        ) : documents.length === 0 ? (
          <div className="px-5 py-8 text-center">
            <p className="text-sm text-muted">Belum ada dokumen.</p>
            <p className="text-xs text-muted mt-1">Upload PDF pertama kamu di atas.</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-user/40 text-left">
                  <th className="px-4 py-2.5 text-xs font-medium text-muted">Filename</th>
                  <th className="px-4 py-2.5 text-xs font-medium text-muted">Tipe</th>
                  <th className="px-4 py-2.5 text-xs font-medium text-muted">Jenis Ujian</th>
                  <th className="px-4 py-2.5 text-xs font-medium text-muted">Status</th>
                  <th className="px-4 py-2.5 text-xs font-medium text-muted text-right">Chunks</th>
                  <th className="px-4 py-2.5 text-xs font-medium text-muted text-right">Ukuran</th>
                  <th className="px-4 py-2.5 text-xs font-medium text-muted">Upload</th>
                  <th className="px-4 py-2.5 text-xs font-medium text-muted"></th>
                </tr>
              </thead>
              <tbody className="divide-y divide-line">
                {documents.map((doc) => (
                  <tr key={doc.file_id} className="hover:bg-user/20 transition-colors">
                    <td className="px-4 py-3 font-mono text-xs text-ink max-w-[180px] truncate">
                      {doc.filename}
                    </td>
                    <td className="px-4 py-3">
                      <span
                        className={cn(
                          "text-[10px] font-mono px-1.5 py-0.5 rounded",
                          doc.doc_type === "soal" ? "bg-marker/40 text-ink" : "bg-ai/15 text-ai",
                        )}
                      >
                        {doc.doc_type}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-xs text-muted">{doc.jenis_ujian}</td>
                    <td className="px-4 py-3">
                      <span
                        className={cn(
                          "text-[10px] px-1.5 py-0.5 rounded font-medium",
                          STATUS_CLASS[doc.status] ?? "bg-line text-muted",
                        )}
                      >
                        {STATUS_LABEL[doc.status] ?? doc.status}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-xs text-muted text-right font-mono">
                      {doc.chunk_count ?? "–"}
                    </td>
                    <td className="px-4 py-3 text-xs text-muted text-right font-mono">
                      {formatBytes(doc.size_bytes)}
                    </td>
                    <td className="px-4 py-3 text-xs text-muted">
                      {formatDate(doc.uploaded_at)}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <button
                        type="button"
                        onClick={() => setToDelete(doc)}
                        disabled={deleting.has(doc.file_id)}
                        className={cn(
                          "text-xs text-muted hover:text-error focus-ring rounded transition-colors",
                          deleting.has(doc.file_id) && "opacity-40 cursor-not-allowed",
                        )}
                      >
                        {deleting.has(doc.file_id) ? "Menghapus…" : "Hapus"}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <DeleteConfirmModal
        document={toDelete}
        isDeleting={toDelete ? deleting.has(toDelete.file_id) : false}
        onConfirm={handleConfirmDelete}
        onCancel={() => setToDelete(null)}
      />
    </>
  );
}
