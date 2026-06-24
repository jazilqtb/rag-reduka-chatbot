import { cn } from "@/lib/utils";
import type { DocumentItem } from "@/types/api";

interface Props {
  document:   DocumentItem | null;
  isDeleting: boolean;
  onConfirm:  () => void;
  onCancel:   () => void;
}

export function DeleteConfirmModal({ document: doc, isDeleting, onConfirm, onCancel }: Props) {
  if (!doc) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-ink/30 backdrop-blur-sm px-4"
      onClick={onCancel}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="bg-canvas w-full max-w-sm rounded-2xl border border-line shadow-soft p-6"
      >
        <h2 className="text-base font-semibold text-ink mb-1">Hapus Dokumen?</h2>
        <p className="text-sm text-muted mb-2">
          Dokumen berikut akan dihapus permanen dari storage dan vector store:
        </p>
        <div className="bg-user rounded-lg px-3 py-2 mb-4 font-mono text-xs text-ink">
          {doc.filename}
          {doc.chunk_count !== undefined && doc.chunk_count !== null && (
            <span className="ml-2 text-muted">({doc.chunk_count} chunk)</span>
          )}
        </div>
        <p className="text-xs text-error mb-4">
          Tindakan ini tidak bisa dibatalkan.
        </p>
        <div className="flex items-center justify-end gap-2">
          <button
            type="button"
            onClick={onCancel}
            disabled={isDeleting}
            className="px-4 py-2 text-sm rounded-md hover:bg-line/60 focus-ring transition-colors"
          >
            Batal
          </button>
          <button
            type="button"
            onClick={onConfirm}
            disabled={isDeleting}
            className={cn(
              "px-4 py-2 text-sm rounded-md font-medium focus-ring transition-colors",
              isDeleting
                ? "bg-error/40 text-canvas cursor-not-allowed"
                : "bg-error text-canvas hover:bg-error/85",
            )}
          >
            {isDeleting ? "Menghapus…" : "Ya, Hapus"}
          </button>
        </div>
      </div>
    </div>
  );
}
