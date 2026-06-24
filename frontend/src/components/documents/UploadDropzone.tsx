import { useRef, useState } from "react";
import { cn } from "@/lib/utils";

interface UploadState {
  filename: string;
  status:   "pending" | "uploading" | "done" | "error";
  error?:   string;
}

interface Props {
  uploads:     UploadState[];
  isUploading: boolean;
  onUpload:    (files: File[], jenis_ujian: string) => void;
  onClear:     () => void;
}

function validateFilename(name: string): string | null {
  if (!/^(soal|jawaban)_[a-zA-Z0-9_]{1,50}\.pdf$/.test(name)) {
    return `Nama file harus format soal_*.pdf atau jawaban_*.pdf (contoh: soal_tryout1.pdf)`;
  }
  return null;
}

export function UploadDropzone({ uploads, isUploading, onUpload, onClear }: Props) {
  const [isDragging,  setIsDragging]  = useState(false);
  const [jenis_ujian, setJenisUjian]  = useState("");
  const [fileErrors,  setFileErrors]  = useState<Record<string, string>>({});
  const [selected,    setSelected]    = useState<File[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = (files: FileList | null) => {
    if (!files) return;
    const arr    = Array.from(files).filter((f) => f.type === "application/pdf");
    const errors: Record<string, string> = {};
    const valid:  File[] = [];

    arr.forEach((f) => {
      const err = validateFilename(f.name);
      if (err) errors[f.name] = err;
      else      valid.push(f);
    });

    setFileErrors(errors);
    setSelected(valid);
  };

  const handleSubmit = () => {
    if (!selected.length || !jenis_ujian.trim()) return;
    onUpload(selected, jenis_ujian.trim());
  };

  const hasErrors = Object.keys(fileErrors).length > 0;
  const canSubmit = selected.length > 0 && jenis_ujian.trim() && !isUploading && !hasErrors;

  return (
    <div className="rounded-xl border border-line p-5 space-y-4">
      <h3 className="text-sm font-medium text-ink">Upload Dokumen</h3>

      <div
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setIsDragging(false);
          handleFiles(e.dataTransfer.files);
        }}
        onClick={() => inputRef.current?.click()}
        className={cn(
          "border-2 border-dashed rounded-bubble p-8 text-center cursor-pointer transition-colors",
          isDragging
            ? "border-ai bg-ai/5"
            : "border-line hover:border-ink/30 hover:bg-user/30",
        )}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf"
          multiple
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
        <div className="text-sm text-muted">
          <span className="font-medium text-ink">Klik atau drag & drop</span> file PDF di sini
        </div>
        <div className="text-xs text-muted mt-1">
          Format: <code className="font-mono">soal_*.pdf</code> atau{" "}
          <code className="font-mono">jawaban_*.pdf</code>
        </div>
      </div>

      {Object.entries(fileErrors).map(([name, err]) => (
        <div key={name} className="text-xs text-error">
          <span className="font-medium">{name}:</span> {err}
        </div>
      ))}

      {selected.length > 0 && (
        <ul className="text-xs space-y-1">
          {selected.map((f) => {
            const state = uploads.find((u) => u.filename === f.name);
            return (
              <li key={f.name} className="flex items-center justify-between font-mono">
                <span className="text-ink">{f.name}</span>
                <span
                  className={cn(
                    "ml-2",
                    !state                         ? "text-muted"
                    : state.status === "done"      ? "text-ai"
                    : state.status === "error"     ? "text-error"
                    : state.status === "uploading" ? "text-marker2 animate-pulse"
                    : "text-muted",
                  )}
                >
                  {!state                          ? "siap"
                  : state.status === "done"        ? "✓ selesai"
                  : state.status === "error"       ? `✗ ${state.error ?? "error"}`
                  : state.status === "uploading"   ? "uploading…"
                  : "pending"}
                </span>
              </li>
            );
          })}
        </ul>
      )}

      <div>
        <label className="block text-xs font-medium text-ink mb-1.5">
          Jenis Ujian / Label <span className="text-error">*</span>
        </label>
        <input
          type="text"
          value={jenis_ujian}
          onChange={(e) => setJenisUjian(e.target.value)}
          placeholder="contoh: Tryout 1, Simulasi SNBT 2026"
          className="w-full px-3 py-2 text-sm rounded-md border border-line bg-canvas focus-ring"
        />
        <p className="text-[10px] text-muted mt-1">
          Label ini tampil di source citation saat chat. Harus sama antara soal + jawabannya.
        </p>
      </div>

      <div className="flex items-center justify-between">
        <button
          type="button"
          onClick={() => { setSelected([]); setFileErrors({}); onClear(); }}
          className="text-xs text-muted hover:text-ink focus-ring rounded transition-colors"
        >
          Reset
        </button>
        <button
          type="button"
          onClick={handleSubmit}
          disabled={!canSubmit}
          className={cn(
            "px-4 py-2 text-sm rounded-md font-medium focus-ring transition-colors",
            canSubmit
              ? "bg-ink text-canvas hover:bg-ink/85"
              : "bg-line text-muted cursor-not-allowed",
          )}
        >
          {isUploading ? "Mengupload…" : `Upload${selected.length > 0 ? ` (${selected.length} file)` : ""}`}
        </button>
      </div>
    </div>
  );
}
