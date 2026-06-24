import { useState, useEffect } from "react";
import { cn } from "@/lib/utils";
import type { AppSettings } from "@/types/api";

interface SettingsPanelProps {
  open:           boolean;
  initial:        AppSettings;
  onClose:        () => void;
  onSave:         (patch: Partial<AppSettings>) => void;
}

export function SettingsPanel({ open, initial, onClose, onSave }: SettingsPanelProps) {
  const [baseUrl, setBaseUrl] = useState(initial.baseUrl);
  const [apiKey,  setApiKey]  = useState(initial.apiKey);

  useEffect(() => {
    if (open) {
      setBaseUrl(initial.baseUrl);
      setApiKey(initial.apiKey);
    }
  }, [open, initial]);

  if (!open) return null;

  const handleSave = (e: React.FormEvent) => {
    e.preventDefault();
    onSave({ baseUrl: baseUrl.trim(), apiKey: apiKey.trim() });
    onClose();
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-ink/30 backdrop-blur-sm px-4"
      onClick={onClose}
    >
      <form
        onSubmit={handleSave}
        onClick={(e) => e.stopPropagation()}
        className="bg-canvas w-full max-w-md rounded-2xl shadow-soft border border-line p-6"
      >
        <h2 className="text-lg font-semibold tracking-tight mb-1">Settings</h2>
        <p className="text-sm text-muted mb-6">
          Konfigurasi koneksi ke backend RAG service.
        </p>

        <div className="space-y-4">
          <div>
            <label htmlFor="baseUrl" className="block text-sm font-medium mb-1.5">
              Base URL
            </label>
            <input
              id="baseUrl"
              type="url"
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              placeholder="http://localhost:8000"
              className={cn(
                "w-full px-3 py-2 text-sm font-mono rounded-md border border-line",
                "bg-canvas focus-ring",
              )}
            />
            <p className="text-xs text-muted mt-1.5">
              Endpoint backend FastAPI yang sudah deploy.
            </p>
          </div>

          <div>
            <label htmlFor="apiKey" className="block text-sm font-medium mb-1.5">
              API Key
            </label>
            <input
              id="apiKey"
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="••••••••••••••••"
              autoComplete="off"
              className={cn(
                "w-full px-3 py-2 text-sm font-mono rounded-md border border-line",
                "bg-canvas focus-ring",
              )}
            />
            <p className="text-xs text-muted mt-1.5">
              Nilai dari env <code className="text-ai">API_KEY</code> di backend.
              Disimpan di localStorage browser ini saja.
            </p>
          </div>
        </div>

        <div className="flex items-center justify-end gap-2 mt-6">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 text-sm rounded-md hover:bg-line/60 focus-ring transition-colors"
          >
            Batal
          </button>
          <button
            type="submit"
            className={cn(
              "px-4 py-2 text-sm rounded-md font-medium focus-ring transition-colors",
              "bg-ink text-canvas hover:bg-ink/85",
            )}
          >
            Simpan
          </button>
        </div>
      </form>
    </div>
  );
}