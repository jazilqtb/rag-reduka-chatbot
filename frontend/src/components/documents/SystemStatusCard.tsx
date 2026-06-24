import { useCallback, useEffect, useMemo, useState } from "react";
import { ApiClient }       from "@/api/client";
import { getHealthDetail } from "@/api/documents";
import { cn }              from "@/lib/utils";
import type { AppSettings, HealthDetailResponse } from "@/types/api";

interface Props {
  settings: AppSettings;
}

const STATUS_COLOR: Record<string, string> = {
  ok:       "text-ai",
  error:    "text-error",
  degraded: "text-marker2",
  down:     "text-error",
};

export function SystemStatusCard({ settings }: Props) {
  const [health,  setHealth]  = useState<HealthDetailResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const client = useMemo(
    () => new ApiClient({ baseUrl: settings.baseUrl, apiKey: settings.apiKey }),
    [settings.baseUrl, settings.apiKey],
  );

  const fetchHealth = useCallback(async () => {
    setLoading(true);
    try {
      const res = await getHealthDetail(client);
      setHealth(res);
    } catch {
      setHealth(null);
    } finally {
      setLoading(false);
    }
  }, [client]);

  useEffect(() => { fetchHealth(); }, [fetchHealth]);

  if (loading && !health) {
    return (
      <div className="rounded-xl border border-line p-4 text-sm text-muted animate-pulse">
        Memeriksa status sistem…
      </div>
    );
  }

  if (!health) {
    return (
      <div className="rounded-xl border border-line p-4 text-sm text-error">
        Tidak bisa terhubung ke backend. Pastikan backend jalan dan API key benar.
      </div>
    );
  }

  const components = Object.entries(health.components);

  return (
    <div className="rounded-xl border border-line bg-canvas p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-ink">Status Sistem</h3>
        <div className="flex items-center gap-2">
          <span
            className={cn("text-xs font-medium font-mono", STATUS_COLOR[health.status] ?? "text-muted")}
          >
            {health.status.toUpperCase()}
          </span>
          <button
            type="button"
            onClick={fetchHealth}
            className="text-xs text-muted hover:text-ink focus-ring rounded transition-colors"
            title="Refresh status"
          >
            ↻
          </button>
        </div>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
        {components.map(([name, comp]) => (
          comp && (
            <div key={name} className="flex items-start gap-2 p-2 rounded-lg bg-user/60">
              <span
                className={cn(
                  "mt-0.5 h-1.5 w-1.5 rounded-full shrink-0",
                  comp.status === "ok" ? "bg-ai" : "bg-error",
                )}
              />
              <div>
                <div className="text-xs font-medium text-ink capitalize">{name}</div>
                <div className="text-[10px] text-muted leading-relaxed line-clamp-2">{comp.detail}</div>
              </div>
            </div>
          )
        ))}
      </div>
    </div>
  );
}
