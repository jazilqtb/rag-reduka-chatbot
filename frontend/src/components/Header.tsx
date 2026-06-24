import { cn } from "@/lib/utils";
import type { AppPage } from "@/types/api";

interface HeaderProps {
  onSettingsClick: () => void;
  onResetClick:    () => void;
  isConfigured:    boolean;
  hasMessages:     boolean;
  currentPage:     AppPage;
}

export function Header({ onSettingsClick, onResetClick, isConfigured, hasMessages, currentPage }: HeaderProps) {
  return (
    <header className="border-b border-line bg-canvas/80 backdrop-blur-sm sticky top-0 z-10">
      <div className="mx-auto max-w-3xl px-6 py-4 flex items-center justify-between">
        <div className="flex items-baseline gap-2">
          <span className="font-bold text-ink text-lg tracking-tight">UTBK Tutor</span>
          <span className="text-xs text-muted font-mono">v0.1</span>
        </div>

        <div className="flex items-center gap-2">
          {currentPage === "chat" && hasMessages && (
            <button
              type="button"
              onClick={onResetClick}
              className="text-sm text-muted hover:text-ink transition-colors px-3 py-1.5 rounded-md focus-ring"
              aria-label="Mulai sesi baru"
            >
              Sesi baru
            </button>
          )}
          <button
            type="button"
            onClick={onSettingsClick}
            className={cn(
              "text-sm px-3 py-1.5 rounded-md focus-ring transition-colors",
              isConfigured
                ? "text-muted hover:text-ink"
                : "bg-marker text-ink font-medium hover:bg-marker2",
            )}
          >
            {isConfigured ? "Settings" : "Mulai · Set API Key"}
          </button>
        </div>
      </div>
    </header>
  );
}
