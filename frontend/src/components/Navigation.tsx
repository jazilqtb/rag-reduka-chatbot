import { cn } from "@/lib/utils";
import type { AppPage } from "@/types/api";

interface NavigationProps {
  currentPage:  AppPage;
  onNavigate:   (page: AppPage) => void;
}

const TABS: { id: AppPage; label: string }[] = [
  { id: "chat",            label: "Chat" },
  { id: "documents",       label: "Dokumen" },
  { id: "getting-started", label: "Panduan" },
];

export function Navigation({ currentPage, onNavigate }: NavigationProps) {
  return (
    <nav className="border-b border-line bg-canvas">
      <div className="mx-auto max-w-3xl px-6">
        <div className="flex gap-0">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => onNavigate(tab.id)}
              className={cn(
                "px-4 py-2.5 text-sm font-medium transition-colors border-b-2 focus-ring",
                currentPage === tab.id
                  ? "border-ai text-ai"
                  : "border-transparent text-muted hover:text-ink hover:border-line",
              )}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>
    </nav>
  );
}
