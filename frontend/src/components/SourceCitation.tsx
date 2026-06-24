import type { SourceItem } from "@/types/api";

interface SourceCitationProps {
  sources: SourceItem[];
}

/**
 * Signature element: source citations are styled like a tutor's margin notes
 * with a yellow highlighter accent under the subject + question number.
 */
export function SourceCitation({ sources }: SourceCitationProps) {
  if (sources.length === 0) return null;

  return (
    <div className="mt-3 pt-3 border-t border-line/70">
      <div className="flex items-center gap-1.5 mb-2 text-xs text-muted uppercase tracking-wider font-medium">
        <span className="inline-block h-1 w-1 rounded-full bg-marker2" />
        Sumber referensi
      </div>
      <ul className="space-y-1.5">
        {sources.map((s, i) => (
          <li key={`${s.source}-${i}`} className="flex items-baseline gap-2 text-xs font-mono">
            <span className="text-muted">
              {i + 1}.
            </span>
            <span className="text-ink">
              <span className="highlight-underline">{s.subject || "Umum"}</span>
              {s.id_soal && <> — soal no. <strong>{s.id_soal}</strong></>}
            </span>
            <span className="text-muted ml-auto truncate max-w-[180px]" title={s.source}>
              {s.source}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}