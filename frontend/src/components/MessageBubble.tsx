import { cn, formatLatency } from "@/lib/utils";
import type { Message } from "@/types/api";
import { SourceCitation } from "./SourceCitation";

interface MessageBubbleProps {
  message: Message;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";

  return (
    <div
      className={cn(
        "flex w-full",
        isUser ? "justify-end" : "justify-start",
      )}
    >
      <div
        className={cn(
          "max-w-[85%] sm:max-w-[80%] px-4 py-3 rounded-bubble shadow-bubble",
          isUser
            ? "bg-user text-ink rounded-br-sm"
            : "bg-canvas text-ink border border-line rounded-bl-sm",
        )}
      >
        {/* Pending state: animasi 3 titik */}
        {message.pending ? (
          <div className="loading-dots flex items-center gap-1 h-5">
            <span /> <span /> <span />
          </div>
        ) : message.error ? (
          <div className="text-error text-sm">
            <span className="font-medium">Gagal:</span> {message.error}
          </div>
        ) : (
          <>
            <div className="whitespace-pre-wrap leading-relaxed text-[15px]">
              {message.content}
            </div>

            {/* Sumber referensi — hanya untuk pesan AI */}
            {!isUser && message.sources && message.sources.length > 0 && (
              <SourceCitation sources={message.sources} />
            )}

            {/* Latency footer — hanya untuk pesan AI */}
            {!isUser && message.latency !== undefined && (
              <div className="mt-2 text-[10px] text-muted font-mono tracking-wide">
                {formatLatency(message.latency)}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}