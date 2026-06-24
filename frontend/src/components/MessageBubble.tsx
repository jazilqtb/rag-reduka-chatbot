import { cn, formatLatency } from "@/lib/utils";
import type { Message } from "@/types/api";
import { SourceCitation } from "./SourceCitation";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

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
            {isUser ? (
              <div className="whitespace-pre-wrap leading-relaxed text-[15px]">
                {message.content}
              </div>
            ) : (
              <div className="leading-relaxed text-[15px] markdown-body">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                    strong: ({ children }) => <strong className="font-semibold text-ink">{children}</strong>,
                    em: ({ children }) => <em className="italic">{children}</em>,
                    h1: ({ children }) => <h1 className="text-xl font-bold mt-4 mb-2 text-ink">{children}</h1>,
                    h2: ({ children }) => <h2 className="text-lg font-bold mt-3 mb-1.5 text-ink">{children}</h2>,
                    h3: ({ children }) => <h3 className="text-base font-semibold mt-2 mb-1 text-ink">{children}</h3>,
                    ul: ({ children }) => <ul className="list-disc pl-5 mb-2 space-y-0.5">{children}</ul>,
                    ol: ({ children }) => <ol className="list-decimal pl-5 mb-2 space-y-0.5">{children}</ol>,
                    li: ({ children }) => <li className="leading-relaxed">{children}</li>,
                    pre: ({ children }) => <>{children}</>,
                    code: ({ className, children }) => {
                      const isBlock = typeof className === "string" && className.startsWith("language-");
                      return isBlock ? (
                        <code className="block bg-line/40 font-mono text-[13px] p-3 rounded-lg my-2 overflow-x-auto whitespace-pre">
                          {children}
                        </code>
                      ) : (
                        <code className="bg-line/60 text-ai font-mono text-[13px] px-1.5 py-0.5 rounded">
                          {children}
                        </code>
                      );
                    },
                    blockquote: ({ children }) => (
                      <blockquote className="border-l-4 border-ai/40 pl-3 italic text-muted my-2">
                        {children}
                      </blockquote>
                    ),
                    a: ({ href, children }) => (
                      <a href={href} target="_blank" rel="noreferrer" className="text-ai underline underline-offset-2 hover:text-ai/80">
                        {children}
                      </a>
                    ),
                    hr: () => <hr className="border-line my-3" />,
                    table: ({ children }) => (
                      <div className="overflow-x-auto my-2">
                        <table className="min-w-full text-sm border-collapse border border-line rounded">{children}</table>
                      </div>
                    ),
                    th: ({ children }) => <th className="bg-line/50 px-3 py-1.5 text-left font-semibold border border-line">{children}</th>,
                    td: ({ children }) => <td className="px-3 py-1.5 border border-line">{children}</td>,
                  }}
                >
                  {message.content}
                </ReactMarkdown>
              </div>
            )}

            {/* Sumber referensi — hanya untuk pesan AI */}
            {!isUser && message.sources && message.sources.length > 0 && (
              <SourceCitation sources={message.sources} />
            )}

            {/* Latency footer — hanya untuk pesan AI */}
            {!isUser && (message.latency !== undefined || message.content) && (
              <div className="mt-2 text-[10px] text-muted font-mono tracking-wide flex items-center gap-2">
                {message.latency !== undefined && (
                  <span>{formatLatency(message.latency)}</span>
                )}
                {message.content && (
                  <span className="opacity-60">
                    ~{Math.ceil(message.content.length / 4)} token
                  </span>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}