import { useRef, useState, KeyboardEvent } from "react";
import { cn } from "@/lib/utils";

interface InputBoxProps {
  onSend:      (text: string) => void;
  disabled:    boolean;
  placeholder: string;
}

export function InputBox({ onSend, disabled, placeholder }: InputBoxProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const submit = () => {
    if (disabled) return;
    const trimmed = value.trim();
    if (trimmed.length < 2) return;
    onSend(trimmed);
    setValue("");
    // Reset height after send
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  };

  const handleKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  // Auto-resize textarea sesuai konten (max 8 rows)
  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setValue(e.target.value);
    const ta = e.target;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 200)}px`;
  };

  const canSend = !disabled && value.trim().length >= 2;

  return (
    <div className="border-t border-line bg-canvas">
      <div className="mx-auto max-w-3xl px-4 py-4">
        <div
          className={cn(
            "flex items-end gap-2 rounded-bubble border bg-canvas px-3 py-2 transition-colors",
            disabled ? "border-line opacity-60" : "border-line focus-within:border-ink/30",
          )}
        >
          <textarea
            ref={textareaRef}
            value={value}
            onChange={handleInput}
            onKeyDown={handleKey}
            placeholder={placeholder}
            disabled={disabled}
            rows={1}
            className={cn(
              "flex-1 resize-none bg-transparent text-[15px] leading-relaxed",
              "placeholder:text-muted outline-none",
              "max-h-[200px] py-1.5",
            )}
          />
          <button
            type="button"
            onClick={submit}
            disabled={!canSend}
            className={cn(
              "shrink-0 w-9 h-9 rounded-full flex items-center justify-center focus-ring transition-all",
              canSend
                ? "bg-ink text-canvas hover:bg-ink/85"
                : "bg-line text-muted cursor-not-allowed",
            )}
            aria-label="Kirim pesan"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M5 12h14M13 5l7 7-7 7" />
            </svg>
          </button>
        </div>
        <p className="text-[11px] text-muted mt-2 px-1">
          <kbd className="font-mono">Enter</kbd> kirim · <kbd className="font-mono">Shift+Enter</kbd> baris baru
        </p>
      </div>
    </div>
  );
}