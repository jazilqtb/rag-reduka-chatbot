import { useEffect, useRef } from "react";
import type { Message } from "@/types/api";
import { MessageBubble } from "./MessageBubble";

interface ChatWindowProps {
  messages:     Message[];
  isConfigured: boolean;
}

const SAMPLE_QUESTIONS = [
  "Jelaskan soal nomor 3 Penalaran Umum dong",
  "Kak, soal nomor 5 di Tryout 2 itu jawabannya kenapa B?",
  "Bahas semua soal Pengetahuan Kuantitatif yuk",
];

export function ChatWindow({ messages, isConfigured }: ChatWindowProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll ke bawah saat pesan baru masuk
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, messages[messages.length - 1]?.content]);

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center px-6">
        <div className="max-w-md w-full text-center">
          <div className="mb-6">
            <span className="inline-block h-12 w-12 rounded-full bg-marker/60 mb-4" />
            <h1 className="text-2xl font-bold tracking-tight">
              Tutor AI buat <span className="highlight-underline">tryout UTBK</span> kamu
            </h1>
            <p className="mt-3 text-muted text-sm leading-relaxed">
              Tanya soal nomor berapa aja. Aku jelasin pembahasannya berdasarkan kunci jawaban
              yang udah diingest ke knowledge base.
            </p>
          </div>

          {!isConfigured ? (
            <div className="rounded-bubble border border-marker2 bg-marker/30 p-4 text-sm">
              <span className="font-medium">Belum siap.</span> Klik tombol kuning di pojok kanan
              atas untuk set API key dan base URL backend.
            </div>
          ) : (
            <div className="space-y-2">
              <p className="text-xs uppercase tracking-wider text-muted font-medium mb-3">
                Coba mulai dari:
              </p>
              {SAMPLE_QUESTIONS.map((q) => (
                <div
                  key={q}
                  className="text-sm text-left px-4 py-2.5 rounded-bubble border border-line text-muted bg-canvas"
                >
                  <span className="text-muted/70">→</span> {q}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-3xl px-4 py-6 space-y-4">
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}