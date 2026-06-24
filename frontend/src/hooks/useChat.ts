import { useCallback, useMemo, useState } from "react";
import { ApiClient } from "@/api/client";
import { clearSession, sendChat } from "@/api/chat";
import { generateSessionId } from "@/lib/utils";
import type { AppSettings, Message } from "@/types/api";

export function useChat(settings: AppSettings) {
  const [messages,  setMessages]  = useState<Message[]>([]);
  const [sessionId, setSessionId] = useState<string>(() => generateSessionId());
  const [isSending, setIsSending] = useState(false);

  const client = useMemo(
    () => new ApiClient({ baseUrl: settings.baseUrl, apiKey: settings.apiKey }),
    [settings.baseUrl, settings.apiKey],
  );

  const send = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || isSending) return;

      // Push user message + pending assistant placeholder
      const userMsg: Message = {
        id:      `u_${Date.now()}`,
        role:    "user",
        content: trimmed,
      };
      const pendingMsg: Message = {
        id:      `a_${Date.now()}`,
        role:    "assistant",
        content: "",
        pending: true,
      };
      setMessages((m) => [...m, userMsg, pendingMsg]);
      setIsSending(true);

      try {
        const res = await sendChat(client, {
          user_id:    settings.userId,
          session_id: sessionId,
          query:      trimmed,
        });

        // Replace pending dengan response asli
        setMessages((m) =>
          m.map((msg) =>
            msg.id === pendingMsg.id
              ? {
                  ...msg,
                  content: res.answer,
                  sources: res.sources,
                  latency: res.meta?.latency_ms,
                  pending: false,
                }
              : msg,
          ),
        );
        // Sync session_id dari server (kalau auto-generated)
        if (res.session_id && res.session_id !== sessionId) {
          setSessionId(res.session_id);
        }
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : "Terjadi kesalahan.";
        setMessages((m) =>
          m.map((msg) =>
            msg.id === pendingMsg.id
              ? { ...msg, content: "", error: errorMsg, pending: false }
              : msg,
          ),
        );
      } finally {
        setIsSending(false);
      }
    },
    [client, isSending, sessionId, settings.userId],
  );

  const reset = useCallback(async () => {
    // Best-effort: clear server-side session, then local state
    try {
      await clearSession(client, settings.userId, sessionId);
    } catch {
      // Silently ignore — kalau auth/network gagal, tetap reset lokal
    }
    setMessages([]);
    setSessionId(generateSessionId());
  }, [client, settings.userId, sessionId]);

  return { messages, sessionId, isSending, send, reset };
}