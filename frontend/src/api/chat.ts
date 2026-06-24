import type { ApiClient } from "./client";
import type { ChatRequest, ChatResponse } from "@/types/api";

export async function sendChat(
  client: ApiClient,
  req:    ChatRequest,
): Promise<ChatResponse> {
  return client.post<ChatResponse>("/v1/chat", req);
}

export async function clearSession(
  client:     ApiClient,
  userId:     string,
  sessionId?: string,
): Promise<void> {
  const path = sessionId
    ? `/v1/session/${userId}?session_id=${encodeURIComponent(sessionId)}`
    : `/v1/session/${userId}`;
  await client.delete<unknown>(path);
}