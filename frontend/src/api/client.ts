// ============================================================================
// HTTP client base — fetch wrapper dengan:
//   - X-API-Key header
//   - JSON serialize / deserialize
//   - Typed error dari backend ErrorResponse
// ============================================================================

import type { ErrorResponse } from "@/types/api";

export class ApiError extends Error {
  readonly status: number;
  readonly code:   string;
  readonly detail: unknown;

  constructor(status: number, body: ErrorResponse) {
    super(body.message || `HTTP ${status}`);
    this.name   = "ApiError";
    this.status = status;
    this.code   = body.error || "unknown_error";
    this.detail = body.detail;
  }
}


export interface ClientConfig {
  baseUrl: string;
  apiKey:  string;
}

export class ApiClient {
  constructor(private cfg: ClientConfig) {}

  private async request<T>(
    method: string,
    path:   string,
    body?:  unknown,
  ): Promise<T> {
    const url = `${this.cfg.baseUrl.replace(/\/$/, "")}${path}`;

    const headers: Record<string, string> = {
      "X-API-Key": this.cfg.apiKey,
    };
    if (body !== undefined) headers["Content-Type"] = "application/json";

    let response: Response;
    try {
      response = await fetch(url, {
        method,
        headers,
        body: body !== undefined ? JSON.stringify(body) : undefined,
      });
    } catch (e) {
      // Network error / CORS / DNS
      throw new ApiError(0, {
        error:   "network_error",
        message: e instanceof Error ? e.message : "Tidak bisa terhubung ke server.",
      });
    }

    // Parse body even on error — backend selalu return JSON
    let payload: unknown;
    try {
      payload = await response.json();
    } catch {
      payload = { error: "invalid_response", message: response.statusText };
    }

    if (!response.ok) {
      const err = (payload as { detail?: ErrorResponse }).detail ?? payload as ErrorResponse;
      throw new ApiError(response.status, err);
    }

    return payload as T;
  }

  get<T>(path:  string): Promise<T> {
    return this.request<T>("GET", path);
  }
  post<T>(path: string, body: unknown): Promise<T> {
    return this.request<T>("POST", path, body);
  }
  delete<T>(path: string): Promise<T> {
    return this.request<T>("DELETE", path);
  }
}