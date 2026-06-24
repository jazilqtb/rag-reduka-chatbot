// ============================================================================
// Small utility helpers — keep this file lean.
// ============================================================================

/** Concatenate classNames, dropping falsy values. */
export function cn(...classes: (string | undefined | false | null)[]): string {
  return classes.filter(Boolean).join(" ");
}

/** Generate a random alphanumeric string of given length. */
function randomString(len: number): string {
  const chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
  let out = "";
  for (let i = 0; i < len; i++) {
    out += chars[Math.floor(Math.random() * chars.length)];
  }
  return out;
}

/** Generate a fresh user_id matching backend regex: usr_<alphanum_underscore>{4,53}. */
export function generateUserId(): string {
  return `usr_demo_${randomString(8)}`;
}

/** Generate a fresh session_id matching backend regex: sess_<alphanum_underscore>{5,54}. */
export function generateSessionId(): string {
  return `sess_${randomString(12)}`;
}

/** Format ms into human-readable string (e.g., 1.2s or 850ms). */
export function formatLatency(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}