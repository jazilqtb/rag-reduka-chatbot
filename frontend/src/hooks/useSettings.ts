import { useEffect, useState } from "react";
import { generateUserId } from "@/lib/utils";
import type { AppSettings } from "@/types/api";

const STORAGE_KEY = "utbk_tutor_settings_v1";

const defaultSettings: AppSettings = {
  baseUrl: "http://localhost:8000",
  apiKey:  "",
  userId:  generateUserId(),
};

function loadFromStorage(): AppSettings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return defaultSettings;
    const parsed = JSON.parse(raw) as Partial<AppSettings>;
    return {
      baseUrl: parsed.baseUrl ?? defaultSettings.baseUrl,
      apiKey:  parsed.apiKey  ?? defaultSettings.apiKey,
      userId:  parsed.userId  ?? defaultSettings.userId,
    };
  } catch {
    return defaultSettings;
  }
}

export function useSettings() {
  const [settings, setSettingsState] = useState<AppSettings>(() => loadFromStorage());

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  }, [settings]);

  const updateSettings = (patch: Partial<AppSettings>) => {
    setSettingsState((prev) => ({ ...prev, ...patch }));
  };

  const isConfigured = settings.apiKey.trim().length > 0;

  return { settings, updateSettings, isConfigured };
}