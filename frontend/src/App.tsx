import { useState } from "react";
import { Header } from "@/components/Header";
import { ChatWindow } from "@/components/ChatWindow";
import { InputBox } from "@/components/InputBox";
import { SettingsPanel } from "@/components/SettingsPanel";
import { useChat } from "@/hooks/useChat";
import { useSettings } from "@/hooks/useSettings";

export default function App() {
  const { settings, updateSettings, isConfigured } = useSettings();
  const { messages, isSending, send, reset } = useChat(settings);

  const [showSettings, setShowSettings] = useState(!isConfigured);

  const placeholder = isConfigured
    ? "Tanya soal UTBK apa aja…"
    : "Set API key dulu di Settings untuk mulai.";

  return (
    <div className="min-h-screen flex flex-col bg-canvas">
      <Header
        onSettingsClick={() => setShowSettings(true)}
        onResetClick={reset}
        isConfigured={isConfigured}
        hasMessages={messages.length > 0}
      />

      <ChatWindow messages={messages} isConfigured={isConfigured} />

      <InputBox
        onSend={send}
        disabled={!isConfigured || isSending}
        placeholder={placeholder}
      />

      <SettingsPanel
        open={showSettings}
        initial={settings}
        onClose={() => setShowSettings(false)}
        onSave={updateSettings}
      />
    </div>
  );
}