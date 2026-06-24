import { useState } from "react";
import { Header }             from "@/components/Header";
import { Navigation }         from "@/components/Navigation";
import { ChatWindow }         from "@/components/ChatWindow";
import { InputBox }           from "@/components/InputBox";
import { SettingsPanel }      from "@/components/SettingsPanel";
import { DocumentsPage }      from "@/pages/DocumentsPage";
import { GettingStartedPage } from "@/pages/GettingStartedPage";
import { useChat }            from "@/hooks/useChat";
import { useSettings }        from "@/hooks/useSettings";
import type { AppPage }       from "@/types/api";

export default function App() {
  const { settings, updateSettings, isConfigured } = useSettings();
  const { messages, isSending, send, reset }       = useChat(settings);

  const [showSettings, setShowSettings] = useState(!isConfigured);
  const [currentPage,  setCurrentPage]  = useState<AppPage>("chat");

  const placeholder = isConfigured
    ? "Tanya soal UTBK apa aja…"
    : "Set API key dulu di Settings untuk mulai.";

  return (
    <div className="min-h-screen flex flex-col bg-canvas">
      {/* Header always visible */}
      <Header
        onSettingsClick={() => setShowSettings(true)}
        onResetClick={reset}
        isConfigured={isConfigured}
        hasMessages={messages.length > 0}
        currentPage={currentPage}
      />

      {/* Navigation tabs */}
      <Navigation currentPage={currentPage} onNavigate={setCurrentPage} />

      {/* Page content */}
      {currentPage === "chat" && (
        <>
          <ChatWindow messages={messages} isConfigured={isConfigured} />
          <InputBox
            onSend={send}
            disabled={!isConfigured || isSending}
            placeholder={placeholder}
          />
        </>
      )}

      {currentPage === "documents" && (
        <DocumentsPage settings={settings} />
      )}

      {currentPage === "getting-started" && (
        <GettingStartedPage onNavigate={setCurrentPage} />
      )}

      {/* Settings modal */}
      <SettingsPanel
        open={showSettings}
        initial={settings}
        onClose={() => setShowSettings(false)}
        onSave={updateSettings}
      />
    </div>
  );
}
