/**
 * Placeholder skeleton App component.
 *
 * NOTE: File ini akan di-replace di Batch 6.3 dengan komponen lengkap
 * (Header + ChatWindow + InputBox + SettingsPanel + hooks).
 *
 * Untuk Batch 6.2 cuma minimal "hello world" supaya skeleton bisa di-build
 * dan diverifikasi: Tailwind classes bekerja, Vite bundle berhasil, fonts
 * loaded dari Google Fonts.
 */
export default function App() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-canvas px-6">
      <div className="max-w-md text-center">
        <span className="inline-block h-12 w-12 rounded-full bg-marker/60 mb-6" />
        <h1 className="text-3xl font-bold tracking-tight">
          UTBK <span className="highlight-underline">Tutor</span>
        </h1>
        <p className="mt-4 text-muted text-sm leading-relaxed">
          Frontend skeleton ready. Komponen chat lengkap akan ditambahkan
          di Batch 6.3.
        </p>
        <div className="mt-8 inline-flex items-center gap-2 text-xs font-mono text-muted">
          <span className="inline-block h-1.5 w-1.5 rounded-full bg-ai" />
          v0.1 · skeleton
        </div>
      </div>
    </div>
  );
}
