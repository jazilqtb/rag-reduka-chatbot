import type { AppPage } from "@/types/api";

interface Props {
  onNavigate: (page: AppPage) => void;
}

export function GettingStartedPage({ onNavigate }: Props) {
  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-3xl px-4 py-8 space-y-10">

        {/* Hero */}
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-ink">
            Cara Menggunakan <span className="highlight-underline">UTBK Tutor AI</span>
          </h1>
          <p className="mt-2 text-muted text-sm leading-relaxed">
            Upload PDF soal dan kunci jawaban UTBK kamu, ingest ke knowledge base,
            lalu tanya langsung di chat. Panduan ini menjelaskan setiap langkahnya.
          </p>
        </div>

        {/* Langkah 1: Format file */}
        <section>
          <div className="flex items-center gap-3 mb-4">
            <span className="flex h-7 w-7 items-center justify-center rounded-full bg-marker text-ink text-sm font-bold">1</span>
            <h2 className="text-base font-semibold text-ink">Format Nama File PDF</h2>
          </div>
          <div className="rounded-xl border border-line p-5 space-y-4">
            <p className="text-sm text-muted">
              Sistem mengenali tipe dokumen dari nama file. Ikuti konvensi berikut:
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="p-3 rounded-lg border border-line space-y-1.5">
                <div className="text-xs font-medium text-ink">File Soal</div>
                <code className="text-sm font-mono text-ai block">soal_&lt;topik&gt;.pdf</code>
                <div className="text-xs text-muted space-y-0.5">
                  <div>✓ soal_literasi_indonesia.pdf</div>
                  <div>✓ soal_penalaran_umum.pdf</div>
                  <div>✓ soal_tryout1.pdf</div>
                </div>
              </div>
              <div className="p-3 rounded-lg border border-line space-y-1.5">
                <div className="text-xs font-medium text-ink">File Jawaban / Kunci</div>
                <code className="text-sm font-mono text-ai block">jawaban_&lt;topik&gt;.pdf</code>
                <div className="text-xs text-muted space-y-0.5">
                  <div>✓ jawaban_literasi_indonesia.pdf</div>
                  <div>✓ jawaban_penalaran_umum.pdf</div>
                  <div>✓ jawaban_tryout1.pdf</div>
                </div>
              </div>
            </div>

            <div className="bg-error/5 border border-error/20 rounded-lg p-3 text-xs text-error">
              <span className="font-medium">Hindari:</span> spasi di nama file, karakter khusus
              selain underscore, atau nama tanpa prefix <code>soal_</code>/<code>jawaban_</code>.
            </div>

            <div className="text-xs text-muted">
              <span className="font-medium text-ink">Catatan:</span> &lt;topik&gt; = alphanumeric +
              underscore, maks 50 karakter. Contoh: <code className="font-mono">penalaran_umum</code>,{" "}
              <code className="font-mono">tryout2026</code>,{" "}
              <code className="font-mono">mat_dasar_1</code>.
            </div>
          </div>
        </section>

        {/* Langkah 2: Struktur isi PDF soal */}
        <section>
          <div className="flex items-center gap-3 mb-4">
            <span className="flex h-7 w-7 items-center justify-center rounded-full bg-marker text-ink text-sm font-bold">2</span>
            <h2 className="text-base font-semibold text-ink">Struktur Isi PDF Soal</h2>
          </div>
          <div className="rounded-xl border border-line p-5 space-y-4">
            <p className="text-sm text-muted">
              Sistem mengekstrak nomor soal dan mata pelajaran dari teks PDF.
              Pastikan format penomoran soal jelas dan konsisten.
            </p>

            <div>
              <div className="text-xs font-medium text-ink mb-2">Format yang direkomendasikan:</div>
              <pre className="bg-user rounded-lg p-4 text-xs font-mono text-ink overflow-x-auto leading-relaxed">{`TRYOUT SNBT 2026
Mata Pelajaran: Penalaran Umum
Jumlah Soal: 20

1. Jika semua A adalah B, dan semua B adalah C,
   maka kesimpulan yang tepat adalah…
   A. Semua A adalah C
   B. Semua C adalah A
   C. Tidak ada A yang C
   D. Beberapa C bukan A
   E. Beberapa A bukan C

2. Bacalah teks berikut dengan seksama.
   [teks...]

   Pernyataan yang sesuai dengan isi teks adalah…
   A. ...
   B. ...`}</pre>
            </div>

            <div className="space-y-2 text-xs text-muted">
              <div className="flex gap-2">
                <span className="text-ai font-medium shrink-0">✓</span>
                Nomor soal ditulis jelas: <code className="font-mono">1.</code>,{" "}
                <code className="font-mono">2.</code>, atau <code className="font-mono">No. 1</code>
              </div>
              <div className="flex gap-2">
                <span className="text-ai font-medium shrink-0">✓</span>
                Cantumkan header mata pelajaran agar source citation akurat
              </div>
              <div className="flex gap-2">
                <span className="text-ai font-medium shrink-0">✓</span>
                Gambar/grafik akan di-caption otomatis oleh AI (multimodal parsing)
              </div>
              <div className="flex gap-2">
                <span className="text-ai font-medium shrink-0">✓</span>
                PDF text-based lebih baik dari PDF scan (OCR tidak selalu akurat)
              </div>
            </div>
          </div>
        </section>

        {/* Langkah 3: Struktur isi PDF jawaban */}
        <section>
          <div className="flex items-center gap-3 mb-4">
            <span className="flex h-7 w-7 items-center justify-center rounded-full bg-marker text-ink text-sm font-bold">3</span>
            <h2 className="text-base font-semibold text-ink">Struktur Isi PDF Jawaban / Kunci</h2>
          </div>
          <div className="rounded-xl border border-line p-5 space-y-4">
            <p className="text-sm text-muted">
              PDF jawaban berisi kunci jawaban beserta pembahasan. Semakin detail
              pembahasan, semakin kaya konteks yang dimiliki chatbot.
            </p>

            <div>
              <div className="text-xs font-medium text-ink mb-2">Format yang direkomendasikan:</div>
              <pre className="bg-user rounded-lg p-4 text-xs font-mono text-ink overflow-x-auto leading-relaxed">{`KUNCI JAWABAN DAN PEMBAHASAN
Tryout SNBT 2026 — Penalaran Umum

1. Jawaban: A
   Pembahasan: Berdasarkan silogisme hipotetis, jika semua A adalah B
   dan semua B adalah C, maka semua A pastilah C. Pilihan B, C, D, E
   tidak dapat disimpulkan dari premis yang diberikan.

2. Jawaban: C
   Pembahasan: Teks menyatakan bahwa... [penjelasan detail]`}</pre>
            </div>

            <div className="space-y-2 text-xs text-muted">
              <div className="flex gap-2">
                <span className="text-ai font-medium shrink-0">✓</span>
                Nomor soal HARUS sama persis dengan di PDF soal
              </div>
              <div className="flex gap-2">
                <span className="text-ai font-medium shrink-0">✓</span>
                Sertakan pembahasan — kualitas jawaban chatbot bergantung pada ini
              </div>
              <div className="flex gap-2">
                <span className="text-ai font-medium shrink-0">✓</span>
                Boleh tanpa pembahasan, tapi chatbot hanya bisa jawab "jawaban X" tanpa penjelasan
              </div>
            </div>
          </div>
        </section>

        {/* Langkah 4: Jenis Ujian */}
        <section>
          <div className="flex items-center gap-3 mb-4">
            <span className="flex h-7 w-7 items-center justify-center rounded-full bg-marker text-ink text-sm font-bold">4</span>
            <h2 className="text-base font-semibold text-ink">Label "Jenis Ujian"</h2>
          </div>
          <div className="rounded-xl border border-line p-5 space-y-3">
            <p className="text-sm text-muted">
              Saat upload, Anda akan diminta mengisi <strong className="text-ink">Jenis Ujian</strong>.
              Ini adalah label yang muncul di source citation saat chat.
            </p>
            <div className="bg-user rounded-lg p-3 text-xs space-y-1.5">
              <div className="text-ink font-medium">Contoh label:</div>
              <div className="text-muted font-mono">Tryout 1, Tryout 2</div>
              <div className="text-muted font-mono">Simulasi SNBT 2026</div>
              <div className="text-muted font-mono">Prediksi UTBK Batch A</div>
            </div>
            <div className="text-xs text-muted">
              <span className="font-medium text-ink">Penting:</span> label soal dan jawaban
              harus <strong>SAMA persis</strong> supaya sistem bisa mencocokkan keduanya.
              Contoh: soal + jawaban keduanya berlabel <code className="font-mono">"Tryout 1"</code>.
            </div>
          </div>
        </section>

        {/* Langkah 5: Upload & Ingest */}
        <section>
          <div className="flex items-center gap-3 mb-4">
            <span className="flex h-7 w-7 items-center justify-center rounded-full bg-marker text-ink text-sm font-bold">5</span>
            <h2 className="text-base font-semibold text-ink">Upload & Ingest</h2>
          </div>
          <div className="rounded-xl border border-line p-5 space-y-3">
            <ol className="space-y-3 text-sm text-muted">
              <li className="flex gap-3">
                <span className="text-ink font-medium shrink-0">a.</span>
                Buka halaman{" "}
                <button
                  type="button"
                  onClick={() => onNavigate("documents")}
                  className="text-ai underline underline-offset-2 focus-ring rounded"
                >
                  Dokumen
                </button>.
              </li>
              <li className="flex gap-3">
                <span className="text-ink font-medium shrink-0">b.</span>
                Drag & drop atau klik area upload. Pilih pasangan file{" "}
                <code className="font-mono text-xs">soal_*.pdf</code> dan{" "}
                <code className="font-mono text-xs">jawaban_*.pdf</code>.
              </li>
              <li className="flex gap-3">
                <span className="text-ink font-medium shrink-0">c.</span>
                Isi "Jenis Ujian" dengan label yang sama untuk kedua file. Klik Upload.
              </li>
              <li className="flex gap-3">
                <span className="text-ink font-medium shrink-0">d.</span>
                Setelah upload selesai, klik tombol{" "}
                <strong className="text-ink">Ingest Dokumen Pending</strong>.
              </li>
              <li className="flex gap-3">
                <span className="text-ink font-medium shrink-0">e.</span>
                Tunggu progress bar selesai. Proses bervariasi tergantung jumlah halaman
                (biasanya 1–5 menit per file).
              </li>
              <li className="flex gap-3">
                <span className="text-ink font-medium shrink-0">f.</span>
                Status dokumen di tabel berubah dari{" "}
                <span className="text-xs px-1.5 py-0.5 bg-marker/40 rounded font-medium">Belum diingest</span>{" "}
                menjadi{" "}
                <span className="text-xs px-1.5 py-0.5 bg-ai/15 text-ai rounded font-medium">Ter-ingest</span>.
              </li>
            </ol>
          </div>
        </section>

        {/* Langkah 6: Chat */}
        <section>
          <div className="flex items-center gap-3 mb-4">
            <span className="flex h-7 w-7 items-center justify-center rounded-full bg-marker text-ink text-sm font-bold">6</span>
            <h2 className="text-base font-semibold text-ink">Mulai Chat</h2>
          </div>
          <div className="rounded-xl border border-line p-5 space-y-3">
            <p className="text-sm text-muted">
              Setelah ingestion selesai, buka tab Chat dan mulai bertanya.
            </p>
            <div className="space-y-2">
              <div className="text-xs font-medium text-ink">Contoh pertanyaan yang bekerja baik:</div>
              {[
                "Jelaskan soal nomor 3 Penalaran Umum",
                "Kenapa jawaban soal 5 Literasi Inggris itu B?",
                "Bahas soal nomor 12 dong",
                "Soal nomor 7 tentang apa?",
              ].map((q) => (
                <div key={q} className="text-xs text-muted font-mono bg-user rounded px-3 py-1.5">
                  → {q}
                </div>
              ))}
            </div>
            <p className="text-xs text-muted">
              Sistem menggunakan 4-layer retrieval: regex → similarity search →
              Redis cache → LLM extractor. Query dengan nomor soal eksplisit mendapat
              hasil terbaik dan tercepat.
            </p>
            <button
              type="button"
              onClick={() => onNavigate("chat")}
              className="mt-2 w-full py-2.5 text-sm font-medium rounded-bubble bg-ink text-canvas hover:bg-ink/85 focus-ring transition-colors"
            >
              Mulai Chat →
            </button>
          </div>
        </section>

        {/* Token count info */}
        <section>
          <div className="flex items-center gap-3 mb-4">
            <span className="flex h-7 w-7 items-center justify-center rounded-full bg-user border border-line text-ink text-sm font-medium">?</span>
            <h2 className="text-base font-semibold text-ink">Tentang "Estimasi Token"</h2>
          </div>
          <div className="rounded-xl border border-line p-5 text-sm text-muted space-y-2">
            <p>
              Setiap response AI menampilkan estimasi jumlah token di footer bubble.
              Ini dihitung di frontend: <code className="font-mono text-xs">~chars ÷ 4</code> (konvensi umum).
            </p>
            <p>
              Token count aktual dari Gemini API mungkin sedikit berbeda. Estimasi ini
              berguna untuk gambaran kasar biaya query.
            </p>
          </div>
        </section>

        {/* FAQ */}
        <section>
          <h2 className="text-base font-semibold text-ink mb-4">FAQ</h2>
          <div className="space-y-3">
            {[
              {
                q: "Apakah bisa upload lebih dari satu tryout?",
                a: "Ya. Upload soal + jawaban per tryout dengan label berbeda (misal: Tryout 1, Tryout 2). Chatbot bisa menjawab dari semua tryout sekaligus.",
              },
              {
                q: "Apa yang terjadi kalau saya re-upload file yang sama?",
                a: "Sistem melakukan incremental ingestion: chunk lama dari file itu dihapus, lalu chunk baru dari file yang sama di-insert. File lain tidak terpengaruh.",
              },
              {
                q: "Seberapa akurat chatbot menjawab?",
                a: "Akurasi bergantung pada kualitas PDF dan kelengkapan pembahasan di file jawaban. PDF dengan pembahasan detail menghasilkan jawaban yang lebih kaya konteks.",
              },
              {
                q: "Berapa lama proses ingestion?",
                a: "Tergantung jumlah halaman dan gambar. Per file soal ~1-3 menit (gambar = caption LLM per gambar). File teks saja lebih cepat.",
              },
              {
                q: "Apakah data saya aman?",
                a: "Data tersimpan lokal di server yang kamu jalankan. Tidak ada data yang dikirim ke pihak ketiga selain Google Gemini API untuk inference (teks soal/jawaban).",
              },
            ].map(({ q, a }) => (
              <div key={q} className="rounded-xl border border-line p-4">
                <div className="text-sm font-medium text-ink mb-1">{q}</div>
                <div className="text-sm text-muted">{a}</div>
              </div>
            ))}
          </div>
        </section>

      </div>
    </div>
  );
}
