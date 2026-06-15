import re
from typing import Optional


class RegexEntityExtractor:
    """
    Class untuk mengekstrak entitas id_soal dan subject dari query teks siswa.
    """

    # ─────────────────────────────────────────────────────────────────────────────
    # 1. SUBJECT PATTERNS
    # ─────────────────────────────────────────────────────────────────────────────
    SUBJECT_PATTERNS: dict[str, list[str]] = {
        "Penalaran Umum": [
            r"pen[a-z]{0,4}r[a-z]{0,4}\s+um[a-z]{0,3}",
            r"\bp\.?\s*u\.?\b",
        ],
        "Penalaran Matematika": [
            r"pen[a-z]{0,4}r[a-z]{0,4}\s+mat[a-z]{0,8}",
            r"\bp\.?\s*m\.?\b",
        ],
        "Literasi Bahasa Inggris": [
            r"lit[a-z]*\s+(?:b[a-z]+\s+)?ingg?[a-z]*",
            r"\bl\.?\s*b\.?\s*i(?:ng)?\.?\b",
            r"b(?:hs?|ahasa)\.?\s+ingg?[a-z]*",
        ],
        "Literasi Bahasa Indonesia": [
            r"lit[a-z]*\s+(?:b[a-z]+\s+)?ind[a-z]*",
            r"\bl\.?\s*b\.?\s*ind[a-z]*\b",
            r"b(?:hs?|ahasa)\.?\s+ind[a-z]*",
        ],
    }

    SUBJECT_PRIORITY = [
        "Literasi Bahasa Inggris",
        "Literasi Bahasa Indonesia",
        "Penalaran Matematika",
        "Penalaran Umum",
    ]

    # ─────────────────────────────────────────────────────────────────────────────
    # 2. SOAL NUMBER PATTERN
    # ─────────────────────────────────────────────────────────────────────────────
    _NO_VARIANTS = r"(?:no(?:mo[rn]|me[rn]|m)?|nomo[rn]|nome[rn])"

    WORD_TO_INT: dict[str, int] = {
        "pertama": 1,
        "satu": 1, "dua": 2, "tiga": 3, "empat": 4, "lima": 5,
        "enam": 6, "tujuh": 7, "delapan": 8, "sembilan": 9, "sepuluh": 10,
        "sebelas": 11,
        "duabelas": 12,   "dua belas": 12,
        "tigabelas": 13,  "tiga belas": 13,
        "empatbelas": 14, "empat belas": 14,
        "limabelas": 15,  "lima belas": 15,
        "enambelas": 16,  "enam belas": 16,
        "tujuhbelas": 17, "tujuh belas": 17,
        "delapanbelas": 18, "delapan belas": 18,
        "sembilanbelas": 19, "sembilan belas": 19,
        "duapuluh": 20,   "dua puluh": 20,
    }

    # Alternatif regex: urutkan panjang → pendek agar tidak partial match
    _WORD_NUM_ALT = "|".join(
        re.escape(w) for w in sorted(WORD_TO_INT.keys(), key=len, reverse=True)
    )
    _WORD_NUM = rf"(?:{_WORD_NUM_ALT})"

    # Ordinal "ke-dua", "ketiga", "ke tiga"
    _ORDINAL_WORD = rf"ke\s*-?\s*(?:{_WORD_NUM_ALT})"

    # Buat prefix yang mendeteksi kata "nomor" ATAU kata hubung (dan, atau, &)
    _PREFIX = r"(?:(?:soal\s+)?" + _NO_VARIANTS + r"\.?\s*|(?:,\s*|\bdan\s+|\batau\s+|&\s+)+)"

    _SOAL_PAT = (
        _PREFIX + r"(\d+)"                                      # G1: digit
        + r"|soal\s+ke\s*-?\s*(\d+)"                            # G2: soal ke-digit
        + r"|" + _PREFIX + r"(" + _WORD_NUM + r")"              # G3: kata bilangan
        + r"|soal\s+(" + _ORDINAL_WORD + r")"                   # G4: ordinal
    )

    SOAL_PATTERN = re.compile(_SOAL_PAT, re.IGNORECASE)

    # ─────────────────────────────────────────────────────────────────────────────
    # 3. HELPER FUNCTIONS
    # ─────────────────────────────────────────────────────────────────────────────
    def _normalize_word_num(self, text: str) -> Optional[str]:
        """Ubah kata bilangan atau ordinal ke string angka."""
        t = text.lower().strip()
        # Hapus prefix ordinal "ke" + tanda hubung/spasi
        t_stripped = re.sub(r"^ke\s*-?\s*", "", t).strip()
        if t_stripped in self.WORD_TO_INT:
            return str(self.WORD_TO_INT[t_stripped])
        if t in self.WORD_TO_INT:
            return str(self.WORD_TO_INT[t])
        return None

    def _extract_soal_id(self, m: re.Match) -> Optional[str]:
        """Ambil id_soal sebagai string dari match object SOAL_PATTERN."""
        if m.group(1):
            return m.group(1)
        if m.group(2):
            return m.group(2)
        if m.group(3):
            return self._normalize_word_num(m.group(3))
        if m.group(4):
            return self._normalize_word_num(m.group(4))
        return None

    def _find_subjects(self, text: str) -> list[tuple[int, int, str]]:
        """Cari semua mention subject, return [(start, end, subject_name), ...]."""
        raw: list[tuple[int, int, str]] = []

        for subject in self.SUBJECT_PRIORITY:
            for pattern in self.SUBJECT_PATTERNS[subject]:
                for m in re.finditer(pattern, text, re.IGNORECASE):
                    raw.append((m.start(), m.end(), subject))

        if not raw:
            return []

        # Sort: posisi naik, panjang match turun (lebih panjang/spesifik menang)
        raw.sort(key=lambda x: (x[0], -(x[1] - x[0])))

        # Hapus overlap — simpan hanya yang tidak bertabrakan dengan match sebelumnya
        result: list[tuple[int, int, str]] = []
        last_end = -1
        for start, end, subject in raw:
            if start >= last_end:
                result.append((start, end, subject))
                last_end = end

        return result

    def _find_soal_numbers(self, text: str) -> list[tuple[int, int, str]]:
        """Cari semua mention nomor soal, return [(start, end, id_soal_str), ...]."""
        result: list[tuple[int, int, str]] = []
        for m in re.finditer(self.SOAL_PATTERN, text):
            id_soal = self._extract_soal_id(m)
            if id_soal:
                result.append((m.start(), m.end(), id_soal))
        return result

    @staticmethod
    def _distance_between_spans(s1s: int, s1e: int, s2s: int, s2e: int) -> int:
        """Jarak karakter antara dua span (0 jika overlap)."""
        if s2s >= s1e:
            return s2s - s1e
        if s1s >= s2e:
            return s1s - s2e
        return 0

    def _find_nearest_subject(
        self,
        soal_start: int,
        soal_end: int,
        subject_spans: list[tuple[int, int, str]],
    ) -> tuple[int, int]:
        """Return (index, distance) subject terdekat dari span soal."""
        best_idx, best_dist = -1, 10**9
        for i, (s_start, s_end, _) in enumerate(subject_spans):
            dist = self._distance_between_spans(soal_start, soal_end, s_start, s_end)
            if dist < best_dist:
                best_dist, best_idx = dist, i
        return best_idx, best_dist

    # ─────────────────────────────────────────────────────────────────────────────
    # 4. MAIN FUNCTION
    # ─────────────────────────────────────────────────────────────────────────────
    def extract_entities(self, query: str) -> list[dict]:
        """
        Ekstrak entity id_soal dan subject dari satu query siswa.
        """
        text = query.lower()

        subject_spans = self._find_subjects(text)
        soal_spans    = self._find_soal_numbers(text)

        if not subject_spans and not soal_spans:
            return []

        if not soal_spans:
            return [{"id_soal": None, "subject": s[2]} for s in subject_spans]

        if not subject_spans:
            return [{"id_soal": s[2], "subject": None} for s in soal_spans]

        if len(subject_spans) == 1:
            subj = subject_spans[0][2]
            return [{"id_soal": s[2], "subject": subj} for s in soal_spans]
        
        if len(soal_spans) == len(subject_spans) and len(soal_spans) > 1:
            return [
                {"id_soal": soal[2], "subject": subj[2]} 
                for soal, subj in zip(soal_spans, subject_spans)
            ]

        results: list[dict] = []
        paired_subj_idx: set[int] = set()

        for s_start, s_end, id_soal in soal_spans:
            idx, _ = self._find_nearest_subject(s_start, s_end, subject_spans)
            results.append({"id_soal": id_soal, "subject": subject_spans[idx][2]})
            paired_subj_idx.add(idx)

        for i, (_, _, subject) in enumerate(subject_spans):
            if i not in paired_subj_idx:
                results.append({"id_soal": None, "subject": subject})

        return results

    def extract_entities_batch(self, queries: list[str]) -> list[list[dict]]:
        """Batch version: proses list of query strings."""
        return [self.extract_entities(q) for q in queries]


# ─────────────────────────────────────────────────────────────────────────────
# 5. TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    test_cases: list[tuple[str, list[dict]]] = [
        ("Di soal nomor 12 penalaran umum, kenapa jawabannya B, bukan C?",
         [{"id_soal": "12", "subject": "Penalaran Umum"}]),
        ("soal nomor 12 bilang pakai konsep limit, bisa dijelaskan ulang?",
         [{"id_soal": "12", "subject": None}]),
        ("Yang nomor 12 penalaram umum itu mirip sama nomor 8 di penalaran matematika nggak sih?",
         [{"id_soal": "12", "subject": "Penalaran Umum"},
          {"id_soal": "8",  "subject": "Penalaran Matematika"}]),
        ("Kalau konsep di nomor 12 penalaran umum dipakai ke nomor 15 penalaran matematika, hasilnya sama nggak?",
         [{"id_soal": "12", "subject": "Penalaran Umum"},
          {"id_soal": "15", "subject": "Penalaran Matematika"}]),
        ("Yang ada grafik naik turun di penalaran matematika itu kenapa jawabannya D ya?",
         [{"id_soal": None, "subject": "Penalaran Matematika"}]),
        ("Kalau limit x mendekati nol itu kan selalu nol, kenapa di soal nomor 3 penalaran matematika nggak?",
         [{"id_soal": "3", "subject": "Penalaran Matematika"}]),
        ("Soal nomor 2 penalaran matematika tentang biaya kirim. Jawaban D Rp32.500. Itu dari mana?",
         [{"id_soal": "2", "subject": "Penalaran Matematika"}]),
        ("Kak, soal no 2 tryout 1 penalaran matematika jawabannya D, bisa dijelaskan cara hitungnya?",
         [{"id_soal": "2", "subject": "Penalaran Matematika"}]),
        ("Di soal no 1 literasi bahasa inggris tryout 1, kenapa jawabannya A bukan D?",
         [{"id_soal": "1", "subject": "Literasi Bahasa Inggris"}]),
        ("Tolong jelaskan pembahasan soal nomor 4 penalaran matematika tryout 1 tentang kecepatan rata-rata.",
         [{"id_soal": "4", "subject": "Penalaran Matematika"}]),
        ("Kak, soal no 3 tryout 1 penalaran matematika jawabannya C, bisa dijelaskan?",
         [{"id_soal": "3", "subject": "Penalaran Matematika"}]),
        ("Jelaskan soal no 2 tryout 1 penalaran matematika kak.",
         [{"id_soal": "2", "subject": "Penalaran Matematika"}]),
        ("Kak jelaskan soal no 4 tryout 1 penalaran matematika.",
         [{"id_soal": "4", "subject": "Penalaran Matematika"}]),
        ("Kak, soal no 5 tryout 1 penalaran matematika itu ada gambar diagram batang kan?",
         [{"id_soal": "5", "subject": "Penalaran Matematika"}]),
        ("soal nomor dua penalaran matematika itu dari mana?",
         [{"id_soal": "2", "subject": "Penalaran Matematika"}]),
        ("no tiga penalaran umum kenapa jawabannya C?",
         [{"id_soal": "3", "subject": "Penalaran Umum"}]),
        ("nomor lima literasi bahasa indonesia itu tentang apa?",
         [{"id_soal": "5", "subject": "Literasi Bahasa Indonesia"}]),
        ("kak jelaskan nomor dua belas penalaran umum dong",
         [{"id_soal": "12", "subject": "Penalaran Umum"}]),
        ("soal kedua penalaran umum itu gimana kak?",
         [{"id_soal": "2", "subject": "Penalaran Umum"}]),
        ("soal ketiga penalaran matematika bisa dijelaskan?",
         [{"id_soal": "3", "subject": "Penalaran Matematika"}]),
        ("soal ke-empat literasi bahasa inggris maksudnya apa?",
         [{"id_soal": "4", "subject": "Literasi Bahasa Inggris"}]),
        ("soal ke tiga penalaran matematika jawabannya C ya?",
         [{"id_soal": "3", "subject": "Penalaran Matematika"}]),
        ("Jelaskan soal kesebelas penalaran umum kak.",
         [{"id_soal": "11", "subject": "Penalaran Umum"}]),
        ("kenapa soal nomor 12 dan tiga belas pada penalarana matematika dan penaralan umum jawabannya begitu?",
         [{"id_soal": "12", "subject": "Penalaran Matematika"},
          {"id_soal": "13", "subject": "Penalaran Umum"}])
    ]

    # Instansiasi class
    extractor = RegexEntityExtractor()

    PASS = "\033[92mPASS\033[0m"
    FAIL = "\033[91mFAIL\033[0m"

    total = passed = 0
    print("=" * 72)
    print("  ENTITY EXTRACTOR — TEST RESULTS")
    print("=" * 72)

    for i, (query, expected) in enumerate(test_cases):
        result = extractor.extract_entities(query)
        ok = result == expected
        total += 1
        if ok:
            passed += 1

        tag = "DIGIT  " if i < 14 else ("KATA   " if i < 18 else "ORDINAL")
        if i == 23: tag = "COMPLEX"
        print(f"\n  [{'PASS' if ok else 'FAIL'}][{tag}] {query!r}")
        print(f"    got:      {json.dumps(result, ensure_ascii=False)}")
        if not ok:
            print(f"    expected: {json.dumps(expected, ensure_ascii=False)}")

    print(f"\n{'='*72}")
    print(f"  Result: {passed}/{total} passed")
    print("=" * 72)