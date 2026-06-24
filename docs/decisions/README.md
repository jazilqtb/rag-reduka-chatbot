# Architecture Decision Records

Records of significant design decisions, format ADR (Architecture Decision
Record). Tiap file mengikuti struktur:

- **Context** — situasi & constraint yang relevan
- **Options considered** — alternatif yang dipertimbangkan
- **Decision** — pilihan yang diambil
- **Trade-offs accepted** — yang sengaja dikorbankan
- **Consequences** — implikasi follow-up

Untuk konteks naratif yang lebih luas, lihat [`../journal.md`](../journal.md).

## Index

| # | Title | Status |
|---|---|---|
| [0001](0001-polyglot-persistence.md) | Polyglot persistence: Redis + PostgreSQL | Accepted |
| [0002](0002-layered-retrieval.md) | 4-layer cost-optimized retrieval strategy | Accepted |
| [0003](0003-hybrid-history.md) | Hybrid history with rolling summary | Accepted |
| [0004](0004-incremental-ingestion.md) | Incremental ingestion (per-source delete-then-insert) | Accepted |