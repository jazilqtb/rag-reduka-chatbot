# ============================================================================
# UTBK Tutor RAG System - Makefile
# ----------------------------------------------------------------------------
# Shortcut commands for common dev tasks.
# Run `make help` (or just `make`) to see all available commands.
# ============================================================================

.DEFAULT_GOAL := help
.PHONY: help install install-backend install-frontend dev dev-frontend \
        test lint format up up-infra down restart logs ps \
        seed schema-dump clean clean-data


# ─── Help (default target) ──────────────────────────────────────────────────
help:
	@echo "UTBK Tutor RAG System - Available commands:"
	@echo ""
	@echo "  Setup:"
	@echo "    make install            Install backend (venv) + frontend (npm) deps"
	@echo "    make install-backend    Install only backend deps"
	@echo "    make install-frontend   Install only frontend deps"
	@echo ""
	@echo "  Development:"
	@echo "    make dev                Run backend in dev mode (uvicorn --reload)"
	@echo "    make dev-frontend       Run frontend in dev mode (vite)"
	@echo ""
	@echo "  Quality:"
	@echo "    make test               Run backend tests with pytest + coverage"
	@echo "    make lint               Lint backend code with ruff"
	@echo "    make format             Format backend code (black + ruff fix)"
	@echo ""
	@echo "  Docker:"
	@echo "    make up                 Start full stack (postgres + redis + backend)"
	@echo "    make up-infra           Start only postgres + redis (for local dev)"
	@echo "    make down               Stop all services"
	@echo "    make restart            Restart the backend service"
	@echo "    make logs               Tail logs from all services"
	@echo "    make ps                 List running services"
	@echo ""
	@echo "  Data:"
	@echo "    make seed               Seed demo data (sample PDFs + ingest)"
	@echo "    make schema-dump        Dump current postgres schema to docs/schema.sql"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean              Remove caches (pycache, pytest, etc.)"
	@echo "    make clean-data         Remove all uploaded data + vector store"


# ─── Setup ──────────────────────────────────────────────────────────────────
install: install-backend install-frontend

install-backend:
	@echo "→ Creating backend venv & installing dependencies..."
	cd backend && python3 -m venv .venv \
	    && ./.venv/bin/pip install --upgrade pip \
	    && ./.venv/bin/pip install -r requirements-dev.txt
	@echo "✓ Backend ready. Activate with: source backend/.venv/bin/activate"

install-frontend:
	@if [ -d frontend ]; then \
	    echo "→ Installing frontend dependencies..." ; \
	    cd frontend && npm install ; \
	    echo "✓ Frontend ready." ; \
	else \
	    echo "ℹ frontend/ does not exist yet — skipping." ; \
	fi


# ─── Development ────────────────────────────────────────────────────────────
dev:
	cd backend && ./.venv/bin/uvicorn main:app --reload --host 0.0.0.0 --port 8000

dev-frontend:
	cd frontend && npm run dev


# ─── Quality ────────────────────────────────────────────────────────────────
test:
	cd backend && ./.venv/bin/pytest -v --cov=src --cov-report=term-missing

lint:
	cd backend && ./.venv/bin/ruff check src tests

format:
	cd backend && ./.venv/bin/black src tests \
	    && ./.venv/bin/ruff check --fix src tests


# ─── Docker ─────────────────────────────────────────────────────────────────
up:
	docker compose up -d
	@echo ""
	@echo "✓ Services starting. Useful URLs:"
	@echo "  Backend docs : http://localhost:$${BACKEND_PORT:-8000}/docs"
	@echo "  Health check : http://localhost:$${BACKEND_PORT:-8000}/v1/health"
	@echo ""
	@echo "  Check status: make ps"

up-infra:
	docker compose up -d postgres redis
	@echo ""
	@echo "✓ Infra ready. Backend can now connect to:"
	@echo "  PostgreSQL : localhost:$${POSTGRES_PORT:-5432}"
	@echo "  Redis      : localhost:$${REDIS_PORT:-6379}"
	@echo ""
	@echo "  Run backend locally: make dev"

down:
	docker compose down

restart:
	docker compose restart backend

logs:
	docker compose logs -f

ps:
	docker compose ps


# ─── Data ───────────────────────────────────────────────────────────────────
seed:
	@if [ ! -f backend/scripts/seed_demo.py ]; then \
	    echo "ℹ backend/scripts/seed_demo.py does not exist yet (Stage 6)." ; \
	    exit 1 ; \
	fi
	cd backend && ./.venv/bin/python scripts/seed_demo.py

schema-dump:
	docker compose exec -T postgres pg_dump \
	    -U $${POSTGRES_USER:-tutor} \
	    -d $${POSTGRES_DB:-tutor_utbk} \
	    --schema-only --no-owner --no-privileges \
	    > docs/schema.sql
	@echo "✓ Schema dumped to docs/schema.sql"


# ─── Cleanup ────────────────────────────────────────────────────────────────
clean:
	@find . -type d -name __pycache__   -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name .pytest_cache -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name .ruff_cache   -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name .mypy_cache   -prune -exec rm -rf {} + 2>/dev/null || true
	@rm -rf backend/.coverage backend/htmlcov
	@echo "✓ Caches cleaned."

clean-data:
	@echo "⚠  This will permanently delete:"
	@echo "   - backend/data/vector_store/*"
	@echo "   - backend/data/debug/*"
	@echo "   - backend/data/raw_docs/*.pdf"
	@echo ""
	@echo "   Press Ctrl+C to abort, or wait 5 seconds to continue..."
	@sleep 5
	@rm -rf backend/data/vector_store/* backend/data/debug/* backend/data/raw_docs/*.pdf 2>/dev/null || true
	@echo "✓ Data cleaned."