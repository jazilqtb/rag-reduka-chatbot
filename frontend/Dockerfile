# syntax=docker/dockerfile:1.4
# ============================================================================
# UTBK Tutor RAG Frontend - Self-contained Multi-stage Build
# ----------------------------------------------------------------------------
# Stage 1 (builder)  : install deps + npm run build → /app/dist
# Stage 2 (runtime)  : nginx alpine serving static files
#
# BuildKit cache mount mempercepat rebuild dengan persist npm cache antar build.
#
# Untuk environment dengan koneksi npm registry lambat / tidak stabil,
# gunakan Dockerfile.local sebagai alternatif (lihat README).
# ============================================================================


# ─── Stage 1: Builder ───────────────────────────────────────────────────────
FROM node:20-slim AS builder

WORKDIR /app

# Copy package files first untuk leverage Docker layer cache
COPY package.json package-lock.json ./

# BuildKit cache mount untuk npm — package re-use antar build
RUN --mount=type=cache,target=/root/.npm \
    npm ci --no-audit --no-fund

# Copy source & build
COPY . .
RUN npm run build


# ─── Stage 2: Runtime ───────────────────────────────────────────────────────
FROM nginx:alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]