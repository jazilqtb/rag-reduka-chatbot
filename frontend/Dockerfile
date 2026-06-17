# ============================================================================
# UTBK Tutor RAG Frontend - Multi-stage build
# Stage 1 (builder)  : install + npm run build → /app/dist
# Stage 2 (runtime)  : nginx alpine serving static files
# ============================================================================


# ─── Stage 1: Builder ───────────────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files first to leverage docker cache
COPY package*.json ./
RUN npm ci

# Copy source & build
COPY . .
RUN npm run build


# ─── Stage 2: Runtime ───────────────────────────────────────────────────────
FROM nginx:alpine AS runtime

# Custom nginx config: serve dist + SPA fallback
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built assets
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
