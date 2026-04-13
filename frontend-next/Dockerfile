# ── Base: install Node deps ─────────────────────────────────────────────────
# Build context is ./frontend-next — package.json is at the root of that dir.
FROM node:18-alpine AS base

WORKDIR /app

# Install deps first — cached layer, only rebuilt when package files change.
COPY package.json package-lock.json* ./
RUN npm ci

EXPOSE 3000

# ── Dev: source is volume-mounted; Next.js HMR handles hot reload ────────────
FROM base AS dev
CMD ["npm", "run", "dev"]

# ── Prod: bake source, build optimized bundle, serve with next start ─────────
FROM base AS prod
COPY . .
RUN npm run build
ENV NODE_ENV=production
CMD ["npm", "start"]
