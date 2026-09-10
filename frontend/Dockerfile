# =============================================================================
# RETINA Frontend - Dockerfile
# =============================================================================
# Multi-stage build for the Next.js frontend application.
#
# Build: docker build -t retina-frontend .
# Run:   docker run -p 3000:3000 retina-frontend
#
# Uses Next.js standalone output for minimal image size.
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Dependencies
# -----------------------------------------------------------------------------
FROM node:20-alpine AS deps

# Install libc6-compat for Alpine compatibility
RUN apk add --no-cache libc6-compat

WORKDIR /app

# Copy package files
COPY package.json package-lock.json* yarn.lock* pnpm-lock.yaml* ./

# Install dependencies based on lock file
RUN \
    if [ -f yarn.lock ]; then yarn --frozen-lockfile; \
    elif [ -f package-lock.json ]; then npm ci; \
    elif [ -f pnpm-lock.yaml ]; then corepack enable pnpm && pnpm i --frozen-lockfile; \
    else npm install; \
    fi

# -----------------------------------------------------------------------------
# Stage 2: Build
# -----------------------------------------------------------------------------
FROM node:20-alpine AS builder

WORKDIR /app

# Copy dependencies from deps stage
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Build arguments for environment configuration
ARG NEXT_PUBLIC_API_URL=http://localhost:3001
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL

# Disable telemetry during build
ENV NEXT_TELEMETRY_DISABLED=1

# Build the application
RUN npm run build

# -----------------------------------------------------------------------------
# Stage 3: Runtime
# -----------------------------------------------------------------------------
FROM node:20-alpine AS runner

WORKDIR /app

# Set production environment
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

# Create non-root user
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

# No public/ directory: this app serves no static assets from a public
# path — no images, favicon, manifest or next/image usage anywhere in
# src/, and frontend/public/ has never existed in this repo. The COPY that
# used to be here referenced it anyway and failed the build outright
# ("/app/public: not found"), so `docker compose up` never reached the
# worker. Everything the runtime actually needs comes from the standalone
# output copied below. Restore a COPY here if real assets are ever added.

# Set proper permissions for prerender cache
RUN mkdir .next && chown nextjs:nodejs .next

# Copy standalone build output
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

# Switch to non-root user
USER nextjs

# Expose the frontend port
EXPOSE 3000

# Set the port environment variable
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

# Health check
# 127.0.0.1, not localhost: inside this image localhost resolves to the IPv6
# [::1] first, while the Next.js standalone server binds IPv4 0.0.0.0, so the
# probe was refused and the container reported unhealthy while serving 200
# perfectly well on the published port.
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://127.0.0.1:3000 || exit 1

# Run the server
CMD ["node", "server.js"]
