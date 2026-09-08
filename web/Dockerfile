# Standalone Builtryte agent UI. The build context is web/ so this directory
# can move into its own repository without reaching back into Audrey.
FROM node:24.20.0-bookworm-slim@sha256:ba849c60be29959425b8734d57b8b4b7d56f98edd9504c9af091d5281095a71e AS build

WORKDIR /workspace
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

# Verified-publisher, non-root NGINX. Pin both the release and manifest digest
# so a rebuild cannot silently acquire a different runtime.
FROM nginxinc/nginx-unprivileged:1.30.4-alpine3.24@sha256:442753882674b49ae2c1de83ed67896131c0777f56df5005e356e62bc3f7e7ce AS runtime

COPY docker/default.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build --chown=101:101 /workspace/dist /usr/share/nginx/html

USER 101
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -q -O /dev/null http://127.0.0.1:8080/healthz || exit 1
