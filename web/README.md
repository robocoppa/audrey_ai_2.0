# Builtryte agent UI

This directory is a self-contained React and TypeScript browser application.
Its production image serves static files with non-root NGINX and proxies the
application API on the same origin. Nothing in the build imports files from the
parent Audrey repository, so `web/` can become its own repository after the
standalone deployment passes its live gate.

## Development

Use Node 24 and the committed lockfile:

```bash
npm ci
npm run dev
npm run lint
npm run typecheck
npm test
npm run e2e
npm run build
```

Vite writes the production application to `dist/`. Development and preview
proxy `/api` and `/v1` to the Audrey backend configured in `vite.config.ts`.

## Production container

The image accepts two runtime settings:

- `AUDREY_UPSTREAM`, normally `http://audrey-ai:8000` on `ollama-net`;
- `AUDREY_UI_MAX_BODY_SIZE`, normally `110m`, which must accommodate Audrey's
  configured upload limit.

The proxy forwards the Cloudflare Access assertion, preserves AG-UI and OpenAI
SSE without response buffering, and streams upload requests instead of making a
second full temporary copy. `/healthz` checks only the UI container; Audrey's
own readiness endpoints remain authoritative for backend dependencies.
The upstream is an explicit container setting. If its address changes, update
that setting and recreate `audrey-ui`.

Do not put an Access JWT, API key, or other user credential in this image or its
environment. The browser presents its same-origin Cloudflare session and Audrey
performs authentication and authorization.

## Reuse boundary

The visual shell and deployment container are portable now, but the current
client speaks Audrey-specific resources, AG-UI events, modes, files, and
preferences. Reusing it for another agent platform should add a typed platform
adapter at that boundary. It should not copy Audrey assumptions into generic UI
components or make the browser authoritative for identity, history, or tools.

Keep the UI here through the first standalone production soak. After that gate,
move this directory intact to its own GitHub repository and remove Audrey's
temporary embedded-static fallback in a separate, reversible change.
