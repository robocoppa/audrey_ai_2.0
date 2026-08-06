# Media fetcher — yt-dlp sidecar for URL video ingest (Phase 41)
#
# Build from the repo root:
#   docker build -f docker/media-fetcher.Dockerfile -t media-fetcher:latest .
#
# The third container, and the only one with internet egress. That is the whole
# reason it exists: `media-worker` is on an `internal: true` network on purpose
# (Phase 34), and giving it egress would restore its route to the ollama port
# published on the host — undoing the fairness invariant by topology. `audrey-ai`
# could download, but a multi-minute transfer has no business occupying the API
# container, and yt-dlp needs updating whenever YouTube changes something, which
# would couple that cadence to rebuilding the API image.
#
# Same narrow-copy approach as media-worker.Dockerfile: this image does NOT
# install the `audrey` package. It imports `audrey.media.fetch`,
# `audrey.media.fetcher` and `audrey.media.service`, none of which import
# anything from the app.

# Same digest-pinned base as audrey.Dockerfile (Phase 31 convention). To bump:
#   docker pull python:3.12-slim
#   docker inspect --format='{{index .RepoDigests 0}}' python:3.12-slim
FROM python:3.12-slim@sha256:46cb7cc2877e60fbd5e21a9ae6115c30ace7a077b9f8772da879e4590c18c2e3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# ffmpeg is here for ONE job: merging the separate video and audio streams a
# site serves for anything above its highest pre-muxed quality. Without it the
# format selector is limited to single-file formats, which on YouTube means
# 360p — enough for a transcript, not enough to read a slide, and the visual
# pass exists to read slides.
#
# ca-certificates is not optional: this is the only container that makes
# outbound TLS connections, and python:slim ships without a trust store.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── yt-dlp ───────────────────────────────────────────────────────────
# PINNED, and pinned deliberately rather than out of habit. yt-dlp's own
# documentation recommends auto-updating, which is reasonable advice for a
# desktop tool and wrong here: this container has write access to every user's
# upload directory, and a downloader that fetches and executes new code at
# runtime is a supply-chain path into that directory that nobody would see.
#
# The cost is real and is accepted: YouTube changes something every few months
# and an unpinned yt-dlp would have fixed it before you noticed. Here the
# symptom is fetches failing with a reason from `friendly_reason`, and the fix
# is bumping this line and rebuilding — a deliberate act, on the schedule of
# whoever runs the box.
ARG YTDLP_VERSION=2025.06.30
RUN pip install --no-cache-dir "yt-dlp==${YTDLP_VERSION}"

# Just the three modules it imports, and the version marker their parent needs.
# `src/audrey/__init__.py` is a single `__version__` assignment with no imports,
# which is what makes this narrow copy work.
#
# NOT the whole `media` package: `worker.py` imports faster-whisper's wrapper
# and `describe.py`, and copying them here would mean this image either carries
# whisper it never runs or fails at import on a module it never uses.
COPY src/audrey/__init__.py           /app/src/audrey/__init__.py
COPY src/audrey/media/__init__.py     /app/src/audrey/media/__init__.py
COPY src/audrey/media/service.py      /app/src/audrey/media/service.py
COPY src/audrey/media/fetch.py        /app/src/audrey/media/fetch.py
COPY src/audrey/media/fetcher.py      /app/src/audrey/media/fetcher.py

# Non-root — but unlike media-worker this one WRITES to the shared mount, so
# the uid matters rather than being hygiene. It must be able to create files
# under the uploads root that audrey-ai can then read and unlink.
RUN useradd --system --create-home --uid 10002 fetcher
USER fetcher

# No healthcheck. A polling sidecar with no listening socket has nothing to
# probe — "is it running" is `docker ps`, and "is it working" is the queue
# draining. Same reasoning as media-worker.

CMD ["python", "-m", "audrey.media.fetcher"]
