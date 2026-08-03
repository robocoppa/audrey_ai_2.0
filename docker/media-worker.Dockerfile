# Media worker — ffmpeg sidecar for video ingest (Phase 34)
#
# Build from the repo root:
#   docker build -f docker/media-worker.Dockerfile -t media-worker:latest .
#
# This image deliberately does NOT install the `audrey` package. The worker
# talks to Audrey over HTTP and imports only `audrey.media.*`, which is stdlib
# -only by design (see src/audrey/media/__init__.py). Installing pyproject.toml
# would drag in fastapi, qdrant-client and sentence-transformers — the last of
# which pulls torch and turns a ~200 MB image into a multi-gigabyte one, for
# code that is never imported here.
#
# So: copy the two files it needs, put them on PYTHONPATH, done. If this ever
# starts needing a third-party package, add it explicitly below rather than
# reaching for the app's dependency list.

# Same digest-pinned base as audrey.Dockerfile (Phase 31 convention). To bump:
#   docker pull python:3.12-slim
#   docker inspect --format='{{index .RepoDigests 0}}' python:3.12-slim
FROM python:3.12-slim@sha256:46cb7cc2877e60fbd5e21a9ae6115c30ace7a077b9f8772da879e4590c18c2e3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# ffmpeg is the entire point of this image and is in no other image we build.
# `ffmpeg` in Debian brings ffprobe with it; both are used.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Just the media package and the version marker its parent needs to import.
# `src/audrey/__init__.py` is a single `__version__` assignment with no
# imports, which is what makes this narrow copy work at all.
#
# NOTE: this also brings `media/framegate.py`, which imports Pillow — not
# installed here. Nothing imports it today, so it is inert. Phase 36 is when
# that stops being true: adding the frame path means adding `pillow` to the
# apt/pip step below, and the failure if you forget is an ImportError at claim
# time, on the box, not at build time.
COPY src/audrey/__init__.py       /app/src/audrey/__init__.py
COPY src/audrey/media             /app/src/audrey/media

# Non-root. The worker only ever reads a read-only mount and writes its own
# scratch dir, so it has no reason to run as root.
RUN useradd --system --create-home --uid 10001 worker \
    && mkdir -p /var/tmp/media-worker \
    && chown -R worker:worker /var/tmp/media-worker
USER worker

ENV WORK_DIR=/var/tmp/media-worker

# No healthcheck. A polling worker with no listening socket has nothing to
# probe — "is it running" is `docker ps`, and "is it working" is the queue
# draining. A fake HTTP endpoint here would only be something else to break.

CMD ["python", "-m", "audrey.media.worker"]
