# Campaign 2 Phase 34 — the media-worker container (ffmpeg, no model)

Turn [phase 33](phase-33-video-job-lifecycle.md)'s stub into a real container
that claims real jobs and does real work with the file — demux the audio, report
its duration, hand it back. No transcription, no model calls, no GPU.

**Status: PLANNED.**

**New container — adds a `media-worker` service to `compose.yaml`.** This is the
first compose change of the video work, and the reason this is its own phase: a
new image, a new network path, and a new set of things that can be missing from
a container is enough surface to verify on its own.

---

## Design decisions

### The sidecar isolates CPU work, not GPU work

Demux, resample and (later) whisper are CPU-bound and long. Running them in
`audrey-ai` puts minutes of work in the request path of a process that is also
serving chat. That is the whole reason for a separate container.

GPU work is the opposite case and stays in `audrey-ai` — see
[phase 36](phase-36-video-visual-assessment.md), where every model call goes back
through passthrough rather than out to Ollama directly.

### The worker holds no Ollama address

This is the invariant to establish now, while the worker has no model calls at
all, so it is already true before phase 36 gives it a reason to break it. The
worker's only outbound target is `audrey-ai`. It holds no `OLLAMA_HOST`, and is
not on a network path that reaches Ollama.

Once it can reach Ollama directly, someone will eventually make it — and every
fairness guarantee in [`scheduling.py`](../../src/audrey/scheduling.py) is
bypassed the moment they do, silently and only under load.

### We extract, we never transform

ffmpeg is used to pull an audio stream and later to pull frames. No re-encoding,
no proxy transcodes, no format normalisation of the stored video. The source is
read, never rewritten.

## What's in scope

- **`docker/media-worker.Dockerfile`** (new) — `ffmpeg`, which is installed in
  no current image, plus the python client. Whisper weights come in
  [phase 35](phase-35-video-transcript.md).
- **`compose.yaml`** — the `media-worker` service: `env_file: - .env` for
  `KB_SERVICE_TOKEN`, a read-only mount of the uploads volume, and no ports.
- **`src/audrey/media/worker.py`** (new) — the claim loop, the audio extraction
  call, the result post. Imports nothing from `audrey.routes` or `audrey.kb`;
  it talks HTTP like any other client.
- **`src/audrey/media/audio.py`** (new) — the ffmpeg invocation and its failure
  modes, separated so it can be tested against a fixture without a container.
- **`config.yaml`** — `kb.video.poll_seconds` and the worker's audio settings.

## What's NOT in scope

- **No whisper.** [Phase 35](phase-35-video-transcript.md).
- **No frame extraction.** [Phase 36](phase-36-video-visual-assessment.md).
- **No model calls of any kind.** The worker has no reason to speak to a model
  yet and should not be given credentials to try.
- **No autoscaling or multiple replicas.** One worker. The lease design permits
  more later; nothing here tests it.

## The parts that will bite

- **ffmpeg is not in any current image.** The Dockerfile is where this phase
  actually lives; most of the risk is in the image, not the code.
- **Late rejection is worse with chunking.** A user can push 300 MB across 40
  requests before anything notices the container is unreadable. Probing the
  first part's container header at session start is the real fix, and it belongs
  here rather than in the worker — by the time the worker sees the file, the
  bytes are already spent.
- **A read-only mount is not a detail.** The worker reads the source and posts
  results over HTTP. It must not be able to write the uploads volume, or the
  single-writer argument in phase 33 quietly becomes untrue for files even if it
  stays true for sqlite.
- **Neither container has `curl`.** Probes run inside via `python3`.
- **Silent video is not a failure.** A file with no audio stream must return a
  duration of zero and a successful result, not an error. It may still have rich
  visual content, and phase 36 will want it.

## Deploy on Unraid

From `/mnt/user/appdata/audrey_ai_2.0`:

```
docker compose up -d --build media-worker
```

`audrey-ai` needs no rebuild — phase 33 already shipped the routes.

## Verification (to be written against the built phase)

**0. ffmpeg is present.**

```
docker exec media-worker sh -c 'command -v ffmpeg || echo MISSING'
```

**1. The worker cannot reach Ollama.** This is the fairness invariant; if it
regresses, the gate stops meaning anything in phase 36.

```
docker exec media-worker python3 -c "import os; print(os.environ.get('OLLAMA_HOST', 'UNSET'))"
docker exec media-worker python3 -c "
import socket; s=socket.socket(); s.settimeout(3)
try: s.connect(('192.168.1.11', 11434)); print('REACHABLE - fix the network')
except Exception as e: print('unreachable, correct:', type(e).__name__)"
```

**2. It claims a real job** and the row moves `pending` → `processing`.

**3. It reports a plausible duration** for the uploaded mp4, matching
`ffprobe` run by hand on the same file.

**4. A video with no audio track succeeds** with duration zero rather than
failing.

**5. The uploads mount is read-only.**

```
docker exec media-worker sh -c 'touch /data/uploads/.probe && echo WRITABLE || echo read-only'
```

**6. A killed worker's job returns to the queue.** `docker kill media-worker`
mid-job; the row must reach `pending` again via the phase 33 lease sweep, not
sit in `processing`.

### Rollback

`docker compose stop media-worker`. Jobs go back to accumulating as `pending`,
which is exactly the phase 32 state — uploads keep working throughout.

## What this unblocks

There is now a real process, in a real container, holding a real video file
with ffmpeg available. [Phase 35](phase-35-video-transcript.md) gives it
something worth doing with the audio it just learned to extract.
