# Lesson 12 — The KB lifecycle: watcher, reconcile, uploads

**Estimated time:** 70-90 minutes if you keep
[`kb/watcher.py`](../../src/audrey/kb/watcher.py),
[`kb/reconcile.py`](../../src/audrey/kb/reconcile.py),
[`kb/uploads_db.py`](../../src/audrey/kb/uploads_db.py), and
[`routes/files.py`](../../src/audrey/routes/files.py) open.

**Goal:** by the end of this lesson, you can answer
*"after a file lands on disk (or gets uploaded through the browser),
what keeps the KB in step with reality — and what catches drift when
something slips through?"*

Lesson 11 built the KB pipeline: file → chunks → embeddings → Qdrant
points → searchable hits. It did the *building*. This lesson covers
the *keeping in step* — three mechanisms that watch, repair, and
extend the index over time.

There are three ideas to keep separate:

```text
watcher    - filesystem events → debounced ingest/delete (real-time)
reconcile  - scheduled sweep that catches what the watcher missed
uploads    - per-user side: browser upload → sqlite index + qdrant + bytes
```

## 1. Context

### Why three different mechanisms?

The pipeline from Lesson 11 takes a file and produces points. Once
those points are in Qdrant, three things can happen that the
pipeline itself doesn't notice:

1. **A file on disk changes.** Someone edits, moves, or deletes a
   file in `/datasets`. Qdrant has no idea — it'll happily serve the
   old chunks until something tells it otherwise.
2. **A change happens while Audrey isn't watching.** Container
   restart, `KB_WATCHER_ENABLED=0` for an afternoon, a bulk
   filesystem operation on a filesystem watchdog struggles with —
   any of these leave Qdrant out of step with the disk.
3. **A user uploads a file through the browser.** That file never
   lives in `/datasets`; it lives in a per-user directory and
   indexes into per-user Qdrant collections. Different lifecycle,
   different permissions, different cleanup story.

Each of those problems calls for a different mechanism. The watcher
handles (1) in real time; the reconciler is the catch-up for (2);
the uploads flow is its own pipeline for (3) with its own sqlite
metadata index. They share the ingest helpers from Lesson 11 but
otherwise run independently.

### The "two sources of truth" problem

Stretching across all three mechanisms is one design issue worth
naming up front: **Audrey keeps file metadata in two stores at
once.** Qdrant holds vector content and a small payload per point.
The uploads-side `sqlite` database holds row-per-file metadata for
fast list and quota lookups. Both stores describe the same files;
keeping them coherent is the central engineering problem of the
uploads half of this lesson.

The pattern that makes it work shows up twice in this lesson — once
for global on-disk content (the reconciler) and once for per-user
uploads (the startup reconcile in `uploads_db.py`). Watch for it.

### What sqlite is, and why it's here alongside Qdrant

Lesson 11 introduced Qdrant. This lesson brings in a second
database — **sqlite** — for a different job. Before any code, the
vocabulary:

- **sqlite** is an embedded relational database. "Relational" means
  it stores rows in tables with named columns, exactly like
  PostgreSQL or MySQL, and you query it with SQL (`SELECT
  filename FROM uploads WHERE user = ? ORDER BY uploaded_at DESC`).
  "Embedded" means it isn't a separate service — there's no
  `sqlite` server process to start, no port to connect to. The
  entire database is a single file on disk
  (`/data/uploads.db` in Audrey's case) and a library you import
  in Python (`import sqlite3` is in the standard library) opens
  that file directly. Your Python process *is* the database
  engine.

- **Qdrant**, by contrast, is a standalone server. It runs in its
  own container, listens on a port, and Audrey talks to it over
  HTTP. It stores vectors (not rows) and answers similarity
  queries (not SQL).

These two stores are good at very different things:

| | sqlite | Qdrant |
|---|---|---|
| **Stores** | Rows in tables (filename, mime, bytes, …) | Vectors with attached payload |
| **Lookup by** | Exact-match keys and indexed filters (SQL `WHERE`) | Vector similarity (cosine nearest-neighbour) |
| **Best at** | "Give me row where `file_id = 'abc'`" — O(1) | "Find the K vectors closest to this query" — milliseconds |
| **Bad at** | Vector search (no notion of similarity at all) | Exact-match lookups *(slower than sqlite for "is this file_id present?")* |
| **Runs as** | A library inside Audrey's process; data is one `.db` file | A separate service in its own container, talking HTTP |
| **Data shape** | Rigid schema; columns and types declared up front | Vector + free-form JSON payload per point |

Both stores describe the *same* uploaded files, but they answer
different questions about them. Qdrant answers "what's in this
file's content that's similar to the user's query?" The sqlite
index answers "what files does this user have, how big are they,
when were they uploaded?" — questions about the file's identity,
not its content.

You could in principle answer the sqlite questions by scrolling
Qdrant and reading payloads — but scrolling a collection means
walking every point (one per chunk), and a single 50-page PDF is
50 points. Listing a user's 20 files would mean walking ~1000
points and grouping them. sqlite reduces that to one indexed
`SELECT` per request, completing in microseconds.

That's the why behind the two-store design: each store does what
it's best at, and the uploads flow's job is to keep them in step.

### Pieces and where they live

| Concern | Lives in | What it owns |
|---|---|---|
| Filesystem event watching | [`kb/watcher.py`](../../src/audrey/kb/watcher.py) | `KBWatcher`, debounced event loop, delete/ingest dispatch |
| Periodic orphan cleanup | [`kb/reconcile.py`](../../src/audrey/kb/reconcile.py) | `KBReconciler`, `reconcile_once()`, scroll-and-check |
| Per-user upload metadata | [`kb/uploads_db.py`](../../src/audrey/kb/uploads_db.py) | `UploadsDB`, sqlite schema, startup `reconcile_with_qdrant` |
| Upload HTTP endpoints | [`routes/files.py`](../../src/audrey/routes/files.py) | `POST/GET/DELETE /v1/files` |
| Upload UI page | [`routes/upload_ui.py`](../../src/audrey/routes/upload_ui.py) | `GET /upload` (static HTML) |


## 2. Read-along

### 2.1 The watcher's mental model

The watcher answers a simple question: when a file in `/datasets`
changes, can Audrey notice and re-ingest *without* waiting for the
next scheduled sweep? Yes — via the `watchdog` library, which uses
inotify on Linux to subscribe to kernel-level filesystem events.

The complication is that `watchdog` runs its observer on a **native
OS thread**, while Audrey is a single-threaded asyncio process. The
watcher's job is to bridge those two worlds cleanly.

The shape from
[`kb/watcher.py:106`](../../src/audrey/kb/watcher.py#L106):

```python
class KBWatcher:
    def __init__(self, *, roots, qdrant, text_embedder,
                 image_embedder, debounce_s=2.0, ...):
        ...
        self._observer: Observer | None = None
        self._task: asyncio.Task | None = None
        self._queue: asyncio.Queue[tuple[EventKind, Path]] | None = None
```

Three moving parts that work in concert:

- **`_observer`** — a watchdog Observer running on a native thread.
  Receives filesystem events from the kernel and calls our handler.
- **`_queue`** — an `asyncio.Queue` that the handler pushes events
  onto. Lives in the asyncio side.
- **`_task`** — a long-lived async task that pulls from the queue,
  debounces, and dispatches ingest / delete.

The handler bridges thread to async. The task does the actual work.
Calling `start()` spins both up; `stop()` tears both down.

### 2.2 Concept spotlight — the thread-to-async bridge

This is one of the few places in Audrey where two concurrency models
meet. Worth slowing down.

`asyncio.Queue` is **not thread-safe**. Calling `queue.put_nowait()`
from a non-asyncio thread is undefined behavior — the asyncio loop
maintains internal state that assumes only the loop's own callbacks
mutate it. You can't just call queue methods from watchdog's thread.

The standard fix: `loop.call_soon_threadsafe()`. It schedules a
callback to run *on the asyncio loop's own thread*, from any other
thread, safely. The watcher uses this at
[`kb/watcher.py:88-104`](../../src/audrey/kb/watcher.py#L88):

```python
def _enqueue(self, kind, raw_path, event):
    if event.is_directory:
        return
    if not raw_path:
        return
    path = Path(raw_path)
    if any(part.startswith(".") for part in path.parts):
        return
    if path.suffix.lower() not in _ALL_SUFFIXES:
        return
    # asyncio.Queue isn't thread-safe; schedule the put on the loop.
    self._loop.call_soon_threadsafe(self._queue.put_nowait, (kind, path))
```

Read the last line carefully: the watchdog thread is asking the
asyncio loop to call `self._queue.put_nowait((kind, path))` on its
own thread the next time it gets a chance. That's the entire bridge.

Two filters apply *before* the put:

- **Dot-path filter** — any path component starting with `.` (the
  file itself or any ancestor directory) is skipped. So a stray
  `.git/HEAD` or `.cache/tmp.md` doesn't enter the queue. Mirrors
  the same rule the offline `_iter_files` crawl uses.
- **Suffix allowlist** — only the suffixes we know how to ingest
  (`.md`, `.pdf`, `.docx`, `.html`, `.htm`, image suffixes, etc.)
  pass through. Editor swap files (`.swp`, `.tmp`, `~`) get
  dropped here.

Filtering at enqueue time means the asyncio side never sees events
it would only have to drop later. Cheaper.

### 2.3 Concept spotlight — Debouncing

**Debouncing is the general technique of *waiting for things to
stop changing before acting on them*.** The name comes from
electrical engineering — a physical switch, when you press it,
doesn't make one clean contact. It physically bounces against the
metal a few times before settling, producing a burst of rapid
on/off signals over a few milliseconds. If a circuit reacted to
every signal, a single press would register as ten. The fix is to
wait until the signal has been quiet for some short interval, then
treat the *last* signal as the real one and ignore the bounce.

Software borrows the idea anywhere it has to react to a noisy
source — a stream of events where a single logical action produces
many physical events, arriving in rapid bursts. UI frameworks
debounce "search as you type" so a query only fires after the user
stops typing for 300 ms. Backoff/retry systems debounce reconnect
attempts. Build systems debounce file-change triggers so saving a
file in your editor doesn't kick off three rebuilds. The general
shape is always the same:

```text
stamp-on-event   - every new event updates "last seen" time
dispatch-on-silence - act only after the source has been quiet
                     for the debounce window
```

**Why the KB watcher needs it.** Inotify (Linux's filesystem-event
mechanism, which `watchdog` wraps) is exactly the noisy source the
technique was invented for. A single user action produces many
events:

- **Editors save by writing a temp file, `rename()`ing it over the
  original, and sometimes touching the mtime separately.** Three
  events on disk for one save.
- **`cp -r /src /datasets/topic/`** of 500 files generates
  created-and-then-modified events as each file is written byte by
  byte, and inotify can deliver them in clusters. A 500-file copy
  might fire 1500 events arriving in rapid bursts.
- **Autosave-every-keystroke editors** turn a paragraph of typing
  into 200+ modify events on the same file in 30 seconds.

If the watcher reacted to every event the moment it arrived, you'd
get 1500 simultaneous ingests racing each other against Qdrant —
wasteful at best, broken at worst. Each one would also embed the
file's *partial* state, since `cp` hasn't finished writing yet.
Debouncing collapses each storm into one action per file, acting
on the file's settled final state.

**How the watcher implements it.** The asyncio side lives in
`KBWatcher._run` —
[`kb/watcher.py:156`](../../src/audrey/kb/watcher.py#L156):

```python
async def _run(self) -> None:
    assert self._queue is not None
    pending: dict[tuple[EventKind, Path], float] = {}
    while True:
        try:
            timeout = self._debounce_s if pending else None
            kind, path = await asyncio.wait_for(self._queue.get(), timeout=timeout)
            pending[(kind, path)] = time.monotonic()
        except TimeoutError:
            pass
        except asyncio.CancelledError:
            return
        now = time.monotonic()
        due = [k for k, t in pending.items() if now - t >= self._debounce_s]
        # Process deletes before ingests for the same path.
        due.sort(key=lambda k: 0 if k[0] == "delete" else 1)
        for key in due:
            pending.pop(key, None)
            kind, path = key
            if kind == "delete":
                await self._delete_vectors(path)
            else:
                await self._handle_ingest(path)
```

Three layers, mapped onto the general shape above:

**The wait.** `asyncio.wait_for(self._queue.get(), timeout=...)`
either pulls one event off the queue or times out after
`debounce_s` seconds. If `pending` is empty there's nothing to
debounce, so the timeout is `None` (wait forever for the next
event); otherwise the timeout matches the debounce window so we
wake up exactly when something *might* be due.

**The stamp-and-overwrite.** `pending[(kind, path)] = time.monotonic()`
is the "stamp-on-event" rule made concrete. `pending` is a dict
keyed by `(kind, path)`, so every new event for the same key
*overwrites* the timestamp. The dict only ever holds the *latest*
arrival time for each unique (event-type, file) pair. A file
firing ten modify events in two seconds collapses to one entry
whose timestamp keeps advancing.

**The dispatch-on-silence.** When the timeout fires,
`[k for k, t in pending.items() if now - t >= self._debounce_s]`
picks out every entry whose stamp is older than the window — i.e.
every (kind, path) that has been quiet long enough. Those get
dispatched once each, with deletes sorted before ingests. (Why
deletes first? Because a `mv old.md new.md` produces a delete-old
+ ingest-new pair. Ingesting new.md before deleting old.md could
let the delete race the new ingest's own no-op `delete_by_source`.
Sorting keeps the order deterministic.)

**The trade-off.** Debouncing buys you collapsed bursts at the
cost of *latency*. A single save isn't visible to search for
`debounce_s` seconds after it lands. With Audrey's 2 second window
that's imperceptible for most uses, but it's the literal mechanism
behind "why didn't my edit show up in search yet?" — sit still
for two seconds and try again.

The edge case is the autosave-on-keystroke editor. Because every
new keystroke stamps the same entry again, the timer *never*
expires while typing is active. The file flushes only when the
user pauses for the full window. That's the intended "wait until
things settle" behavior — not a bug, just the cost of choosing
debouncing over a fixed-interval flush.

**Where you'll see this pattern again.** Anywhere a system reacts
to high-frequency events from a noisy source: UI search-as-you-type
(200-500 ms typical), button presses on physical hardware, websocket
reconnect attempts with backoff, autosave-to-cloud-storage,
config-reload-on-file-change. The general shape — stamp-on-event,
dispatch-on-silence — is the same; only the window length and the
final action change.

### 2.4 Watcher delete handling

When a file disappears (`rm`, the destination side of a rename
already processed), we want to delete its vectors from whichever
collection actually has them — and only that one. The path's suffix
tells us which: a text suffix (`.md`, `.txt`, `.pdf`, …) can only
have lived in `kb_text`; an image suffix only in `kb_images`. We
skip the wrong-collection call rather than firing both, which
roughly halves watcher-driven Qdrant delete load on bulk operations
([`kb/watcher.py:215`](../../src/audrey/kb/watcher.py#L215)):

```python
async def _delete_vectors(self, path: Path) -> None:
    src = str(path)
    suffix = path.suffix.lower()
    try:
        if suffix in IMAGE_SUFFIXES:
            if self._image is not None:
                await self._qdrant.delete_by_source(
                    src, collection=self._qdrant.image_collection,
                )
        else:
            await self._qdrant.delete_by_source(
                src, collection=self._qdrant.text_collection,
            )
        log.info("kb.watcher: requested delete of vectors for %s", path)
    except Exception as e:
        log.warning("kb.watcher: delete %s failed: %s", path, e)
```

Three things to internalize from this small block:

- **Branch on the suffix.** It's reliable here because the same
  `_ALL_SUFFIXES` allowlist gated the file into the queue at
  `_QueueHandler._enqueue`. Anything that reaches `_delete_vectors`
  has a known-text or known-image suffix; nothing else gets through.
- **Use the QdrantKB-supplied collection names** (`self._qdrant.text_collection`,
  not the literal string `"kb_text"`). If a future deployment
  renames collections in `config.yaml`, the watcher honors it
  automatically. The same principle applies anywhere in the
  codebase: ask the wrapper for its configured names.
- **`except Exception` with a log line.** The watcher must stay
  alive. If one delete fails (Qdrant transport hiccup, a malformed
  payload, anything), we log it and move on. Letting the exception
  propagate would kill the asyncio task and stop processing all
  future events — much worse than missing one cleanup.

### 2.5 Watcher boundaries — what it can't catch

The watcher is reactive: it does the right thing as long as it's
running. Anything that happens while it's *not* running is invisible
to it. Concrete cases:

- The container restarts. Watchdog wasn't subscribed for the
  reboot window.
- `KB_WATCHER_ENABLED=0` was set for a stretch (e.g. during a
  bulk reingest to avoid double-processing).
- A bulk `rm -rf /datasets/topic` finished before the observer
  could thread all the events through.
- An atomic-rename pattern that some filesystems implement in a
  way watchdog doesn't observe cleanly.

In every one of these, Qdrant ends up holding vectors for files
that no longer exist on disk. They sit there forever — searchable
orphans pointing at content the user can't reach. That's the
gap reconcile exists to close.

### 2.6 Reconcile — the catch-up pass

The contract is simple: walk every point in `kb_text` and
`kb_images`, group by `payload.source`, check whether each unique
source path still exists on disk, and call `delete_by_source` for
any that don't.

[`kb/reconcile.py:88`](../../src/audrey/kb/reconcile.py#L88):

```python
async def _scroll_sources(qdrant: QdrantKB, *, collection: str) -> dict[str, int]:
    by_source: dict[str, int] = {}
    for _point_id, payload in await qdrant.scroll_collection(
        collection, page_size=_SCROLL_PAGE_SIZE,
    ):
        source = str(payload.get("source") or "")
        if not source:
            continue
        by_source[source] = by_source.get(source, 0) + 1
    return by_source
```

`qdrant.scroll_collection` is the facade method that hands us
`[(point_id, payload), ...]` for every point in the collection.
Reconcile aggregates by `source` and counts points per source so
we can log "deleted 1 orphan with several dozen chunks" rather than
just "1 orphan."

The "is this orphan?" check is one line —
[`kb/reconcile.py:123`](../../src/audrey/kb/reconcile.py#L123):

```python
if Path(source).exists():
    continue
```

Synchronous `Path.exists()` inside an async function. Ruff
flags this (ASYNC240); the comment in the code calls it out
explicitly:

> `Path.exists` is synchronous; call directly. We're not on a
> tight latency budget — this loop runs every `interval_s`, not
> per-request.

That's the principle: synchronous Path methods are fine on
**offline / scheduled** code paths, not on the chat hot path.
The cost of context-switching to a thread for each `exists()` call
across thousands of points would dwarf the cost of just calling
it directly.

What reconcile **excludes** is also worth knowing — from the module
docstring at
[`kb/reconcile.py:15-19`](../../src/audrey/kb/reconcile.py#L15):

> Excludes per-user collections (`kb_user_text_*` / `kb_user_images_*`) on
> purpose — the sqlite uploads index already reconciles those against qdrant
> at startup. Mixing the two paths would let this reconciler delete
> legitimately-uploaded files just because the host can't see the user's
> filename (which it can't — uploads live in qdrant only, no on-disk source).

Uploaded files don't have a `payload.source` pointing at the host
filesystem; they live in `/data/uploads/<sanitized_user>/<file_id>.<ext>`
which the global reconciler shouldn't even think about. The
exclusion is by skipping the per-user collection names entirely —
`reconcile_once` only ever sweeps `kb_text` and `kb_images`.

### 2.7 Concept spotlight — scroll-style pagination

The pattern shared by reconcile and the per-user file lister is
the **scroll** API. Where regular Qdrant queries (text search,
filter-by-id) return ranked or filtered results in one shot,
scroll walks the *whole* collection by handing back a cursor on
each page.

`QdrantKB.scroll_collection` —
[`kb/qdrant.py`](../../src/audrey/kb/qdrant.py) — looks like:

```python
def _scroll_collection_sync(self, collection, page_size):
    out = []
    next_page = None
    while True:
        points, next_page = self._client.scroll(
            collection_name=collection,
            limit=page_size,
            offset=next_page,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            out.append((str(p.id), p.payload or {}))
        if next_page is None:
            break
    return out
```

Three things to keep separate from regular search:

- **No query vector.** Scroll doesn't care about similarity. It
  just walks everything.
- **No ranking.** Results come back in collection order, not score
  order.
- **Whole-collection cost.** Each page is a Qdrant round-trip;
  walking a million-point collection means thousands of round-trips
  and proportional memory. **Not safe on the request hot path.**

`scroll_collection` materializes everything into a list — fine at
Audrey's scale (thousands of points, not millions), but a future
version on a much larger collection would want to switch to a
generator or callback shape.

### 2.8 The lifecycle wrapper pattern

`KBWatcher` and `KBReconciler` share the same shape. Both manage a
long-lived `asyncio.Task` that runs in the background for the life
of the process.

`KBReconciler` ([`kb/reconcile.py:171`](../../src/audrey/kb/reconcile.py#L171)):

```python
class KBReconciler:
    def __init__(self, *, qdrant, interval_s=1800.0, ...):
        ...
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._interval_s <= 0:
            log.info("kb.reconcile: interval_s=0, periodic loop disabled")
            return
        self._task = asyncio.create_task(self._run(), name="kb-reconcile")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run(self) -> None:
        try:
            try:
                await reconcile_once(self._qdrant, ...)
            except Exception as e:
                log.warning("kb.reconcile: startup sweep raised: %s", e)
            while True:
                await asyncio.sleep(self._interval_s)
                try:
                    await reconcile_once(self._qdrant, ...)
                except Exception as e:
                    log.warning("kb.reconcile: sweep raised: %s", e)
        except asyncio.CancelledError:
            return
```

The same pattern in `KBWatcher.start` / `stop`. Internalize:

- **`start()` spawns a task; doesn't block.** Returns immediately
  after the task is scheduled.
- **`stop()` cancels and awaits.** The cancel triggers
  `asyncio.CancelledError` inside the running task; the wrapper
  swallows it and returns cleanly. The `await self._task` ensures
  shutdown waits for the task to actually finish, not just fire
  the cancel and move on.
- **Nested try inside `_run`.** The outer `try/except CancelledError`
  is the clean-shutdown path. The inner `try/except Exception` is
  the "one bad sweep shouldn't kill the loop" path. Without that
  inner guard, a single Qdrant blip would terminate the task and
  no more sweeps would ever run.

The orchestrator's lifespan (Lesson 5) calls `start()` on each at
startup and `stop()` on each during shutdown. That's the entire
integration.

The first reconcile sweep runs immediately at startup, then the task settles
into the configured `interval_s` cadence. That startup pass catches drift that
accumulated while the watcher was off before Audrey serves for long. The admin
endpoint `POST /v1/admin/kb/reconcile` still exists for ad-hoc sweeps between
periodic intervals.

### 2.9 The per-user uploads flow

Pivot now to the third mechanism. Users upload files through the
browser; the upload has to land on disk, index into a per-user
Qdrant collection, and show up in the user's file list without
scrolling Qdrant on every request.

`POST /v1/files`
([`routes/files.py:145`](../../src/audrey/routes/files.py#L145))
runs eight steps in order — each one assumes the previous succeeded
and unwinds in reverse on failure:

1. **Auth via `Depends(require_user)`.** OWUI validates the JWT;
   the route never sees an unauthenticated request. The user's
   email *is* the user id — no spoofable `?user=` param.
2. **Ensure user collections exist**
   ([`kb/user_store.py`](../../src/audrey/kb/user_store.py)).
   Lazily creates `kb_user_text_<sanitized>` and
   `kb_user_images_<sanitized>` plus the payload indexes on first
   upload.
3. **Pre-flight quota check**
   ([`routes/files.py:170`](../../src/audrey/routes/files.py#L170)).
   If the user is already at or over their byte budget, 413
   immediately without touching disk.
4. **Stream to disk** with a per-upload cap
   ([`routes/files.py:115`](../../src/audrey/routes/files.py#L115)).
   1 MB chunks, written to
   `<upload_root>/<sanitized_user>/<file_id>.<ext>`; cap exceeded
   → 413 and unlink.
5. **Sniff mime via libmagic**
   ([`kb/extract.py`](../../src/audrey/kb/extract.py)). Bytes are
   the truth; extension and `Content-Type` are hints. Stops
   `a.png.exe` from sneaking through.
6. **Post-stream quota check** — catches uploads that crossed the
   line during streaming (we only know the actual size after).
7. **Ingest into Qdrant**
   ([`kb/ingest.py`](../../src/audrey/kb/ingest.py)). The Lesson 11
   pipeline, writing to the per-user collection and stamping each
   point's payload with `user` and `file_id`.
8. **Record in sqlite**
   ([`routes/files.py:259`](../../src/audrey/routes/files.py#L259)).
   After Qdrant succeeded; if this write fails, we roll back the
   Qdrant upsert to avoid a phantom file.

The whole point of the ordering — stream first, then validate,
then ingest, then index — is that **each step assumes the previous
steps succeeded.** If any step fails, we unwind in reverse and
return the appropriate error code.

### 2.10 Concept spotlight — the "two stores agree" contract

§1 introduced *why* there are two stores: sqlite for "what files
does this user have?" questions and Qdrant for "what's in those
files?" questions. The cost of that split is the engineering
problem this section unpacks — both stores have to stay in step
without anything coordinating them at the database layer.

The contract from
[`kb/uploads_db.py:8-13`](../../src/audrey/kb/uploads_db.py#L8):

> - Qdrant has the file's content → sqlite has a row for it.
> - Qdrant has nothing for a file_id → sqlite must not either, *unless
>   the row never claimed Qdrant content in the first place*.

That second clause is a patch, and the bruise it covers is worth
looking at. The rule was originally unconditional, and it was
correct for as long as every row in `uploads` described a file that
had already been ingested — "no points" could only mean the row was
a leftover. Video broke that: an uploaded video sits at
`status='pending'` holding bytes on disk with no points anywhere, on
purpose, until the media worker reaches it. The sweep read those as
leftovers and deleted them on every boot. Nothing errored. The row
just wasn't in `GET /v1/files` after a restart, with the mp4 still
on disk and nothing left pointing at it.

The lesson isn't "add a carve-out". It's that an invariant is a
claim about what the rows in a table *mean*, and adding a row type
that means something new silently repeals it. The code enforcing
the invariant doesn't change and doesn't complain — it just starts
being wrong.

The upload flow enforces both invariants in the order shown
above. The delete flow does the same in reverse:

```python
@router.delete("/{file_id}", response_model=DeleteResponse)
async def delete_file(file_id, request, me=Depends(require_user)):
    user = me.email
    ...
    # sqlite first — once the index row is gone, list/quota immediately
    # reflect the delete even if the qdrant calls below take a beat.
    deleted_row = await db.delete_upload(file_id, user=user)
    ...
    await asyncio.gather(
        qdrant.delete_by_file_id(file_id, user=user, collection=text_col),
        qdrant.delete_by_file_id(file_id, user=user, collection=image_col),
    )
    ...
    return DeleteResponse(file_id=file_id, deleted=deleted_row)
```

Notice the ordering: sqlite delete first, then Qdrant deletes,
then the disk unlink. That order is deliberate — the list endpoint
reads sqlite, so the moment the row is gone the file vanishes from
the user's view even if Qdrant is slow. The Qdrant cleanup and
disk unlink are housekeeping that can take their own time.

But what about drift that happens *outside* the request flow?
Manual `qdrant` purges, container restarts mid-upload, a sqlite
file restored from an old backup? That's what
[`reconcile_with_qdrant`](../../src/audrey/kb/uploads_db.py#L957)
is for — a two-direction sweep:
- **Backfill.** Anything in Qdrant that's missing from sqlite gets
  added. So if sqlite was restored from an old backup, any uploads
  Qdrant has since accumulated get picked up.
- **Prune.** Anything in sqlite whose `file_id` no longer exists
  anywhere in the user's Qdrant collections gets dropped. So if a
  Qdrant collection was manually purged, sqlite catches up. Scoped
  to rows that claim Qdrant content — `status='ready'` with at least
  one chunk — for the reason in §2.10. A row that never made the
  claim isn't evidence of drift.

This is called once at startup from the lifespan, **before**
Audrey serves any traffic. The precondition matters — it assumes
no concurrent uploads, which is true during the startup window.

### 2.11 Failure modes

Five things that can go wrong, and what the system does about each:

- **Watcher dies mid-event.** Per-event `try/except` keeps the
  task alive; the next reconcile sweep catches whatever a fully
  dead watcher missed.
- **Reconcile sweep raises on one collection.** The outer
  `try/except` writes the error to `summary.error`; the sister
  collection still gets swept and the loop survives.
- **Upload crash mid-flight.** Orphan bytes left on disk under a
  `file_id` that was never recorded anywhere. Disk-space leak,
  not a correctness issue; cleanup is currently manual.
- **Qdrant succeeds but sqlite write fails.** The rollback in
  [`routes/files.py:upload_file`](../../src/audrey/routes/files.py)
  deletes the just-upserted Qdrant points and 500s. The rollback is
  itself wrapped in a `try/except` so a Qdrant outage during cleanup
  doesn't mask the original sqlite error — the rollback failure is
  logged with the file_id, the original 500 still surfaces to the
  client, and the next boot's `reconcile_with_qdrant` cleans up the
  orphan points.
- **User uploads while reconcile is mid-sweep.** The startup
  reconcile runs before serving traffic, and ad-hoc admin
  reconciles skip per-user collections — so concurrent uploads
  are unaffected.


## 3. Comprehension questions

**1. "The watcher was off for 6 hours during a host reboot. I
deleted 30 files from `/datasets` during that time. What happens
to their vectors?"**

The watcher missed every delete event. The vectors stay in `kb_text` /
`kb_images` until reconcile notices that `Path(source).exists()` returns False
for each of the 30 sources. On current Audrey, the reconciler runs one sweep
immediately at startup, so those stale vectors should be cleaned up soon after
boot; later drift is caught on the normal interval. The admin endpoint
`POST /v1/admin/kb/reconcile` triggers a sweep on demand between intervals.

**2. "I uploaded a file, got a 200 response, then immediately
called `GET /v1/files` and didn't see it. What broke and how would
I diagnose?"**

If the 200 is real and the same authenticated user immediately calls
`GET /v1/files`, the sqlite row should already exist: the upload route records
sqlite before returning success. First confirm the list request is using the
same OWUI identity and that the client is not showing stale UI state. Then
check `audrey-ai` logs for `uploads_db.record failed` or `files: ingest
failed`; those would normally pair with a 500, not a 200. If neither is
present, the upload is in sqlite and the list/read path is the thing to inspect.

**3. "Why does reconcile skip per-user collections, but the
watcher doesn't even know per-user collections exist?"**

They solve different problems. The watcher only watches
`/datasets` paths; per-user uploads never land there, so the
watcher has nothing to notice. Reconcile *could* in principle
walk per-user collections, but their `payload.source` points at
`/data/uploads/...` paths that the host might evict (uploads
directory cleanup, container ephemeral storage) — a
`Path.exists()` check would falsely declare uploaded files as
orphans and delete the user's data. So reconcile explicitly
excludes user collections, and the uploads-side has its own
sqlite-driven reconcile that uses the *sqlite row* as the
authority, not the disk.

**4. "I changed `kb.text_collection` from `kb_text` to
`audrey_text` in `config.yaml`. What still works, what breaks?"**

Ingest works (`ingest_text_file` uses `qdrant.text_collection`).
Search works (`search_text` uses it too). The periodic reconciler and watcher
work because they use the configured collection names. The admin one-shot
reconcile endpoint is a current exception: it calls `reconcile_once(qdrant)`
without passing configured names, so it still sweeps the default collection
names. What also breaks is **anything that was already in the old `kb_text`
collection** — it is still there, but nothing references it, so those vectors
become a separate orphan problem the configured reconcile cannot help with.
You would manually sweep or drop the old collection after the rename.

**5. "A user uploads a 5 MB PDF. Trace what happens to it across
disk, Qdrant, and sqlite."**

- **Disk.** Streamed to
  `<upload_root>/alice_email/<file_id>.pdf` in 1 MB chunks.
  Stays there as the source of truth for the original bytes.
- **Qdrant.** PDF text is extracted (pypdf), chunked (a handful of
  chunks for a few MB of text), each chunk gets embedded (768-d via
  nomic-embed),
  written as a point in `kb_user_text_alice_email` with payload
  including `user`, `file_id`, `filename`, `mime`, `bytes`,
  `uploaded_at`, `chunk_idx`, the chunk text itself.
- **sqlite.** One row in `uploads` keyed by `file_id`:
  `(file_id, user, filename, mime, bytes, kind='text',
  collection=kb_user_text_alice_email, chunks=5, uploaded_at)`.

The list endpoint reads sqlite. The search endpoint reads Qdrant.
The delete endpoint touches all three (sqlite first, then
Qdrant by `(file_id, user)` filter, then disk unlink).


## When you're ready for the next lesson

You've now seen the full lifecycle of a KB entry: how it's
created (the previous lesson), how it stays in step with reality
(this lesson's watcher and reconciler), and how it's managed at
the per-user level (this lesson's uploads flow). The three
mechanisms are independent in code but share the same Qdrant
collections, the same embedders, and the same "two stores agree"
pattern that runs through the uploads side.

The published lessons cover the request hot path end-to-end —
classification, routing, deep mode, tool use, function calling, and
the KB. The next lesson backs up to the very front of the pipeline:
the two earliest graph nodes that prepend context (datetime and
memory recall) before classify reads a single user word. It lives in
[`lesson-13-memory-and-context-injection.md`](lesson-13-memory-and-context-injection.md).
