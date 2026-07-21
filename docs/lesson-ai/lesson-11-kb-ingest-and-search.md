# Lesson 11 — The knowledge base: ingest and search

**Estimated time:** 70-90 minutes if you keep
[`src/audrey/kb/`](../../src/audrey/kb/) and
[`routes/kb.py`](../../src/audrey/routes/kb.py) open.

**Goal:** by the end of this lesson, you can answer
*"a user asks about something covered in our curated docs — what
happened between their question and the snippet the model used?"*

Lesson 9 showed `kb_search` getting dispatched as a tool call. Lesson 10
opened the function-calling protocol that carried the call. This lesson
opens what's behind the tool: how files on disk become searchable
points, how a question becomes a vector, and why the merged hits come
back in the order they do.

There are three ideas to keep separate:

```text
ingest    - file on disk → tokens → embedding → Qdrant point
search    - question → embedding → Qdrant ANN lookup → hits
merge     - global + per-user hits, ranked together
```

## 1. Context

### Why a KB at all?

A language model is a frozen snapshot of whatever its training corpus
contained, compressed into weights. That gives it broad general
knowledge but means it is blind to three categories of information:

1. **Anything it wasn't trained on.** Your private notes, scanned
   manuals, internal documents — none of that was on the public web
   when the model was built, so the weights have no record of it.
2. **Anything below its training threshold.** The training corpus
   covers most topics shallowly; only the most-discussed material
   gets the dense, repeated examples that translate into reliable
   recall. A specialist topic with limited public coverage will be
   thin in the weights even if it was technically present.
3. **Anything that changed after the training cutoff.** Models ship
   with a fixed knowledge horizon. Anything updated, published, or
   discovered since that date does not exist as far as the weights
   are concerned.

A retrieval-augmented system (RAG) gives the model a way to look those up
on demand: when the question is in scope, the model issues a
`kb_search` tool call (Lesson 9); we hand it the matching snippets
and let it answer with them in context.

### Whole-system map

Before the diagram, two pieces of vocabulary the rest of the lesson
leans on:

- An **embedding** is a list of numbers — a few hundred or a few
  thousand floats — that represents a piece of text or an image as
  a single point in a high-dimensional space. The model that
  produces embeddings is trained so that pieces with similar
  meaning land near each other in that space. Two paragraphs about
  the same topic produce embeddings that are close together;
  unrelated paragraphs produce embeddings far apart. "Close" and
  "far" are measured by the angle between the two lists of
  numbers — small angle, similar meaning.
- A **vector database** is a piece of software that stores
  embeddings and answers one specific question really fast: "given
  this new embedding, find me the K stored embeddings that are
  closest to it." A regular database stores rows you look up by
  exact key; a vector database stores points you look up by
  *nearness*. The point of using one is speed — with millions of
  stored vectors, naive comparison would be too slow per request;
  a vector database uses an approximate-nearest-neighbour index
  that returns reliable top-K results in milliseconds.

The vector database Audrey uses is **Qdrant**, an open-source
service that runs alongside Audrey in its own container.

With those defined, the KB has two jobs that happen at two
different moments in time. **Ingest** is the slow, one-time-per-file
work of turning a document into something searchable: read the
bytes, cut them into pieces, run each piece through the embedder,
store the resulting embeddings in Qdrant. **Search** is the fast,
every-request work of turning a user's question into a query
against that store: embed the question, ask Qdrant for the closest
matches, hand the matches back.

The two halves share the *same* embedder and the *same* Qdrant
collections — that's the only way the math works out, because a
query vector can only be compared against vectors produced by the
same model. But they run on completely different schedules:

- **Ingest is offline.** It happens once when a file is added to
  `/datasets`, again when the file changes, and then never until
  something changes again. No human is waiting on it. It's fine for
  ingest to take seconds per file or minutes per directory — it's
  not on anyone's critical path. Think of it like building an
  index for a book: slow up front, then you have the index forever.
- **Search is online.** It happens every time the model dispatches
  a `kb_search` tool call, which is in the middle of a live chat
  with a user staring at a streaming response. It must complete in
  tens or low-hundreds of milliseconds or the user notices the
  pause. Search reuses everything ingest built — without
  pre-computed vectors and a Qdrant index, the per-request work
  would be far too slow.

The diagrams below trace each half top to bottom.

```text
INGEST (offline / batch)
  /datasets/<topic>/file.md
        │
        │ load_text() — extension-dispatched loader
        ▼
        raw text
        │
        │ chunk_text() — token windows with overlap
        ▼
        list[Chunk]
        │
        │ TextEmbedder.embed_many() — nomic-embed-text via Ollama
        ▼
        list[vector(768d)]
        │
        │ build_text_point() + upsert_text() — deterministic IDs
        ▼
        Qdrant collection: kb_text

SEARCH (online / per chat round)
  user question
        │
        │ TextEmbedder.embed_one()
        ▼
        vector(768d)
        │
        │ qdrant.search_text() — top-k cosine ANN
        ▼
        list[KBHit] from kb_text
        │  (+ list[KBHit] from kb_user_text_<sanitized> if user has one)
        │
        │ score-sort, slice to top_k
        ▼
        merged hits → tool result → model
```

Two collections, two embedders, one Qdrant. Images get their own
parallel pipeline using CLIP instead of nomic-embed-text. We'll trace
text first, then image.

### Pieces and where they live

| Concern | Lives in | What it owns |
|---|---|---|
| File loading (per format) | [`kb/chunk.py`](../../src/audrey/kb/chunk.py) | `.md`, `.pdf`, `.docx`, `.html` → string |
| Chunking (tokens → windows) | [`kb/chunk.py`](../../src/audrey/kb/chunk.py) | `chunk_text()` |
| Text embeddings | [`kb/embed.py`](../../src/audrey/kb/embed.py) | `TextEmbedder` → Ollama `/api/embed` |
| Image embeddings | [`kb/embed.py`](../../src/audrey/kb/embed.py) | `ImageEmbedder` → CLIP ViT-B-32 |
| Qdrant wrapper | [`kb/qdrant.py`](../../src/audrey/kb/qdrant.py) | `QdrantKB`, point construction, search |
| Upload-side mime guard | [`kb/extract.py`](../../src/audrey/kb/extract.py) | libmagic sniffing + extract |
| Crawl + ingest orchestration | [`kb/ingest.py`](../../src/audrey/kb/ingest.py) | `ingest_path`, `ingest_text_file` |
| Per-user collection naming | [`kb/user_store.py`](../../src/audrey/kb/user_store.py) | `kb_user_text_<sanitized>` |
| HTTP query endpoints | [`routes/kb.py`](../../src/audrey/routes/kb.py) | `/v1/kb/query`, `/v1/kb/query/image` |
| Tool wrapper Audrey discovers | [`tools-server/app.py`](../../tools-server/app.py) | `kb_search`, `kb_image_search` |

The CLI ([`kb/cli.py`](../../src/audrey/kb/cli.py)) and the
watcher/reconcile loop ([`kb/watcher.py`](../../src/audrey/kb/watcher.py),
[`kb/reconcile.py`](../../src/audrey/kb/reconcile.py)) sit on top of
this pipeline but are deferred — the next lesson covers them.


## 2. Read-along

### 2.1 The two collections (and why two)

[`kb/qdrant.py:1-24`](../../src/audrey/kb/qdrant.py#L1) opens with the
shape of the world:

```python
"""
Two collections:
  - `kb_text`   : 768-d (nomic-embed-text via ollama /api/embed)
  - `kb_images` : 512-d (CLIP ViT-B-32 via sentence-transformers)
"""
```

Two collections because two embedders. Text vectors are 768
dimensions; image vectors are 512. Qdrant requires every vector in a
collection to share a dimension, so the dim difference alone forces
two indexes. They also use different similarity neighbourhoods — words
that mean similar things cluster in the text space; pictures that
look like one another (and, because CLIP shares a text/image space,
text descriptions of pictures) cluster in the image space.

**Concept spotlight — how "close" gets measured.**
§1 introduced embeddings as points in a high-dimensional space and
"closeness" as the angle between them. The specific number we ask
Qdrant for is **cosine similarity** — literally the cosine of that
angle. Cosine 1.0 means "same direction" (very similar); 0.0 means
"perpendicular" (unrelated); -1.0 means "opposite" (the embedder
thinks they contradict). Every score that comes back from a Qdrant
search is a cosine value in `[-1.0, 1.0]`, and Audrey just sorts
them descending. Crucially, Qdrant has no idea what the vectors
*mean* — it only knows how to find the K closest to a query
vector, fast.

[`kb/qdrant.py:41-42`](../../src/audrey/kb/qdrant.py#L41) pins the
two dims as module constants:

```python
TEXT_DIM = 768
IMAGE_DIM = 512
```

These are reused by `user_store.ensure_user_collections` when
creating per-user collections so the per-user side matches the global
side and `_search_text_merged` can merge their hits by raw score.

One bridge before we dive in: this lesson assumes the model has
*already* decided to issue a `kb_search` tool call. That decision —
classifier picks "knowledge" intent, router selects a tool-capable
model, ReAct loop dispatches the call — lives upstream in Lesson 7
(classification and routing) and Lesson 9 (the ReAct loop itself).
Here we start the moment Audrey receives the `kb_search` dispatch and
ends when we hand it a list of hits.

### 2.2 A file becomes points

The unit of storage in Qdrant is a **point** — an `(id, vector,
payload)` triple. For text, one point per chunk. For images, one
point per image. The journey from a Markdown file on disk to a row
of points is `ingest_text_file` —
[`kb/ingest.py:103-131`](../../src/audrey/kb/ingest.py#L103):

```python
async def ingest_text_file(path, *, qdrant, embedder, chunk_tokens, overlap_tokens):
    raw = load_text(path)
    if not raw:
        return 0
    chunks = chunk_text(raw, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
    if not chunks:
        return 0
    source = normalize_source(path)
    mtime = path.stat().st_mtime
    vectors = await embedder.embed_many([c.text for c in chunks])
    await qdrant.delete_by_source(source, collection=qdrant.text_collection)
    points = [
        build_text_point(source=source, chunk_idx=c.idx, text=c.text,
                         vector=v, mtime=mtime)
        for c, v in zip(chunks, vectors, strict=True)
    ]
    await qdrant.upsert_text(points)
    return len(points)
```

Five steps. Read top to bottom:

1. **Load.** `load_text` picks a parser from the extension —
   [`kb/chunk.py:50-65`](../../src/audrey/kb/chunk.py#L50). `.md`,
   `.txt`, `.rst`, `.log`, `.csv` go straight through `read_text()`;
   `.pdf` goes through pypdf; `.docx` through python-docx; `.html`
   through BeautifulSoup. Unsupported suffixes return `None` and the
   crawl skips the file.
2. **Chunk.** Token windows with overlap. The next subsection unpacks
   the math.
3. **Embed.** Batched calls to Ollama (default batch size 64). One
   HTTP round-trip per batch, vectors come back in the same order as
   the inputs.
4. **Delete-before-upsert.** Crucial for shrinking files (someone
   trims a stale section, a script regenerates a doc with less
   output, a user re-uploads a smaller v2). If yesterday's ingest
   produced a dozen chunks and today the file shrank to a few, we
   want the now-orphaned tail gone. `delete_by_source` clears every
   point whose payload `source` matches this file before the upsert
   writes the current set.
5. **Upsert.** Build a `PointStruct` per chunk with a deterministic
   ID and write the batch.

**Concept spotlight — deterministic IDs.**
[`kb/qdrant.py:56-57`](../../src/audrey/kb/qdrant.py#L56):

```python
def point_id(*, source: str, kind: str, idx: int) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{source}:{kind}:{idx}"))
```

UUIDv5 is a hash-based UUID: the same input string always produces
the same UUID. So `point_id(source="/datasets/topic/file.md",
kind="text", idx=3)` is *the same UUID every time we run ingest*.
Qdrant's upsert is keyed by ID — same ID means "replace this point's
vector and payload", not "create a new one." That's what makes
re-running ingest idempotent: unchanged chunks overwrite themselves
with identical vectors; changed chunks (e.g. a paragraph edit) get a
new vector at the same ID. Combined with the delete-before-upsert
step, the index stays in lockstep with what's on disk.

### 2.3 Concept spotlight — chunking

The model's context window has a budget. So does the relevance
signal of a long document — a 50-page reference book mentions
"granite" in a dozen sections, but only one is about identification.
Chunking is how we slice documents into independently-searchable
units.

[`kb/chunk.py:98-122`](../../src/audrey/kb/chunk.py#L98):

```python
def chunk_text(text, *, chunk_tokens=1000, overlap_tokens=100):
    cleaned = text.strip()
    if not cleaned:
        return []
    enc = _encoder()
    tokens = enc.encode(cleaned)
    if len(tokens) <= chunk_tokens:
        return [Chunk(text=cleaned, idx=0)]
    if overlap_tokens >= chunk_tokens:
        overlap_tokens = chunk_tokens // 5
    stride = chunk_tokens - overlap_tokens
    out: list[Chunk] = []
    for i, start in enumerate(range(0, len(tokens), stride)):
        end = min(start + chunk_tokens, len(tokens))
        piece = enc.decode(tokens[start:end]).strip()
        if piece:
            out.append(Chunk(text=piece, idx=i))
        if end >= len(tokens):
            break
    return out
```

Three things to internalize.

**Tokens, not characters.** The tokenizer is `cl100k_base`, the same
one Lesson 7's complexity gate uses. We chunk in token space because
that's the dimension the embedder cares about — `nomic-embed-text`
has an ~8K-token input cap, so 1000-token windows leave it plenty of
headroom. Character counts wouldn't tell us anything useful here.

**Stride and overlap.** With `chunk_tokens=1000` and
`overlap_tokens=100`, the stride is 900. Each window starts 900
tokens after the previous; the last 100 tokens of window N are also
the first 100 tokens of window N+1. Why? Because a relevant sentence
might straddle the boundary. Without overlap, if your answer is "the
last paragraph of page 4 plus the first paragraph of page 5," each
half lands in a different chunk and neither chunk alone is a great
match for the query. Overlap means *both* chunks contain the bridging
sentences, so both score reasonably and at least one will likely
clear top-K.

**Short docs are a single chunk.** The
`if len(tokens) <= chunk_tokens: return [Chunk(...)]` short-circuit
(line 118) exists so a half-page note isn't pointlessly windowed.

**One consequence worth internalizing: paragraphs don't always land
in exactly one location.** Tokens inside an overlap region (e.g.
tokens 900-999 with the default settings) appear in *two* chunks,
so they're embedded twice and stored as two separate Qdrant points.
A sentence that sits squarely in the middle of a chunk is searchable
via one point; a sentence near a boundary is searchable via two and
may show up twice in the top-K results. That's the cost overlap buys
you for the straddling-sentence fix.

The tail of the loop has one extra guard in the current code: if the final
window would add very little new content beyond the previous chunk, Audrey skips
that near-duplicate tail chunk. That prevents a wasted embed call and Qdrant
point while preserving the content that was already covered by the overlap.

### 2.4 The embedder, in code

Now the actual class that turns text into vectors —
[`kb/embed.py:107`](../../src/audrey/kb/embed.py#L107):

```python
@dataclass(slots=True)
class TextEmbedder:
    ollama: OllamaClient
    model: str = "nomic-embed-text"
    timeout_s: float = 60.0
    batch_size: int = 64

    async def embed_one(self, text: str) -> list[float]:
        out = await self.embed_many([text])
        return out[0]

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        ...
```

`TextEmbedder` is a thin wrapper around the Ollama client. Two public
methods:

- `embed_many(texts)` — send a batch of strings to Ollama in one
  HTTP call and get back a list of vectors. This is what ingest
  uses, because ingest typically has dozens or hundreds of chunks
  to embed at once.
- `embed_one(text)` — embed a single string. This is what search
  uses, because a search has exactly one input: the user's question.

`embed_one` is just a one-line wrapper that calls `embed_many` with
a single-element list. So all roads lead to `embed_many`, which
batches its input (default 64 strings per Ollama call) so a 500-chunk
file produces about 8 HTTP requests instead of 500.

**One small post-processing step: normalization.** After Ollama
returns each vector, Audrey scales it down so its total length is 1.
This is a math trick that makes cosine-similarity comparisons faster
and lets the same stored vectors work if we ever switch Qdrant to a
different distance metric. The details are in `_normalize` at
[`kb/embed.py:163`](../../src/audrey/kb/embed.py#L163); it's a one-
time cost at ingest and search, and you can largely treat it as a
background detail.

The interesting part is the constraints that come out of all this:

- **A collection is locked to a specific vector size.** When we
  create `kb_text`, we tell Qdrant "every vector here will be 768
  numbers long." If you swap to an embedder that produces 1024-long
  vectors, the next ingest will fail with a dim-mismatch error.
  You can't migrate in place — you drop the collection and
  rebuild.
- **The embedder used for search must match the embedder used for
  ingest.** Two different embedders produce vectors in entirely
  different mathematical spaces. Comparing a vector from one space
  against vectors from another gives numbers back, but they don't
  mean anything — like comparing temperatures in Celsius to
  temperatures in pounds.
- **Within one collection, every vector came from the same
  embedder.** This is what lets §2.5's global+user merge sort hits
  by raw score: the scores are directly comparable because the
  vectors they came from live in the same space.

### 2.5 The query path

`/v1/kb/query` is short —
[`routes/kb.py:99-117`](../../src/audrey/routes/kb.py#L99):

```python
@router.post("/query", response_model=QueryResponse)
async def kb_query(req: TextQuery, request: Request) -> QueryResponse:
    qdrant = getattr(request.app.state, "qdrant", None)
    embedder = getattr(request.app.state, "text_embedder", None)
    if qdrant is None or embedder is None:
        raise HTTPException(status_code=503, detail="KB is not initialized")
    t0 = time.perf_counter()
    vec = await embedder.embed_one(req.query)
    hits, had_user = await _search_text_merged(qdrant, vec, top_k=req.top_k, user=req.user)
    elapsed = time.perf_counter() - t0
    kb_search_seconds.labels(kind="text", had_user_collection=str(had_user).lower()).observe(elapsed)
    kb_search_hits.labels(kind="text").observe(len(hits))
    return QueryResponse(query=req.query, results=[...])
```

Four steps. Embed the query → search → record metrics → return.

The pieces of FastAPI to notice:

- **`request.app.state`** is how lifespan-created globals reach
  handlers. `qdrant` and `text_embedder` were attached at startup
  (Lesson 5).
- **`response_model=QueryResponse`** in the decorator makes FastAPI
  validate the return value and auto-document the response in
  `/openapi.json`. That's the same schema custom-tools reads to
  build the `kb_search` tool spec for the model.

The merge logic is in `_search_text_merged` —
[`routes/kb.py:119-142`](../../src/audrey/routes/kb.py#L119):

```python
async def _search_text_merged(qdrant, vec, *, top_k, user):
    """Search global kb_text and, if the user has one, their kb_user_text_* too. Merge by score.
    ...
    Score-merge precondition: both collections must use the same embedder
    (currently 768-d nomic-embed-text, cosine) so the raw scores are comparable.
    """
    coros = [qdrant.search_text(vec, top_k=top_k)]
    had_user = False
    if user:
        user_col = user_text_collection(user)
        if await qdrant.collection_exists(user_col):
            coros.append(qdrant.search_text(vec, top_k=top_k, collection=user_col))
            had_user = True
    results = await asyncio.gather(*coros)
    merged: list[KBHit] = [h for batch in results for h in batch]
    merged.sort(key=lambda h: h.score, reverse=True)
    return merged[:top_k], had_user
```

When a user supplies their `user` id and has a personal collection,
we fire two Qdrant searches in parallel via `asyncio.gather`, pull
top-K from each, then re-sort by raw cosine score and slice. The
caller asks for K=5; we may have up to 10 hits to choose from.

### 2.6 Concept spotlight — global + user merge

The merge works because the precondition holds: both collections
share an embedder. Every score in either result list is "cosine
distance from the same query vector, against vectors produced by the
same model." So sorting them together gives a global ranking that's
internally consistent.

If we ever spun up a per-user collection on a different embedder, the
scores would look comparable on paper (both are floats in the same
range) but mean different things — the merge would produce arbitrary
ordering. The docstring at
[`routes/kb.py:129-130`](../../src/audrey/routes/kb.py#L128) pins
that contract:

> If a per-user collection ever ships with a different model, switch
> to reciprocal-rank-fusion rather than sorting by raw score.

RRF — reciprocal rank fusion — combines ranked lists by rank position
instead of raw scores. Slower and less precise when scores *are*
comparable, but the right tool when they aren't.

A more subtle consequence: a user with a small but very on-topic
personal collection can fill all top-K slots, pushing the global
hits off the result. That's usually the right outcome (their private
notes about *this* topic *should* beat a general reference) but
it's worth knowing the merge is winner-take-most on close matches,
not a strict round-robin.

### 2.7 Image search has three input modes

The image side is the most asymmetric. There's one model (CLIP) but
three ways to give it an input — text, URL, or base64 bytes — because
CLIP's text encoder and image encoder share an embedding space. The
text "a black labrador" and an actual picture of a black lab land
within a few cosine degrees of each other.

`/v1/kb/query/image` (handler at
[`routes/kb.py:164`](../../src/audrey/routes/kb.py#L164)) picks which
encoder to call based on which field of the request body was supplied
([`routes/kb.py:176-181`](../../src/audrey/routes/kb.py#L176)):

```python
if req.image_url:
    vec = await embedder.embed_url(req.image_url)
elif req.image_b64:
    vec = await embedder.embed_b64(req.image_b64)
else:
    vec = await embedder.embed_text(req.query or "")
```

All three return a 512-D vector in the same space; the search after
that is identical to the text path.

**Concept spotlight — shared embedding space.**
CLIP was trained on `(image, caption)` pairs with a contrastive loss:
pull image and caption vectors together, push unmatched pairs apart.
After enough training, both encoders settle into one shared 512-d
space where "concept-equivalent" inputs cluster regardless of input
modality. That's the trick that makes "search images by text
description" work without a separate text-to-image model:

```text
"black labrador" → CLIP text encoder    → 512-d vector
   labrador.jpg → CLIP image encoder    → 512-d vector
                                         these end up close.
```

The image URL path (`embed_url`) is also where the SSRF guards
matter. Audrey is a network-reachable service; an unauthenticated
caller (or an authenticated one trying to probe internal addresses)
could feed `embed_url` a `http://qdrant:6333/` to see what comes
back. `_validate_image_url`
([`kb/embed.py:86-99`](../../src/audrey/kb/embed.py#L86)) rejects
non-HTTPS schemes and any hostname that resolves to a private,
loopback, link-local, or otherwise non-public IP. The fetch then
runs with `follow_redirects=False` and a streaming byte cap so a
permitted public host can't 302 the connection into something
internal or OOM us with a giant payload. SSRF defense is a sidebar
to this lesson; the rules are spelled out in
[`kb/embed.py:40-99`](../../src/audrey/kb/embed.py#L40) if you want
to dig in. A future lesson on Audrey's routes-and-security surface
will expound on this — the same defense pattern applies to any
endpoint that accepts a user-supplied URL.

### 2.8 How tools-server reaches back into Audrey

The model never talks to Audrey's KB directly. It dispatches
`kb_search` (or `kb_image_search`) as a tool call (Lesson 9), which
hits the custom-tools server, which then HTTP-proxies into Audrey's
`/v1/kb/query`. Closing the loop:
[`tools-server/app.py:406`](../../tools-server/app.py#L406):

```python
async def kb_search(req: KBSearchRequest) -> KBSearchResponse:
    client: httpx.AsyncClient = app.state.audrey
    payload: dict[str, Any] = {"query": req.query, "top_k": req.top_k}
    if req.user:
        payload["user"] = req.user
    try:
        r = await client.post("/v1/kb/query", json=payload)
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Audrey KB unreachable: {e}",
        ) from e
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    body = r.json()
    return KBSearchResponse(query=req.query, results=body.get("results", []))
```

A few things click into place:

- **Why two services?** The same isolation that lets us swap out the
  Brave web-search backend without touching Audrey applies here.
  custom-tools is the contract; the routes behind it can evolve.
- **The `user` field is end-to-end.** Audrey overrides any model-
  supplied `user` with the authenticated session user before
  dispatching (Lesson 9 §2.7 covered this), so by the time the model
  thinks it's calling `kb_search(user="alice@example.com")`, the
  string actually flowing on the wire is whatever the OWUI session
  authenticated as.
- **Where does the dispatch loop see `kb_search` exist?**
  custom-tools' `/openapi.json` is generated by FastAPI from this
  function signature (Lesson 9 §2.1, Lesson 10 §2.2). Audrey reads
  that at startup, turns each tagged-`tools` operation into a
  `ToolSpec`, and includes them in the `tools` array of every chat
  request (Lesson 10 §2.1).

### 2.9 Failure modes

The KB introduces five failure modes the chat pipeline has to
handle. None of them are fatal; all of them show up.

**1. Qdrant is down.**
`search_text` raises a connection error from inside `asyncio.to_thread`.
`/v1/kb/query` doesn't catch it; FastAPI returns a 500 to custom-tools,
which surfaces it as an HTTP error to the model. The model usually
apologises and either tries again next round (rare, model-dependent)
or answers without the KB. Lesson 9 §2.6 covered how `dispatch_one`
turns the upstream error into a `role: "tool"` message with `is_error`
so the model sees what failed.

**2. Embedder (Ollama) is down.**
Same shape, different line. `embed_one` raises an `OllamaError`. The
query path 500s; downstream the model treats it as a tool failure.
Note this affects ingest too — the CLI errors out, the watcher logs
the failed file and moves on (Lesson 12 covers the watcher's error
behavior).

**3. Dim mismatch on upsert.**
Symptom: ingest fails at `qdrant.upsert_text(...)` with a clear
"vector dimension mismatch" from qdrant-client. Cause: someone
changed `kb.text_embedder` in `config.yaml` to a different-dim
model without dropping `kb_text` first. Recovery: drop and rebuild
the collection, then re-ingest.

**4. Per-user collection drift.**
A user uploads a file that gets indexed into
`kb_user_text_alice_example_com`, then deletes their account / email
changes / etc. The collection stays in Qdrant; nothing points at it
from Audrey. Not a bug — the uploads-side reconciliation job covered
in Lesson 12 is what cleans these up.

**5. SSRF-rejected image URL.**
`_validate_image_url` raises `ValueError`; the route catches it
([`routes/kb.py:182`](../../src/audrey/routes/kb.py#L182)) and 422s
with the reason. From the model's side this looks like any tool
failure: it gets the rejection message in the `role: "tool"`
content, and the right thing is to ask the user for a different URL
rather than retry. Redirect responses get caught specially
([`kb/embed.py:197-206`](../../src/audrey/kb/embed.py#L197)) and
name the redirect target so the user can resupply the final URL.

There is also a sixth mode worth knowing: **the KB returns hits but
they're irrelevant.** Cosine ANN always returns *something* — the
top-K closest vectors, not "the K vectors above a relevance
threshold." With no matches at all, you still get five hits scoring
0.2, 0.18, 0.17… Audrey doesn't filter by score; the model sees
them and is expected to judge relevance from the text alone. That's
where description quality matters: the model has to read the snippet
and decide it's actually about the user's question.


## 3. Comprehension questions

These are operational scenarios. Try to answer by yourself
first, then check against the code.

**1. "I re-ingested a Markdown file after editing it down. The file
used to chunk into a dozen pieces; now it chunks into a few. What
happens to the now-surplus chunks in Qdrant?"**

`ingest_text_file` runs `delete_by_source(source,
collection=qdrant.text_collection)` before the upsert
([`kb/ingest.py:122`](../../src/audrey/kb/ingest.py#L122)). That
clears every point in `kb_text` whose payload `source` matches the
file's absolute path — all of the old chunks. The upsert then writes
the current ones with deterministic IDs. Net effect: the old points
are gone, the current points present, no stale orphans. The
deterministic-ID + delete-before-upsert combo is what makes ingest
idempotent even when chunk
counts shift.

**2. "We want to switch from `nomic-embed-text` (768-d) to a 1024-d
model. What breaks, where do you find out, and what do you do?"**

The `kb_text` collection was created with `size=768` — Qdrant
enforces this at upsert time. The next ingest run will fail at
`qdrant.upsert_text(...)` with a vector-dimension-mismatch error from
qdrant-client. You'll see it as a per-file error logged by
`ingest_path`'s outer `try/except` and surfaced in
`stats.errors`. Fix: drop the `kb_text` collection (Qdrant CLI or
admin UI), restart Audrey so `ensure_collections` recreates it with
the new dim, then run a full ingest. Same drill for `kb_images` /
CLIP. The `TEXT_DIM = 768` constant in
[`kb/qdrant.py:41`](../../src/audrey/kb/qdrant.py#L41) and the
embedder config in `config.yaml` both have to move together; if
they drift, you'll see it on the next ingest.

**3. "A user has a personal KB collection with a few uploaded notes
about a specific hike. They ask about the hike. The merged result
has zero global hits even though our `/datasets/hiking/` has plenty
of material. Why?"**

`_search_text_merged` pulls top-K from each collection then sorts by
raw cosine score and slices to `top_k`
([`routes/kb.py:138-142`](../../src/audrey/routes/kb.py#L138)). If
the user's personal notes are closer matches to the query — likely
when the notes use their exact phrasing — they can sweep all five
slots. The global hits *are* in the candidate list; they just rank
below all the user's hits. That's usually the intended outcome
(private docs about the user's specific hike outrank general
references) but if you wanted a guarantee that each side contributes
at least one hit, you'd need to switch the merge to round-robin or
RRF.

**4. "Image search on a plain English text query returns hits with
cosine scores around 0.2 — much lower than text search typically
gets. Is the KB broken?"**

Probably not. CLIP's text-to-image cosine scores live in a narrower
band than text-to-text scores from `nomic-embed-text` — different
training distributions, different score densities, different
"close enough" thresholds. AGENTS.md captures the principle ("CLIP
text-to-image scores can look low and still be correct"). To
verify: pull one of the top hits and look at it. If it's actually
relevant, the system is working as designed — the bare cosine value
is just a different scale. If it's not, then you have a real
relevance bug to chase (or a collection that hasn't been ingested
with the images you expected).

**5. "Qdrant restarts in the middle of a chat. The next user message
triggers a `kb_search` tool call. Walk what happens — protocol view
first, then code path."**

Protocol view: the model dispatches the call (Lesson 10 §2.5);
custom-tools' `kb_search` proxies into Audrey's `/v1/kb/query`;
Audrey's embedder runs (Ollama is up); the Qdrant search raises
because the qdrant-client connection is broken; `/v1/kb/query`
returns 500; custom-tools surfaces it as HTTP 500; Audrey's
`dispatch_one` (Lesson 9 §2.6) packages the failure as a
`role: "tool"` message with `is_error=True`; the model receives the
error result and the ReAct loop continues. On the *next* round the
model usually does one of two things: try `kb_search` again (and
succeed if Qdrant recovered) or fall back to answering from training.
In code: the failure surfaces in `_search` at
[`kb/qdrant.py:_search`](../../src/audrey/kb/qdrant.py) as a
qdrant-client exception, propagates out of
`asyncio.to_thread`, and is not caught by the route. The fact that
the ReAct loop has `max_rounds` (Lesson 9 §2.11) means a flapping
Qdrant doesn't trap the conversation forever — at most three rounds
of retries, then the model is forced to prose.


## When you're ready for the next lesson

You've now walked the full KB pipeline once: file on disk →
chunks → vectors → Qdrant points → search → merged hits → tool
result → model. The mental model from this lesson is enough to
debug the chat-time half of any KB question.

The next lesson opens the operational half — the watcher that
re-ingests files when they change, the periodic reconcile loop that
catches drift the watcher misses, and the per-user upload flow with
its SQLite-backed metadata store. The pipeline you just learned
runs underneath all of those.
