# Lesson 4 — The platform and the request lifecycle

**Estimated time:** 70-90 minutes (read 40, walk through the code 30-50).

**Goal:** by the end of this lesson, you can answer the question *"You type a
question into Audrey and press Send. What runs where, in what order, until the
answer has finished streaming, and how does a bot calling the API reach the
same pipeline?"* You should be able to answer at a level where every term has
a referent in your head, even if you don't yet know how each piece is
implemented.

We're not going deep on any single component today. The point is to build a
**map**: which containers exist, which doors requests come through, which code
drives the pipeline, and where each kind of data lives. Later lessons fill in
each region.


## 1. Context

### 1.1 Why Audrey has its own front end

A model server can sit behind a chat app someone else wrote and answer
whatever that app sends. Audrey has its own browser app instead, because
three promises need the server to own facts that a third-party app would
keep for itself:

- **An answer survives a dropped connection.** The work belongs to a **run**
  that Audrey owns, not to the connection that asked for it, so a refresh or
  a closed laptop doesn't lose the answer.
- **The conversation is the one Audrey stored.** Conversations and their
  messages are rows in Audrey's own database, and each new turn is answered
  against that stored history, never against a transcript the browser sends.
- **A person's data stays theirs.** Audrey keeps its own accounts, so
  conversations, memories and files belong to a fixed account rather than to
  an email address that might change. People in a browser prove who they are
  through Cloudflare Access; bots and scripts use a personal access token.

The browser app lives in this repository (the `web/` directory) and runs in
its own small container. It displays conversations; it never owns them.

### 1.2 The whole-system map

Here's everything that runs, and who talks to whom:

```text
 YOU, IN A BROWSER
   │
   ▼
 Cloudflare Access ..... checks your sign-in; adds a signed identity header
   │
   ▼
 cloudflared ........... the tunnel from Cloudflare into the home server
   │
   ▼
 audrey-ui (nginx) ..... serves the app's files; forwards /api and /v1
   │
   ▼
 audrey (FastAPI) ...... identity, conversations and runs, the pipeline, files
   ▲   │
   │   ├──► Ollama ............ local and cloud models
   │   ├──► Qdrant ............ vector search
   │   ├──► custom-tools ...... the tool sidecar: web search, memory, chat history
   │   └──► /data ............. SQLite databases and uploaded files
   │
   └──── a bot or script on the home network: /v1 with a personal token

 media-worker  ──► audrey ..... claims video, audio and scanned-PDF jobs
 media-fetcher ──► audrey ..... claims URL download jobs
```

Five of these are Audrey's own containers, defined in
[`compose.yaml`](../../compose.yaml): `audrey`, `audrey-ui`, `custom-tools`,
`media-worker` and `media-fetcher`. Ollama and Qdrant are managed separately on
the same server, and cloudflared runs on the server itself.

| Container | Its job | Why it's separate |
|---|---|---|
| `audrey` | The FastAPI application: identity, conversations and runs, the pipeline, files, the knowledge base | It's the core; everything else is a client or a helper |
| `audrey-ui` | nginx, serving the browser app's built files and forwarding API calls | Static files and the public edge don't need Python |
| `custom-tools` | The tool sidecar: web search, opening web pages, memory, chat history, file listings | Tools touch the internet and parse pages someone else wrote; a crash there can't take the core down |
| `media-worker` | Turns video, audio and scanned PDFs into text | Minutes of CPU work on untrusted files, in a box with no route out |
| `media-fetcher` | Downloads videos from URLs | It needs the internet, so it gets internet access and nothing else |

Hold one sentence for the whole picture:

> **The browser renders; Audrey decides; the sidecars do risky work in a box.**

The browser never decides what a conversation contains or who may see a file.
Audrey makes every decision and keeps every record. The sidecars do the jobs
most likely to go wrong: talking to the internet, parsing files someone else
made, burning CPU for minutes at a time. Each runs where its failure is
contained. Lesson 5 opens `compose.yaml` and shows how each container's network
and mounts enforce that.


## 2. Read-along

Open these files in your editor as we go. Don't try to understand every line
yet; just *locate* the things we point at.

- [`web/docker/default.conf.template`](../../web/docker/default.conf.template):
  the nginx configuration inside `audrey-ui`.
- [`src/audrey/routes/app/runs.py`](../../src/audrey/routes/app/runs.py): the
  browser's entry point, and runs.
- [`src/audrey/routes/openai/routes.py`](../../src/audrey/routes/openai/routes.py):
  the OpenAI-compatible entry point.
- [`src/audrey/routes/openai/pipeline.py`](../../src/audrey/routes/openai/pipeline.py):
  the code that drives the pipeline.
- [`src/audrey/pipeline/graph.py`](../../src/audrey/pipeline/graph.py): the
  pipeline as a LangGraph graph.

### 2.1 One origin for the browser

Every request from your browser goes to one place: `audrey-ui`, an nginx web
server. nginx is configured as a list of `location` blocks, each saying what to
do with requests whose path matches. Open
[`default.conf.template`](../../web/docker/default.conf.template). Three blocks
matter:

- `location /`, at the bottom, serves the browser app itself: the HTML,
  JavaScript and CSS produced when `web/` is built. A path that isn't a real
  file gets `index.html`, so the app can handle addresses of its own.
- `location ^~ /assets/` serves the build's asset files, and answers 404 for a
  missing one instead of the app page.
- The block at [`default.conf.template:62`](../../web/docker/default.conf.template#L62)
  forwards anything under `/api` or `/v1` to Audrey:

```nginx
location ~ ^/(api|v1)(/|$) {
    proxy_http_version 1.1;
    proxy_set_header Connection "";
    proxy_set_header Host $http_host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Host $http_host;
    proxy_set_header X-Forwarded-Proto $audrey_forwarded_proto;
    proxy_set_header Cf-Access-Jwt-Assertion $http_cf_access_jwt_assertion;
    proxy_buffering off;
    proxy_cache off;
    proxy_request_buffering off;
    proxy_read_timeout 3600s;
    proxy_send_timeout 3600s;
    proxy_pass $audrey_upstream;
}
```

> **Concept spotlight — origin, and why having one matters.** An **origin** is
> the scheme, host name and port of a URL: `https://audrey.example.com` is one
> origin, and the same host on another port is a different one. Browsers
> enforce their security rules per origin. A page may freely call its own
> origin; calling a different one needs that server's explicit permission (a
> mechanism called CORS), and cookies don't travel along unless both sides opt
> in. Serving the app and the API from one origin removes all of that. The app
> calls `/api/agent` as a plain relative path, the browser attaches the sign-in
> cookie automatically, and the security policy nginx sends with every
> response can be strict: `connect-src 'self'` in the `Content-Security-Policy`
> header tells the browser this page may only talk to its own origin, which
> makes it much harder for injected script to send data anywhere else.
> **One origin: to the browser, the app and its API are one site.**

The forwarding block does three more things worth knowing.

**It passes the identity header through untouched.** The
`Cf-Access-Jwt-Assertion` line copies the header Cloudflare Access added. Its
value is a **JWT** (JSON Web Token): a small JSON document, here holding your
email and an account id from Cloudflare, plus a cryptographic signature. nginx
doesn't check it; Audrey does, using Cloudflare's published public keys. A
signature can only be made with Cloudflare's private key, so checking it in
Audrey means a request that reaches Audrey some other way can't pretend to be
a signed-in browser.

**It lets streams flow.** `proxy_buffering off` makes nginx relay each piece of
Audrey's response the moment it arrives, instead of collecting the whole
response first. A streamed answer depends on that: each event should reach
your screen when Audrey emits it. `proxy_request_buffering off` does the same
in the other direction, so a large upload streams to Audrey instead of being
saved in full inside the UI container first. `proxy_read_timeout 3600s` lets a
long deep answer hold the connection open for up to an hour.

**It looks Audrey's address up at request time.** `proxy_pass` names a
variable, set near the top of the server block at
[`default.conf.template:37`](../../web/docker/default.conf.template#L37):

```nginx
resolver 127.0.0.11 valid=10s ipv6=off;
set $audrey_upstream ${AUDREY_UPSTREAM};
```

Docker gives each container an IP address when the container is created, and
recreating `audrey` (which every deploy does) can change it. nginx normally
resolves a host name once, at startup, and keeps that address forever, so a
long-running nginx would keep sending traffic to a container that no longer
exists. A variable in `proxy_pass` forces a fresh lookup through Docker's
built-in DNS server (`127.0.0.11`), cached for ten seconds. `${AUDREY_UPSTREAM}`
itself is filled in when the container starts (the nginx image substitutes
environment variables into its `*.template` files), and
[`compose.yaml:122`](../../compose.yaml#L122) sets it to `http://audrey:8000`.

One more boundary sits outside nginx: `compose.yaml` publishes the UI's port
only on the server's own loopback address, `127.0.0.1`
([`compose.yaml:119`](../../compose.yaml#L119)). Only programs on the server
itself can reach it, which in practice means cloudflared. Nobody on the home
network can get around Cloudflare Access by browsing to the server directly.

### 2.2 Two front doors, one pipeline

Audrey has two families of routes, and the path prefix tells you which one
you're in. The native application's routes are grouped under `/api` at
[`routes/app/__init__.py:14`](../../src/audrey/routes/app/__init__.py#L14):

```python
router = APIRouter(prefix="/api")
```

The OpenAI-compatible routes are grouped under `/v1` at
[`routes/openai/routes.py:69`](../../src/audrey/routes/openai/routes.py#L69):

```python
router = APIRouter(prefix="/v1", tags=["openai"])
```

| | The native application API | The OpenAI-compatible API |
|---|---|---|
| Prefix | `/api` | `/v1` |
| Usually called by | the browser app | bots, scripts and evaluation runs |
| Usual route in | Cloudflare, the tunnel, `audrey-ui` | straight to `audrey`'s published port on the home network |
| Proves identity with | the Cloudflare Access header | `Authorization: Bearer aud_pat_…`, a personal access token |
| Conversation history | kept by Audrey; the client sends only the new message | kept by the client, which sends the whole conversation every time |
| Streams back | AG-UI events | OpenAI chat-completion chunks |

These are the usual pairings, not walls. The browser app reads a few `/v1/me/…`
routes for its data controls, and a personal token with the right scope may
call the native run routes.

**Who is calling, in two sentences.** Both doors start with the same FastAPI
dependency, [`require_user`](../../src/audrey/auth.py#L308), which verifies a
Cloudflare Access header against Cloudflare's public keys, or looks a bearer
token starting `aud_pat_` up in Audrey's own database (unexpired and not
revoked). Either way the result is a `Principal`: one Audrey account, with a
stable user id and a private storage namespace that an email change can't
rename, which the lesson on identity and access takes apart later in the
course.

Now the two entry points. The `/v1` one is
[`chat_completions`](../../src/audrey/routes/openai/routes.py#L120):

```python
@router.post("/chat/completions")
async def chat_completions(
    payload: ChatCompletionRequest,
    request: Request,
    me: AuthedUser = Depends(require_user),
):
```

The browser's is [`run_agui_agent`](../../src/audrey/routes/app/runs.py#L1266):

```python
@router.post("/agent")
async def run_agui_agent(
    payload: AgUiRunRequest,
    request: Request,
    mode: Annotated[_Mode | None, Query()] = None,
    model: Annotated[str | None, Query(min_length=1, max_length=200)] = None,
    skill: Annotated[str | None, Query(min_length=1, max_length=200)] = None,
    principal: Principal = Depends(_run_access),
) -> StreamingResponse:
```

Read both the way [Lesson 2 §1](lesson-02-foundations-libraries.md#1-fastapi)
taught. The decorator registers the function for `POST /v1/chat/completions`
or `POST /api/agent`. The request body is parsed into a Pydantic model, and a
body that doesn't match gets a 422 before your code runs. `Depends(...)` runs
the identity check first; if it fails, the caller gets 401 or 403 and the
function never runs. `/v1` routes receive an `AuthedUser`, an older shape kept
for compatibility that carries the `Principal` inside it. Native routes receive
the `Principal` directly; `_run_access` is `require_scope("compat:full")`, the
same check plus a scope requirement when the caller uses a personal token.

The doors look different, but they converge. Whenever the answer is streamed,
both end up in the same function,
[`_stream_via_pipeline`](../../src/audrey/routes/openai/pipeline.py#L274).
§2.4 explains that function, and §2.5 follows the browser's door all the way
into it. First, the pipeline itself.

### 2.3 The pipeline graph, and what a node looks like

Open [`graph.py:637`](../../src/audrey/pipeline/graph.py#L637):

```python
g: StateGraph = StateGraph(PipelineState)
g.add_node("datetime", node_datetime)
g.add_node("memory_recall", node_memory_recall)
g.add_node("classify", node_classify)
g.add_node("complexity", node_complexity)
g.add_node("fast_path", node_fast_path)
g.add_node("escalate_bridge", node_mark_escalated)
g.add_node("planner", node_planner)
g.add_node("deep_panel", node_deep_panel)
g.add_node("research", node_research)
g.add_node("synthesize", node_synthesize)
g.add_node("reflect", node_reflect)
```

This is the whole pipeline laid out as a graph, one node per step. Each node is
an `async def` function defined inside
[`build_graph`](../../src/audrey/pipeline/graph.py#L180), the function this
code belongs to; `grep -n "async def node_" src/audrey/pipeline/graph.py` lists
them. (Defining them *inside* `build_graph` is deliberate, and Lesson 5 explains
why.)

The topology, meaning which node runs after which, follows at
[`graph.py:650`](../../src/audrey/pipeline/graph.py#L650):

```python
g.set_entry_point("datetime")
g.add_edge("datetime", "memory_recall")
g.add_edge("memory_recall", "classify")
g.add_edge("classify", "complexity")
g.add_conditional_edges(
    "complexity", route_after_complexity,
    {"fast": "fast_path", "deep": "planner"},
)
g.add_conditional_edges(
    "fast_path", route_after_fast_path,
    {"end": END, "escalate": "escalate_bridge"},
)
g.add_edge("escalate_bridge", "planner")
# audrey_research forks to the staged research branch; all other deep
# virtual models take the standard panel → synthesize path.
g.add_conditional_edges(
    "planner", route_deep_kind,
    {"panel": "deep_panel", "research": "research"},
)
g.add_edge("deep_panel", "synthesize")
g.add_edge("synthesize", "reflect")
# The research node writes `content` itself (the Writer is the answer), so
# it skips synthesize and goes straight to reflect.
g.add_edge("research", "reflect")
# On a reflect retry, re-enter the same branch the request used.
g.add_conditional_edges(
    "reflect", route_after_reflect,
    {"end": END, "retry": "deep_panel", "retry_research": "research"},
)
```

Read it top to bottom. Every request starts at `datetime` and flows through
`memory_recall`, `classify` and `complexity`. At `complexity` it splits: a
short prompt goes to `fast_path`, where one model answers; a long one goes to
`planner`. After the planner, `audrey_research` takes the staged `research`
branch and every other deep model takes `deep_panel` (several models answer in
parallel) and then `synthesize` (one model reconciles their drafts). Both
branches end at `reflect`, which checks the answer and can send the request
round its branch once more. A fast answer that comes out too thin can also
escalate into the planner.

The same thing as a picture:

```text
   datetime       ← put today's date at the top of the conversation
      │
      ▼
  memory_recall   ← add any saved memories that match the prompt
      │
      ▼
   classify       ← code? reasoning? general? vision?
      │
      ▼
  complexity      ← short prompt or long one?
      │
   ┌──┴─────────────────────┐
 short                     long
   │                        │
   ▼                        ▼
 fast_path               planner     ← split the question into subtasks
   │                        │
   ├─ ok? ──► END      ┌────┴─────────────┐
   │                 other          audrey_research
   └─ too thin? ──►    │                  │
      planner          ▼                  ▼
      (escalate)   deep_panel          research   ← research, verify, write
                       │                  │
                       ▼                  │
                   synthesize             │
                       │                  │
                       └────────┬─────────┘
                                ▼
                             reflect     ← good enough? if not, rerun the branch once
                                │
                               END
```

Don't memorize this; you'll see it many more times. The point for now: **the
pipeline is a graph, and each node is a function.**

What does a node look like? Here's the first one,
[`node_datetime`](../../src/audrey/pipeline/graph.py#L257), with its docstring
left out:

```python
async def node_datetime(state: PipelineState) -> dict[str, Any]:
    sys_msg = datetime_system_message()
    return {"messages": [sys_msg, *state["messages"]]}
```

That's it. A node is a function that:

1. Reads from `state`, a dictionary of everything the pipeline knows so far.
2. Does some work.
3. Returns a dictionary of the fields it wants to update.

LangGraph merges the returned dictionary into the state and runs the next node.
Every node, even the most complex ones like `node_deep_panel`, is structurally
just this.

`PipelineState` is defined in
[`pipeline/state.py`](../../src/audrey/pipeline/state.py) as a `TypedDict`
(see [Lesson 1 §4](lesson-01-foundations.md#4-typed-dictionaries-typeddict) if
that's unfamiliar). The short version: it's a regular dictionary whose allowed
keys and their types are declared up front, so the type checker can catch typos
and editors can autocomplete.

### 2.4 Two drivers over the same building blocks

Here's a fact that surprises most people reading this code for the first time:
the graph is not the only thing that runs the pipeline.

LangGraph runs a graph with `await graph.ainvoke(state)`: every node runs, and
you get the final state back at the end. That suits an answer you wait for, but
a streamed answer needs more. It has to put a "Thinking" banner on the screen
before classification has even finished, report each deep-panel worker as it
completes, and pass the synthesizer's words along one by one. `ainvoke` can't
interleave any of that with the work. So Audrey has a second driver,
[`_stream_via_pipeline`](../../src/audrey/routes/openai/pipeline.py#L274): a
hand-written function that calls the same building blocks the graph's nodes
call (the classifier, the model registry, the planner, the deep panel's
workers, the synthesizer) and emits stream events as it goes.

> **Concept spotlight — async generators.** `_stream_via_pipeline` is an
> `async def` that contains `yield`, which makes it an **async generator**.
> You met `yield` in [Lesson 1 §2](lesson-01-foundations.md#2-context-managers)
> as the point where a context manager pauses. Here it pauses to hand one value
> (one piece of the stream) to whoever is consuming the generator with
> `async for`, then waits until the next value is wanted, and it can `await`
> other work in between. FastAPI's `StreamingResponse` is such a consumer: it
> writes each yielded value to the network the moment it appears. That's how
> one long-running function becomes a stream of events on your screen.

| | The graph | The streaming driver |
|---|---|---|
| Defined in | [`build_graph`](../../src/audrey/pipeline/graph.py#L180) | [`_stream_via_pipeline`](../../src/audrey/routes/openai/pipeline.py#L274) |
| Serves | non-streaming `/v1` requests, through [`_generate_via_pipeline`](../../src/audrey/routes/openai/pipeline.py#L153) | every native run, and streaming `/v1` requests |
| Decides fast or deep | inside the graph, in [`node_complexity`](../../src/audrey/pipeline/graph.py#L335) | itself, before classification, so the first banner goes out at once |
| Produces | one final state, returned as one JSON response | a stream of events: banners, stages, words |

The streaming driver's fast-or-deep decision is a short chain of rules, starting
at [`pipeline.py:343`](../../src/audrey/routes/openai/pipeline.py#L343)
(shortened here):

```python
forced_deep = payload.model in ("audrey_deep", "audrey_cloud", "audrey_local", "audrey_research")
forced_fast = payload.model == "audrey_fast"
image_turn = has_image_part(decision_messages)
if image_turn and not (forced_deep and describe_enabled(cfg)):
    use_deep = False
elif forced_deep:
    use_deep = True
elif forced_fast:
    use_deep = False
else:
    # audrey_auto: deep if the prompt is long OR explicitly asks for
    # depth (short-but-demanding prompts the length gate misses).
    use_deep = complex_ or deep_intent
```

A turn with an attached image goes fast, because a vision model has to see the
image, unless you explicitly picked a deep model. The deep models are always
deep, `audrey_fast` is always fast, and
`audrey_auto` goes deep when the prompt is long or asks for depth. The lesson
on classification and routing walks these rules properly.

The two drivers overlap in two places you should know about.

**On a fast turn whose model can call tools, the streaming driver runs the
graph itself.** The tool loop (the ReAct loop of a later lesson) lives in the
graph, so the driver hands the turn to it and sends the finished answer as one
piece. The comment at
[`pipeline.py:474`](../../src/audrey/routes/openai/pipeline.py#L474) says so:

```python
# If the chosen model is tool-capable and tools are registered, route
# the streaming request through the graph so the ReAct loop can fire.
# Mid-stream tool dispatch isn't supported — we emit one chunk after
# the loop completes, rather than streaming tokens during ReAct rounds.
```

A fast turn on a model *without* tool support doesn't touch the graph; it
streams word by word through
[`stream_fast_path`](../../src/audrey/pipeline/fast_path.py#L244). Before that,
[`_prepare_stream_context`](../../src/audrey/routes/openai/pipeline.py#L1553)
adds the server date and recalls the authenticated person's memories. Recall
is server work: the answering model doesn't need tool support to receive
those facts. It receives no instructions to call tools this branch can't run.

**On a deep turn, the driver repeats the graph's opening steps in its own
code.** [`_phase_thinking`](../../src/audrey/routes/openai/pipeline.py#L1594)
calls the same context helper as the plain fast stream, then plans subtasks.
Together they mirror the graph's `datetime`, `memory_recall` and `planner`
nodes. The context helper uses `datetime_system_message`, `recall_for_request`
and `memory_system_message`; those are also the graph nodes' building blocks.
The fast branch that runs the graph doesn't call the helper again: the graph
already supplies its context.

That's the maintenance rule for this whole section. **One set of building
blocks, two drivers: the graph for an answer you wait for, the streaming driver
for an answer you watch.** When you change *what* the pipeline does, change it
in the shared helpers, and check that both drivers still call them in the same
order.

### 2.5 One message, end to end

Now we can trace one message all the way through. You're in the browser, in a
conversation that already has a few turns. You type *"what is BTRFS?"* with the
model picker on `audrey_auto`, and press Send.

The whole trip, before we walk it:

```text
Send
 → browser: sends only the new message      POST /api/agent
 → Cloudflare Access, the tunnel, nginx      the identity header added, then passed on
 → require_user                              who is this? → a Principal
 → create_run                                your conversation? this model? history from the database
 → begin_run                                 the message and the run, in one transaction
 → launch                                    a task Audrey owns, not the connection
 → the response                              a stream of AG-UI events, with X-Audrey-Run-ID
 → the streaming driver                      fast or deep
 → events                                    rendered as AG-UI, sent to the browser
 → finish                                    the answer saved; the archive woken
```

**1. The browser sends only your new message.** The chat screen uses an AG-UI
client library, whose requests can carry the whole transcript. Audrey's
transport function
[`latestActionFetch`](../../web/src/agentTransport.ts#L1) cuts the request down
to the newest user message before sending it to `/api/agent`. The conversation
id travels as `threadId`, and the chosen model rides along in the URL. There's
no password or token in the browser's code: the request is same-origin, so the
browser attaches the Cloudflare Access session cookie by itself.

**2. Cloudflare Access, the tunnel, nginx.** Access checks the session cookie
and adds the signed identity header; cloudflared carries the request into the
server; nginx forwards it to `audrey` with the header intact (§2.1).

**3. FastAPI asks who you are before the route runs.** The route's
`principal: Principal = Depends(_run_access)` parameter runs `require_user`,
which verifies the header and loads your account. A bad header gets 401; an
account that's pending approval or disabled gets 403. Either way the route
never runs.

**4. The route takes the text and calls `create_run`.**
[`_agui_user_content`](../../src/audrey/routes/app/runs.py#L999) insists the
message is user text, and
[`create_run`](../../src/audrey/routes/app/runs.py#L1029) does the real work. It
starts by loading the conversation, scoped to its owner, at
[`runs.py:1037`](../../src/audrey/routes/app/runs.py#L1037):

```python
conversation = await store.conversations.get(
    user_id=principal.user_id,
    conversation_id=conversation_id,
)
if conversation is None:
    raise HTTPException(status_code=404, detail="Conversation not found.")
```

The lookup includes your user id, so someone else's conversation id finds
nothing and gets the same 404 as an id that doesn't exist. The route never
reveals that another person's conversation exists.

**5. It checks that you may use this model.**
[`resolve_model`](../../src/audrey/model_catalog.py#L270) looks the requested model
up in the catalog your account is allowed to use, and a model you can't use
gets 404. Hiding a model from the picker isn't enforcement; this is.

**6. It settles what will shape the turn.** That means a **skill** (a packaged
instruction set for one kind of task, chosen by you or automatically), your
saved preferences (time zone, response style), any attachments (each must be a
ready file you own), and Project context if the conversation belongs to a
Project. Then it loads **the conversation's history from Audrey's database**,
again scoped to you, at
[`runs.py:1116`](../../src/audrey/routes/app/runs.py#L1116):

```python
previous_records = await store.conversations.list_messages(
    user_id=principal.user_id,
    conversation_id=conversation_id,
)
```

Anything the browser might have sent as history is ignored; the docstring of
[`AgUiRunRequest`](../../src/audrey/routes/app/runs.py#L155) says so in as many
words.

On a new conversation, title generation is auxiliary work. The first turn
is saved with a cleaned version of its prompt as the fallback title. Only
after the answer task releases its resources does
[`_generate_title_after_run`](../../src/audrey/routes/app/runs.py#L566)
ask the small title model for a better label. A conditional database update
protects any rename or other conversation change made while it was pending.
The browser refreshes the title when that background work finishes; a shutdown
can leave the fallback title, but can't undo the saved message or answer.

**7. It records the turn, all or nothing.** At
[`runs.py:1165`](../../src/audrey/routes/app/runs.py#L1165) (most arguments
left out):

```python
started = await store.conversations.begin_run(
    user_id=principal.user_id,
    conversation_id=conversation_id,
    user_content=payload.content,
    ...
)
```

[`begin_run`](../../src/audrey/app_state/repositories.py#L644) writes your
message, an empty assistant message the answer will fill, and a run record in
one database transaction, so either all three exist or none do. If the
conversation already has a run in progress, it refuses, and the route answers
409 Conflict: one active run per conversation.

**8. It launches the run as a task Audrey owns.** `create_run` hands everything
to [`NativeRunManager.launch`](../../src/audrey/routes/app/runs.py#L503), which
starts the run with `asyncio.create_task` and keeps it in its own table of live
runs. The task belongs to Audrey's process, not to your HTTP connection.

**9. It answers with the run's event stream.** Back in `run_agui_agent`, at
[`runs.py:1296`](../../src/audrey/routes/app/runs.py#L1296):

```python
# The manager owns the run independently of this response. A browser
# refresh drops the SSE connection but must not cancel the answer.
response = await _agui_stream_response(
    run_id=created.id,
    request=request,
    principal=principal,
    cursor=AgUiCursor(source_sequence=0, part=None),
)
response.headers["X-Audrey-Run-ID"] = created.id
return response
```

The response is a stream of events for this run. The `X-Audrey-Run-ID` header
tells the browser the run's id, and from then on the browser talks about the
run by id. Stop is `POST /api/runs/{run_id}/cancel`; `GET /api/runs/{run_id}`
reports its status. If the connection drops, the run carries on.

**10. The run enters the shared streaming driver.** The task runs
[`_stream_via_selected_model`](../../src/audrey/routes/app/runs.py#L337):

```python
async def _stream_via_selected_model(
    app,
    payload: ChatCompletionRequest,
    messages,
    options,
    *,
    ...
):
    """Dispatch native workflows normally and direct entries straight to Ollama."""

    if not payload.model.startswith("audrey_passthrough/"):
        async for frame in _stream_via_pipeline(
            app,
            payload,
            ...
            event_context=event_context,
            ...
        ):
            yield frame
        return
```

For one of Audrey's own models like `audrey_auto`, that's the same
`_stream_via_pipeline` a streaming `/v1` request uses, plus an `event_context`
saying which run, conversation and assistant message the events belong to.
(The rest of the function handles **direct models**: one specific model you
picked by name, streamed straight from Ollama with no pipeline around it.)

**11. Fast or deep.** `audrey_auto` with a short, ordinary question means
fast. The "Thinking" stage starts at once, the classifier decides the question
is `general`, and the model registry picks the best healthy `general` model.
If that model can call tools, the graph runs the tool loop; if not, the model
streams word by word (§2.4).

**12. Events flow back, not text.** Every piece of output passes through
[`OpenAIStreamSession`](../../src/audrey/routes/openai/streaming.py#L89). Look
at how it handles a piece of answer text, in
[`content_frame`](../../src/audrey/routes/openai/streaming.py#L151):

```python
def content_frame(self, text: str) -> str:
    if not self._adapter._role_emitted:
        raise RuntimeError("assistant role frame must precede content")
    if self._adapter._terminal_emitted:
        raise RuntimeError("cannot emit content after stream terminal")
    frame = self._adapter.render(self._events.text_delta(text))
    assert frame is not None
    return frame
```

The text first becomes a typed **run event** (`self._events.text_delta(text)`),
and only then is that event rendered in OpenAI's wire format. Audrey keeps one
internal vocabulary of events (a stage started, some answer text, a tool was
called, a model was used, the run finished) and renders it per client. A `/v1`
client receives the OpenAI rendering. A native run ignores those rendered
frames and keeps the events, which go into the run's buffer.
[`AgUiRunEventAdapter`](../../src/audrey/pipeline/agui.py#L234) turns each one
into **AG-UI** events, an open protocol for streaming an agent's progress to a
user interface (`STEP_STARTED` for a stage, `TEXT_MESSAGE_CONTENT` for answer
text, `RUN_FINISHED` at the end), and those reach your browser.

**13. The run finishes and is saved.** When the run's final event arrives,
[`_persist_terminal`](../../src/audrey/routes/app/runs.py#L704) records the
outcome:

```python
finished = await self._store.conversations.finish_run(
    user_id=live.owner_user_id,
    run_id=live.started.run.run_id,
    outcome=terminal.status,
    assistant_content=assistant_content,
    ...
)
if finished is not None and self._archive_wake is not None:
    self._archive_wake()
```

`finish_run` writes the answer, the models used, token counts, sources and tool
calls into the run and the assistant message. Then it wakes the background
worker that copies the finished turn into the chat archive, where later turns
can search it. From this moment, a reload shows the conversation straight from
the database.

That's the whole trip. Three ideas carry most of it:

- **The server owns the conversation.** The browser sends one new message and
  renders what comes back; history, permissions and the record of every answer
  are Audrey's.
- **The request starts the run; it doesn't own it.** The run is a task inside
  Audrey with its own id, so a dropped connection costs you the live view, not
  the answer.
- **Events are the spine.** One internal event vocabulary, rendered per client.

The lessons on the native application return to all three.

A streaming `/v1` request takes a shorter route to the same driver.
`chat_completions` skips the conversation lookup and the run record: the client
sent the whole conversation itself, and Audrey keeps only a searchable archive
copy of each turn. The decision at
[`routes.py:283`](../../src/audrey/routes/openai/routes.py#L283) then picks the
driver:

```python
if payload.stream:
    return StreamingResponse(
        _stream_via_pipeline(
            app, payload, messages, options,
            ...
        ),
        media_type="text/event-stream",
    )

return await _generate_via_pipeline(
    app, payload, messages, options,
    ...
)
```

With `"stream": true` the bot gets the same streaming driver as the browser,
rendered as OpenAI chunks. Without it, `_generate_via_pipeline` runs the graph
and returns one JSON response.

### 2.6 Where data lives, and where to look

Owning conversations and files means Audrey keeps data in several places. Some
of those places are the truth, and some are copies arranged for searching.
Telling them apart matters the day one of them is lost.

> **Concept spotlight — authority versus projection.** An **authority** is the
> place a fact is true by definition: lose it, and the fact is gone. A
> **projection** is a copy of facts arranged for a purpose, usually search,
> that can be rebuilt from an authority. A search index over your documents is
> a projection; the documents are the authority. The rule of thumb:
> **if you can rebuild it, it's a projection.** Back up authorities; rebuild
> projections.

| Where | What's in it | Kind |
|---|---|---|
| The application database ([`application.sqlite_path`](../../config.yaml#L72)) | accounts, tokens, conversations, messages, runs, preferences, Projects | authority |
| The uploads database ([`kb.uploads_db_path`](../../config.yaml#L1043)) and the files under [`kb.upload_root`](../../config.yaml#L1025) | what each person uploaded, its processing status, and the bytes | authority |
| The knowledge-base source folders (mounted read-only into `audrey`) | the shared documents and images | authority, kept outside Audrey |
| Qdrant's knowledge-base and upload collections | vectors for searching documents, images and uploads | projection |
| The chat archive (the sidecar's [`ChatArchiveStore`](../../tools-server/chat_archive.py#L413): SQLite plus a Qdrant index) | a searchable copy of past turns | projection of native conversations; the only copy of `/v1` turns |
| Qdrant's memory collection (the sidecar's [`MemoryStore`](../../tools-server/db.py#L128)) | facts people asked Audrey to remember | **authority**: the one that lives inside Qdrant |

Two rows deserve a second look. The chat archive's copy of native conversations
can be rebuilt on demand: the admin route
[`rebuild_canonical_chat_projection`](../../src/audrey/routes/admin.py#L407)
re-delivers every finished native turn from the application database. Saved
memories are the exception to "Qdrant is search": they're stored as Qdrant
points rather than rows in the application database, so nothing can rebuild
them. What protects them is a backup. The sidecar snapshots the memory
collection when it starts and then once a day, and keeps a configured number
of the newest snapshots
([`memory_backup_retain`](../../tools-server/settings.py#L92)) in a private
folder outside Qdrant's storage. Each snapshot is bundled with a copy of the
chat archive's database, because that's where the sidecar records which
accounts have been purged (deleted along with their data), and a restore must
not bring a purged person's memories back. The
[memory backup guide](../reference/box-operations.md#10-memory-backups-and-isolated-restore-verification)
covers checking a backup and recovering from one; the built-in check restores
into a throwaway collection and never touches the live one.

Finally, the lookup table, for when something goes wrong or you want to
understand something properly:

| If you're asking… | Look in… |
|---|---|
| "Where does a browser message enter Audrey?" | [`routes/app/runs.py:1266`](../../src/audrey/routes/app/runs.py#L1266) (`run_agui_agent`) |
| "Where does a bot's request enter?" | [`routes/openai/routes.py:120`](../../src/audrey/routes/openai/routes.py#L120) (`chat_completions`) |
| "Who is this caller?" | [`auth.py:308`](../../src/audrey/auth.py#L308) (`require_user`) |
| "Why did this turn go fast or deep?" | [`routes/openai/pipeline.py:274`](../../src/audrey/routes/openai/pipeline.py#L274) (`_stream_via_pipeline`), or [`pipeline/graph.py:335`](../../src/audrey/pipeline/graph.py#L335) (`node_complexity`) when not streaming |
| "Why did it pick model X?" | [`pipeline/classify.py`](../../src/audrey/pipeline/classify.py) and [`models/registry.py`](../../src/audrey/models/registry.py) |
| "Why did the answer wait before starting?" | [`pipeline/fair_gate.py`](../../src/audrey/pipeline/fair_gate.py) (the local GPU queue) and [`routes/inflight.py`](../../src/audrey/routes/inflight.py) (the per-user cap) |
| "Where do tools get called?" | [`pipeline/react.py`](../../src/audrey/pipeline/react.py) |
| "What does the browser actually receive?" | [`pipeline/agui.py:234`](../../src/audrey/pipeline/agui.py#L234) (`AgUiRunEventAdapter`) |
| "Is everything up?" | [`routes/admin.py:185`](../../src/audrey/routes/admin.py#L185) (`readiness_status`, `GET /v1/admin/readiness`), which Lesson 5 introduces |

Bookmark this. You'll come back.


## 3. Comprehension questions

Try answering each question yourself before reading the answer.

**1. You refresh the page halfway through a long deep answer. What happens to
the answer, and how does the screen catch up?**

The answer keeps being written. The run is a task owned by
`NativeRunManager`, not by the HTTP request, and the comment in
`run_agui_agent` spells out the contract: a refresh drops the stream but must
not cancel the answer. The reloaded page reads the conversation from the
database, sees that the newest answer belongs to a run that's still `running`,
and shows it as in progress. It then checks the run's status every few seconds
through `GET /api/runs/{run_id}`, and when the run finishes it reloads the
finished messages from the database. Stop still works meanwhile, because Stop
names the run by id, not the dropped connection.

The server can do better than that: `/api/runs/{run_id}/ag-ui-events` resumes a
live event stream after the last event a client saw, from a bounded buffer of
recent events. The current page doesn't use it after a reload, so the words
don't stream back in; the finished answer appears when the run ends.

**2. A bot calls `/v1/chat/completions` with `"stream": true` and `audrey_auto`.
Which driver serves it, and how does its trip differ from the browser's?**

The streaming driver, `_stream_via_pipeline`: the same function a native run
enters. The differences are all around it. The bot proves its identity with a
personal token instead of the Access header. It sends the whole conversation
itself, and Audrey creates no conversation or run records for it, only a
searchable archive copy of the turn. Its events are rendered as OpenAI chunks
and written straight to the bot's connection, with no run manager in between:
if the bot hangs up, the generation is cancelled. If the same bot sent
`"stream": false`, the graph would serve it through `_generate_via_pipeline`.

**3. Why does the browser send only the newest message, and why does the server
ignore any history a client sends anyway?**

Because the server owns the conversation. `create_run` reloads the history from
the database after checking who owns the conversation. If Audrey trusted a
transcript from the client, a modified client could rewrite earlier turns, put
words in the assistant's mouth, or slip in messages the user never sent, and
the model's context would disagree with the stored conversation. Sending one
message also keeps requests small however long the conversation gets. Both
sides enforce it: `latestActionFetch` trims the request in the browser, and
`AgUiRunRequest`'s docstring records that the server ignores submitted history
even when a different client sends it.

**4. The Qdrant volume is wiped and Qdrant restarts empty. What is lost, what
can come back, and what does Audrey's next restart do about it?**

Saved memories are the one real loss, and the backup limits it. Restoring the
newest memory snapshot brings back everything as it was when the snapshot was
taken; anything saved or corrected since has to be told to Audrey again. A
restore is a careful manual job. The purge records have to be reapplied before
memories are served, so purged accounts stay gone, and a memory someone
deleted after the snapshot comes back and has to be found and deleted again.
If Qdrant and the backup folder are both lost, the memories are gone for good.

Everything else in Qdrant is a projection. The knowledge-base and upload
collections can be rebuilt from the source folders and from each upload's
original or the text extracted from it, chat-history search from the
archive's SQLite rows, and the archive's copy of native conversations from the
application database. Conversations, accounts and preferences aren't in Qdrant
at all.

At the next restart, the uploads database stays the authority. The startup
check, [`reconcile_with_qdrant`](../../src/audrey/kb/uploads_db.py#L2810),
reports files whose search content is missing instead of treating them as
deleted. Every file keeps its owner, its status and its place in the owner's
quota, still shows in the owner's list, and can still be opened. What the
restart doesn't do is rebuild the missing vectors: until the files are indexed
again, a search over them finds nothing.

**5. Audrey's container has just been recreated by a deploy, and the next
browser request fails. A minute later everything works. What's the likely
story, and why doesn't it last longer?**

During the recreate, `audrey` is briefly down and then comes back, possibly
with a new IP address. Requests that arrive while it's starting fail: nginx
can't reach it, or Audrey isn't ready yet, because its startup work (Lesson 5)
runs before it accepts requests. It doesn't last because nginx looks Audrey's
address up through Docker's DNS on every request, caching the answer for only
ten seconds (`resolver 127.0.0.11 valid=10s` and the variable in
`proxy_pass`). If nginx had resolved `audrey` once at its own startup, it would
keep sending traffic to the old address until someone restarted `audrey-ui`
too.

**6. Ollama fails in the middle of a native answer. Where does the failure
surface, and what does the person see?**

The response's status line went out long ago (`200 OK`, at the start of the
stream), so there's no HTTP error code left to send. The failure has to travel
inside the stream. Take a fast answer: the code that calls the model catches
the error, adds a short `[ollama error: …]` note after the words already
written, and ends the stream with an error outcome. The stream session turns
that outcome into a `failed` run and an `incomplete` assistant message, which
reach the browser as AG-UI events and are written to the database by
`_persist_terminal`. If an exception escapes all of that, `NativeRunManager`'s
own `except` still finishes the run as failed, with the error code
`pipeline_error`. Either way the run ends in a recorded state, never as a run
stuck at `running`, and a reload shows the failure from the database. Compare a
non-streaming `/v1` request: nothing has been sent yet when Ollama fails, so the
route can still answer with an honest HTTP status, `502 Bad Gateway`.

## When you're ready for the next lesson

The next lesson covers `compose.yaml`, `config.py` and `main.py`:
[Lesson 5 — Configuration and startup](lesson-05-configuration-and-startup.md).
It opens the containers as a map of who may reach what, follows a setting from
YAML and the environment into running code, and walks what FastAPI's
`lifespan` builds before Audrey accepts its first request, including the stores
and background workers this lesson kept pointing at.
