# Lesson 14 — Fair scheduling: how Audrey shares one GPU

**Estimated time:** 45-60 minutes if you keep
[`pipeline/fair_gate.py`](../../src/audrey/pipeline/fair_gate.py) and
[`routes/inflight.py`](../../src/audrey/routes/inflight.py) open.

**Goal:** by the end of this lesson, you can answer
*"when two users hit Audrey at the same time, who runs first — and
why doesn't a runaway client wedge the box?"*

Earlier lessons traced one request from the door to a model reply
and back. This lesson is about what happens when *more than one
request is in flight at the same time*. A single GPU can only do so
much in parallel, and a self-hosted box has no autoscaler to hide
behind. The answer is two small layers, each solving a different
problem.


## 1. Context

Picture two users sending prompts at the same moment. The box has one
GPU. Whichever request arrived first technically *could* run first,
but a naive "first come, first served" queue is unfair in a way that
matters: if Alice has already fired off twelve research prompts and
Bob walks in with a quick question, Bob waits behind all twelve.
That's not just a bad user experience — it actively encourages a
single noisy client to monopolize the box.

A different angle: even with fair scheduling at the GPU, what stops a
runaway client (a buggy script, a stuck retry loop) from shoving
hundreds of requests at the front door? Each one consumes scheduler
memory, holds an HTTP connection, allocates state objects. The GPU
gate would eventually serialize them, but the box would already be
struggling under the parked weight.

Audrey's answer is **two layers**, each with its own job:

```text
inflight (per-user cap)   - blocks at scheduler entry; protects memory
  └─ fair gate (GPU slot) - blocks at model.chat(); protects VRAM
```

A compact way to remember the split: *in-flight caps the user; the
gate caps the box.*

- **In-flight** is a per-user counter. "How many of *your* requests
  are simultaneously alive in my scheduler?" — capped at 3 by
  default. Request number four for the same user *waits*; it
  doesn't error.
- **The gate** is a per-user round-robin around the single GPU
  slot. "Which user's request gets the GPU next?" — alternates
  across users so a new arrival can slip in ahead of an existing
  user's backlog.

The two layers compose: an incoming request first acquires an
in-flight slot at the route boundary, then (much later, when the
pipeline reaches the model call) acquires the gate around the actual
Ollama call. Cloud calls skip the gate entirely — they're not
competing for local VRAM.


## 2. Read-along

### 2.1 Why two layers, not one

A reasonable first instinct is "one queue should be enough." Walk
through why it isn't.

A single global queue at the front door (say, an `asyncio.Semaphore`
sized to N) protects memory but has no idea who's queued. Alice's
twelfth research prompt blocks Bob's first question — back to the
unfair scenario.

A single per-user fair queue at the GPU protects fairness but has
no idea how many requests one user has parked behind it. A runaway
client pushes 200 requests; they all sail through the front door,
all park behind the gate, all sit on memory.

Those are the two problems and they have different shapes. One is
about *who gets the GPU next* — a scheduling decision. The other is
about *how many of one user's requests should exist at all* — an
admission decision. The clean fix is to make them separate.

The lookup is also different. The gate is acquired *every time
Audrey calls Ollama locally*. In fast mode that's once per request;
in deep mode the planner fans out to a pool of sub-question workers
and the synthesizer runs after — each touch of the GPU is a separate
`gate.acquire()`. So you'll see the gate threaded through
[`pipeline/fast_path.py`](../../src/audrey/pipeline/fast_path.py),
[`pipeline/deep_panel.py`](../../src/audrey/pipeline/deep_panel.py),
[`pipeline/synthesize.py`](../../src/audrey/pipeline/synthesize.py),
[`pipeline/react.py`](../../src/audrey/pipeline/react.py). The
in-flight cap, by contrast, is acquired exactly *once per request*,
right at the route boundary in
[`routes/openai.py:246`](../../src/audrey/routes/openai.py#L246) and
[`routes/openai.py:341`](../../src/audrey/routes/openai.py#L341).
The whole pipeline executes inside that single `async with` block.

### 2.2 The fair gate

Open [`pipeline/fair_gate.py`](../../src/audrey/pipeline/fair_gate.py).
It's short — under 200 lines — but the data structures and the
release logic are doing real work. Walk through it top-to-bottom.

#### What "the slot" actually is

The gate holds an integer, `_available`, starting at `concurrency`
(default 1, set via the `gpu.concurrency` key in `config.yaml`). To
acquire the gate is to subtract 1; to release is to add 1. The
*concept* is identical to an `asyncio.Semaphore`. Why not just use
one? Because a semaphore is strict FIFO — waiters resume in the
order they parked. That gives no per-user fairness, which is the
whole reason this class exists. The gate replaces a vanilla
semaphore with hand-rolled waiter bookkeeping so the *order* of
grants can be controlled.

#### The two data structures

[`fair_gate.py:80-84`](../../src/audrey/pipeline/fair_gate.py#L80-L84):

```python
self._waiters: OrderedDict[str, deque[asyncio.Future[None]]] = OrderedDict()
self._last_granted: str | None = None
```

`_waiters` maps `user_id → queue of parked waiters`. Two things to
notice. First, it's an `OrderedDict`, not a plain `dict`: insertion
order matters, and we move entries to the back when their waiter
gets granted but the deque still has more. That's the "round-robin
order" — head of the dict is "next user in line." Second, the value
is a `deque[Future]`, not a deque of coroutines. We'll come back to
why Futures.

`_last_granted` is a single string: the bucket that received the
most recent grant. The release path uses it to *skip* that bucket
when picking the next winner, as long as anyone else is queued. It
exists because OrderedDict ordering alone isn't enough to get
true alternation — more on that below.

#### Concept spotlight: `asyncio.Future` as a one-shot signal

If you've used `asyncio` before, you've probably mostly used
coroutines (`async def` functions you `await`). A `Future` is a
lower-level primitive: an *object* that represents a result that
will be set later. You can `await` a Future, which suspends until
someone calls `.set_result(value)` (or `.set_exception(...)`) on
it. The awaiter then resumes with that value.

The gate uses Futures as cross-task signals. When user A's task
needs to wait for the slot, it does:

```python
fut = asyncio.get_running_loop().create_future()
self._waiters[bucket].append(fut)
await fut                                # suspends here
```

When someone else releases the slot and the release path picks A's
future, it calls:

```python
fut.set_result(None)
```

That resolves A's `await fut` and A's coroutine resumes. The Future
is a *signal*, not a value — `None` is fine because what we care
about is "go," not "go with payload." Semaphores work this way
under the hood too; we're just doing it explicitly so we can
control which Future gets resolved next.

#### The acquire path

[`fair_gate.py:91-140`](../../src/audrey/pipeline/fair_gate.py#L91-L140)
is an `@asynccontextmanager` — the surface is `async with`. The
flow:

1. **Cloud bypass** (`location != "local"`): yield immediately.
   Cloud calls don't compete for local VRAM, so there's no point
   queueing them.
2. **Bucketing**: `bucket = (user_id or "").strip() or ANON_USER_BUCKET`.
   An anonymous or whitespace-only `user_id` funnels into the
   `__anon__` synthetic bucket, which round-robins against
   authenticated buckets like any other. (The constant lives in
   [`audrey/scheduling.py`](../../src/audrey/scheduling.py), shared
   between the gate and the in-flight registry so they can't drift.)
3. **Fast path** (`fair_gate.py:104-108`): under the lock, if a
   slot is free *and* nobody is queued, just take it. No Future
   needed. This is the common case at low load.
4. **Slow path** (`fair_gate.py:110-118`): create a Future, append
   it to this user's deque, then `await fut` outside the lock. The
   release path will eventually resolve our Future and our
   `async with` body runs.
5. **Cancellation handling** (`fair_gate.py:121-134`): if our
   coroutine is cancelled while parked, we clean ourselves out of
   the deque so the release path doesn't try to grant a slot to a
   dead Future. (There's a complementary defense in `_release`,
   coming up.)
6. **Finally**: on exit, call `_release()`.

#### The release path and the `_last_granted` trick

[`fair_gate.py:141-188`](../../src/audrey/pipeline/fair_gate.py#L141-L188)
is where the fairness actually happens. The intent is round-robin:
when the slot frees, prefer a *different* user from the one that
just had it. But why is this hard?

Imagine the OrderedDict is `{alice: [a2, a3], bob: [b1]}` after
Alice's first acquire releases. The naive picker — "take the head
of the dict" — would grant to Alice again (she's at the head), and
the round-robin never happens. You might fix this by moving Alice
to the back of the dict after each grant. But Alice still has
queued waiters, so on the *next* release, she's at the back of the
dict, Bob is at the head — fine. But what if Alice fires a fourth
request after Bob is served? It appends to Alice's existing
deque (because her key already exists), so dict order doesn't
move. She's still at the back, Bob is at the head, and now there's
nobody at the back to take a turn from Alice — but Alice is *also*
the only one left if Bob's deque drains. The naive scheme fails
silently in the multi-arrival case.

`_last_granted` cuts through the muddle. The picker rule is:

> Prefer any bucket *other* than `_last_granted`. Fall back to
> `_last_granted` only when it's the sole queued user.

That single piece of state guarantees alternation while there are
multiple buckets, and gracefully re-grants to the same user when
they're the only one waiting.

Concretely, picture Alice queuing three requests, then Bob arriving
with one while Alice's first is mid-flight. Alice's first releases.
The picker sees Alice and Bob both queued; `_last_granted == "alice"`,
so it skips Alice and grants Bob. Without the skip, Bob would have
sat through Alice's a2 and a3 first. With it, Bob slips into slot
two and Alice's backlog drains after. That's the scenario the
fairness rule protects against, and it's why the module exists in
this form rather than as a plain semaphore.

#### The done-future sweep

Before picking, `_release` walks every deque and pops `done()`
futures from the head ([`fair_gate.py:150-161`](../../src/audrey/pipeline/fair_gate.py#L150-L161)).
A future is `done()` if it's been resolved, cancelled, or had an
exception set. Where do dead futures come from?

Cancellation. When a client disconnects mid-wait, the parked
coroutine raises `CancelledError`, which marks its future
cancelled. The `acquire` path's except handler removes the future
from the deque — but there's an unavoidable window between the
future being marked cancelled and the handler actually running.
If a release fires inside that window and the dead future is at
the head of a deque, the picker would otherwise call
`set_result(None)` on an already-cancelled future, which raises
`InvalidStateError`. The head-sweep defuses that race.

This is the kind of thing that's easy to "optimize away" because
the sweep looks redundant with the cancel-side cleanup. It isn't —
the sweep is the only thing that handles the in-between window.
The module docstring at the top of the file spells this out so
nobody removes it on a tidy-up pass.

#### Cloud bypass and `concurrency.config`

The cloud bypass exists because cloud calls don't touch the local
GPU. They go out over HTTP to an Ollama bridge, sit on a remote
machine's hardware, and return tokens. There's no VRAM to protect,
so there's no point making them wait. The `gate.acquire("cloud-x",
location="cloud")` call still happens in
[`pipeline/fast_path.py`](../../src/audrey/pipeline/fast_path.py)
and elsewhere — the gate just yields immediately without touching
its internal state.

`concurrency=1` is the default in `config.yaml`'s `gpu` section,
matching a "one model running at a time" strategy. If you bumped it
to 2 (e.g., on a box with two GPUs the registry knows how to
target separately), the gate would simply allow two concurrent
holders before anyone parks. The round-robin logic is unchanged —
it'd just kick in at the third concurrent caller rather than the
second.

### 2.3 The in-flight cap

Open [`routes/inflight.py`](../../src/audrey/routes/inflight.py).
Smaller file, simpler structure, but worth working through because
the eviction rule has a subtle race that the current code is
designed around.

#### What the cap is

A per-user `asyncio.Semaphore`. Default size 3 (from
`fairness.max_inflight_per_user` in `config.yaml`). When user A
acquires their fourth slot, they block on
`sem.acquire()` until one of their existing three releases.
There's no error — request four just *starts later*. From A's side,
"the request took an extra second to begin" rather than "the
request failed."

The registry as a whole holds a `OrderedDict[str, Semaphore]` —
one semaphore per active user — capped by `max_tracked_users`
(default 1024). That cap is a soft guideline, not a hard ceiling.
When you exceed it, the registry logs a warning and increments
`audrey_inflight_cap_breached_total` so you can see how often it
happens.

#### Concept spotlight: `asyncio.Semaphore` as a counter

A semaphore is a counter with a wait list. It starts at some
initial value `N`. Each `await sem.acquire()` decrements the
counter; when the counter would go negative, the caller parks. Each
`sem.release()` increments the counter; if anyone is parked, the
oldest waiter wakes up and the counter stays at zero (the wake
consumes the increment).

Three useful properties:

1. **It's a counter, not a lock.** A semaphore initialized at 3
   can be held by three coroutines simultaneously. A lock (size 1)
   is a special case.
2. **Acquire blocks; it doesn't error.** A coroutine that calls
   `acquire()` on an exhausted semaphore just suspends. It'll
   resume when somebody releases.
3. **It's per-event-loop.** This is single-process state. A
   second uvicorn worker would have its own semaphores;
   coordination would require something out-of-process (Redis,
   etc.). Audrey deliberately runs single-process for exactly
   this reason — it makes per-user bookkeeping trivial.

The per-user semaphores are lazy-created: there's no entry until
the user actually shows up. That's important for the next part.

#### The eviction problem

A naive registry would hold every user that ever made a request,
forever. That leaks memory. So the registry evicts.

The simple rule is "when adding a new user would exceed
`max_tracked_users`, evict the least-recently-used idle user."
LRU is implemented via OrderedDict ordering: the most recently
touched entry is moved to the tail, so the head is the LRU. "Idle"
means nobody is holding or waiting on that user's semaphore — safe
to drop, the user can re-register on their next request.

The subtle part is *what counts as idle*. The straightforward
version: "the semaphore's count equals its max." But the actual
in-use count is only incremented *after* `sem.acquire()` returns,
which is *after* the caller has parked and resumed. That's a
problem when the eviction loop runs in the window between "user B
called `_reserve` and got a semaphore back" and "user B's
`sem.acquire()` resolved and `_inuse` got bumped." During that
window, B looks idle. A third user could trigger eviction, evict
B's semaphore, and leave B parked on a semaphore that's no longer
tracked. When B's holder eventually releases, the registry sees no
record of B; the next request for B gets a brand-new semaphore.
Two holders can briefly run side by side, exceeding the per-user
cap.

The fix is to track *reservations*, not just *holders*. The
registry now bumps a counter under its lock the moment it hands a
semaphore out, before the caller awaits acquisition. Eviction
treats any non-zero reservation as "in use" and skips that bucket.
[`routes/inflight.py:58-98`](../../src/audrey/routes/inflight.py#L58-L98)
shows the rule: `_reserve` increments inside the same `async with
self._lock` block where it inspects or creates the semaphore;
`_drop_reservation`
([`inflight.py:100-103`](../../src/audrey/routes/inflight.py#L100-L103))
decrements on cancellation; the `slot()` context manager decrements
at the same time it drops `_inuse`.

When eviction can't find anyone idle, the registry accepts the
overflow — it'd rather hold one extra entry briefly than block the
caller — and bumps the breach counter so you can graph it.

#### Cancellation cleanup

If a waiter is cancelled while parked on `sem.acquire()`, the
`slot()` context manager's `except BaseException` clause calls
`_drop_reservation`
([`inflight.py:116-121`](../../src/audrey/routes/inflight.py#L116-L121)).
Without that, the reservation counter would leak — the bucket
would look perpetually in-use, never eligible for eviction.
`BaseException` is intentional: it covers both `CancelledError`
(which doesn't inherit from `Exception` in modern Python) and any
other unusual exit. Whatever happens, the reservation comes off.

#### `_safe_bucket`

[`inflight.py:148-153`](../../src/audrey/routes/inflight.py#L148-L153):
when the wait time exceeds one second, the registry logs a line so
you can see who's getting throttled. The log uses `_safe_bucket`,
which trims an email to an 8-character local-part prefix. The user
ID at this layer is an email address (the same one
[`auth.py`](../../src/audrey/auth.py)'s `AuthedUser.email`
resolves), and you don't want full email addresses landing in
container logs. The trim is small but it's a deliberate privacy
gesture worth keeping.

### 2.4 How they fit together at the route boundary

Both objects are constructed in the lifespan handler in
[`main.py:60-66`](../../src/audrey/main.py#L60-L66), stashed on
`app.state.gate` and `app.state.inflight`, and pulled by the route
per request. The in-flight cap wraps the *whole* pipeline call:

```python
# routes/openai.py
async with inflight.slot(user_id):
    final = await _run_graph_with_metrics(graph, state)
```

The gate wraps each *individual* Ollama call inside the pipeline,
many layers deeper. A deep-mode request, for example, holds one
in-flight slot for its entire lifetime, and inside that time it
acquires-and-releases the gate once per sub-question worker plus
once for synthesis. The two layers never see each other directly —
the gate doesn't know who the user is at the framework level, and
the in-flight cap doesn't know what the pipeline is doing. They
just both look at the same `user_id`.

A process restart drops both — the gate's `_waiters` and the
registry's `_sems` are in-memory state on the uvicorn event loop.
That's fine: in-flight requests die with the process anyway, and
the cold state on restart is correct (no users tracked, nothing
queued).


## 3. Comprehension questions

1. **A user complains their first prompt of the day takes 8
   seconds to start. The Grafana board shows
   `audrey_user_inflight_blocked_seconds` near zero but
   `audrey_gpu_gate_wait_seconds` p99 at 7 seconds. Which layer is
   the bottleneck and what would you try first?**

2. **You bump `gpu.concurrency` from 1 to 2. What changes about
   the fair gate's behavior, and what risks have you taken on at
   the box level?**

3. **A buggy client retries aggressively and pushes 200 requests
   in five seconds, all under the same `user_id`. Walk what
   happens: which layer triggers first, what does the user see,
   what does memory look like, and what would the operator see in
   metrics?**

4. **`audrey_auto` routes a small chitchat prompt to fast mode
   over a cloud model. Does either layer acquire anything? Why?**

5. **A streaming deep-mode request is cancelled mid-flight — the
   browser tab closed. The deep panel had six sub-question
   workers; two had already finished and returned, one was inside
   `gate.acquire()` and held the slot, three were parked on
   `await fut` waiting for the gate. What needs to happen for the
   gate to end up in a clean state, and which line of
   `fair_gate.py` does the load-bearing cleanup?**

6. **You see `audrey_inflight_cap_breached_total` ticking up a
   few times an hour. The default `max_tracked_users` is 1024.
   What does this tell you about your user base, and would you
   change the cap?**
