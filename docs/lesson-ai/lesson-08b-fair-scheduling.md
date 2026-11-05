# Lesson 8b — Fair scheduling: how Audrey shares one GPU

**Estimated time:** 30-45 minutes if you read with the source files open.

**Goal:** by the end of this lesson, you can answer *"when several requests want
the GPU at once, who goes next, and why can't one person's backlog, or Audrey's
own background work, hold up everyone else's chat?"*

[Lesson 7](lesson-07-calling-models.md) showed a local model call waiting for the
GPU gate before it reaches Ollama, and a cloud call going straight through. This
lesson is about that wait. Audrey runs one local generation at a time by
default, while people chat, bots call the API and agents work through their
tasks. Two queues decide who goes next, and one rule settles priority between
someone waiting at the screen and work that nobody is watching.

## 1. Context

### 1.1 One GPU, many callers

A local model call holds the GPU for seconds, sometimes minutes, and a deep
answer makes several in a row. By default Audrey runs one local generation at a
time (`gpu.concurrency`): A second model running alongside would compete for the
same GPU memory, and switching models means loading new weights. Meanwhile,
local calls come from many places:

| Local model calls from | Treated as |
|---|---|
| Chat turns in the browser, and bots on `/v1` | interactive |
| A new conversation's title, generated after its first answer | interactive |
| Image descriptions, including frames from an uploaded video | interactive |
| Agents working through a saved task, and page watches checking a page on a schedule | background |

Background work runs on its own, with nobody waiting at the screen for the next
word. When it finishes matters, but a few seconds either way don't.

A plain first-come, first-served line would get this wrong in three ways:

- **One person's backlog would hold up everyone.** With five of Alice's deep
  questions queued, Bob's quick question would wait for all of them.
- **Background work would hold up chat.** An agent's long task queued ahead of a
  chat would keep the person at the screen waiting for work nobody is watching.
- **Nothing would limit how much one caller can start.** A runaway script could
  park hundreds of requests inside Audrey, each holding memory and a connection.

### 1.2 Two queues and one priority rule

```text
a request (one person; interactive or background)
 -> per-user cap   a few of this person's requests at a time      inflight.py
     -> the request runs: routing, retrieval, tools, ...
         -> each local model call
             -> GPU gate   one generation at a time; who's next?  fair_gate.py
                 -> Ollama
```

- **The per-user cap** limits how many of one person's requests are in flight at
  once, three by default. The next one waits at the door. It doesn't fail.
- **The GPU gate** decides who uses the GPU next. It takes turns between people,
  not between requests, so one person's long queue doesn't push anyone else
  back.
- **The priority rule** applies in both: Interactive work goes before background
  work, until the background work has waited long enough, 30 seconds by default.
  After that, it takes its turn like anyone else's.

In short: The cap limits a person, the gate shares the machine, and the priority
rule keeps background work out of a chat's way without starving it. Neither queue
interrupts a generation that has started. Both only decide who goes next when a
slot comes free.

Lesson 1 met both kinds of slot, and the rule that a slot has to come back however
a request ends. Keeping that rule is harder than it looks, and §2.6 shows why.

### 1.3 The files

| Job | File |
|---|---|
| The GPU gate: turns between people, priority and aging | [`fair_gate.py`](../../src/audrey/pipeline/fair_gate.py) |
| The per-user cap, and the permit background work gives back | [`inflight.py`](../../src/audrey/routes/inflight.py) |
| Whether work is interactive or background, and the aging setting | [`scheduling.py`](../../src/audrey/scheduling.py) |
| How agents and watches wait for both queues | [`background_admission.py`](../../src/audrey/background_admission.py) |
| Building one gate and one cap for the whole process | [`main.py:140`](../../src/audrey/main.py#L140 'gpu_concurrency = int(cfg.raw.get(') |
| The settings | [`config.yaml:1408`](../../config.yaml#L1408 'gpu:') |

## 2. Read-along

### 2.1 Who takes what, and when

Startup builds one gate and one per-user registry, and every request shares
them. The two are taken at different depths:

- **A per-user slot: once per request, around all of it.** A chat turn takes one
  before the pipeline starts and keeps it until the answer is finished
  ([`pipeline.py:327`](../../src/audrey/routes/openai/pipeline.py#L327 'async with inflight.slot(user_id):')),
  whether the turn goes fast or deep, to a local model or a cloud one. A
  direct-model chat
  ([`runs.py:483`](../../src/audrey/routes/app/runs.py#L483 'async with app.state.inflight.slot(user_id):')),
  a `/v1` request, a new title and each frame of an uploaded video
  ([`media.py:135`](../../src/audrey/routes/media.py#L135 'async with app.state.inflight.slot(req.user):'))
  take one the same way.
- **The GPU gate: once per local generation, around only that call.** A deep
  answer's workers, its synthesis and each round of a tool loop take the gate
  for one generation and give it back.

```text
one deep answer          holds one per-user slot from start to finish
  worker 1               takes the gate for one generation, then gives it back
  worker 2               the same
  a worker's web search  no gate: other people's calls use the GPU meanwhile
  synthesis              takes the gate for one generation
```

Tool calls run between a loop's rounds, outside the gate
([`_gate_ctx`](../../src/audrey/pipeline/react.py#L545)), so a slow web search
never holds the GPU. The price is that someone else's call can slip in between
rounds, and Ollama may have to load the first model again.

Three more rules complete the picture:

- **Cloud calls skip the gate, but not the cap.** They run on someone else's
  hardware, so they take no GPU turns. They still count toward how many of a
  person's requests are in flight.
- **Both queues know people by their storage namespace** (Lesson 6b). Every
  caller passes the same id, so a person's chats, their bots, their video's
  frames and their agents all queue as one person.
- **Two kinds of local call skip the gate on purpose:** the router's
  classification from Lesson 8, which runs on almost every turn, and embeddings.
  Neither has to wait behind a deep answer. The catch is spelled out above the
  router setting
  ([`config.yaml:12`](../../config.yaml#L12 'router:')): A call that doesn't
  wait its turn competes with the running generation for GPU memory, so the
  router's model has to stay small.

### 2.2 The gate takes turns between people

A plain semaphore can count slots, but it always wakes the longest waiter, and
there's no way to tell it "skip Alice this time". So the gate keeps its own
waiting lists ([`FairLocalGate`](../../src/audrey/pipeline/fair_gate.py#L39)):
one queue per person, in an ordered dictionary. Each waiter parks on an asyncio
future: an object one task waits on until another task gives it a result. The
gate gives a waiter its turn by setting that result.

Taking a slot is one locked step
([`fair_gate.py:74`](../../src/audrey/pipeline/fair_gate.py#L74 '# Availability and enqueue must be one transition')):
A free slot with nobody waiting is taken at once, and otherwise the caller joins
the end of that person's queue. If checking and joining were two steps, a
release could land between them, and the caller would sit in a queue that
nobody would look at again.

Giving a slot back ([`_release`](../../src/audrey/pipeline/fair_gate.py#L126))
hands it straight to the next waiter. It never becomes free in between, so a
newcomer can't take it from under people who were already waiting. The choice
goes like this:

```text
a slot comes back
 -> drop waiters whose request was cancelled
 -> nobody left?    the slot becomes free
 -> which tier?     interactive priority if anyone waiting has it (§2.3),
                    otherwise background
 -> which person?   the first in line who wasn't served last in that tier
 -> which request?  that person's oldest waiter in the tier
```

Say Alice has three calls waiting, and Bob arrives with one while Alice's first
is running. When it finishes, the gate passes over Alice, who was served last,
and Bob goes next. Alice's other two follow. Turns go to people, so ten waiting
calls don't earn Alice ten turns in a row.

### 2.3 Interactive first, background after a while

A request can't choose its own priority. Trusted server code marks background
work, by setting a context variable that every call made inside the block can
read ([`local_workload`](../../src/audrey/scheduling.py#L36)). Agents'
steps and watches' checks run inside that block
([`background_admission.py:28`](../../src/audrey/background_admission.py#L28 'with local_workload("background", queued_at=queued_at):')).
Everything else is interactive, the default, and nothing a browser sends can
change it.

When a slot comes back, a waiter has **interactive priority** if it's
interactive, or if it's background work that has waited at least
`background_aging_seconds`
([`fair_gate.py:173-177`](../../src/audrey/pipeline/fair_gate.py#L173-L177)):

```python
def _is_interactive_priority(self, waiter: _Waiter, now: float) -> bool:
    return (
        waiter.workload == "interactive"
        or now - waiter.enqueued_at >= self._background_aging_seconds
    )
```

If anyone waiting has interactive priority, only they compete, taking turns
between people as in §2.2. Background work that hasn't aged gets the GPU only
when no one with interactive priority is waiting.

Aging is what keeps background work from starving. Under steady chat, an agent's
step that has waited the full interval stops giving way and takes its turn in the
rotation like a person. The setting is a priority threshold, not a deadline: The
running generation, and the waiters ahead of it, still finish first. A
background call's age counts from when it first asked for a slot, and it keeps
that age through the retries in §2.5, so a stream of chats can't keep resetting
its clock.

### 2.4 The per-user cap

Each person gets a small semaphore of their own, created at their first request
and sized by `max_inflight_per_user`
([`_PrioritySemaphore`](../../src/audrey/routes/inflight.py#L74)). It applies
the same priority rule when it hands a slot on
([`_wake_next`](../../src/audrey/routes/inflight.py#L141)): Interactive and aged
waiters go first, in the order they arrived. A request over the cap waits for
one of the person's earlier requests to finish. It isn't refused, so a bot
needs no retry logic, and the person just sees the answer start a little later.

The registry keeps those semaphores in a map ordered by last use
([`UserInflightRegistry`](../../src/audrey/routes/inflight.py#L159)), for up to
`max_tracked_users` people. When someone new arrives and the map is full, it
drops the least recently used person who has nothing in flight
([`_reserve`](../../src/audrey/routes/inflight.py#L188)). It never drops anyone
with a request running or waiting: Their next request would get a fresh
semaphore, and for a while they could exceed their cap. So a request counts as
in flight from the moment `_reserve` hands it the semaphore, before it starts
waiting, which closes the gap in which a request that has just arrived would
look idle. If every tracked person has something in flight, the registry goes
over its limit and logs a warning, rather than make anyone wait. The limit is a
soft one.

When a request waits more than a second at the door, Audrey logs it with a
shortened id ([`_safe_bucket`](../../src/audrey/routes/inflight.py#L307)), so a
full email address, the namespace of an account from before Audrey kept its own
accounts, never lands in the logs.

### 2.5 Background work gives its place back

Holding a per-user slot while waiting for the GPU causes a problem of its own.
Picture Alice's agent in a Work step, holding her only per-user slot (the cap can
be set to 1) and waiting for the GPU behind Bob's answer. Alice sends a chat. At
the gate, her chat would go ahead of the agent's unaged call, but it never gets
there: It's stuck at the door, behind the very call it would have passed.

So background calls enter both queues through
[`BackgroundGenerationGate`](../../src/audrey/background_admission.py#L13),
which holds the per-user slot on a **permit**: an agreement to hand the slot
back if its owner's chat starts waiting.

```text
a background call
 -> takes a per-user slot, with a give-way permit
 -> waits for the GPU, and listens for "give way" at the same time
      the GPU comes first     -> mark the permit running, then generate;
                                 nothing can take the slot back now
      "give way" comes first  -> leave the GPU queue, return the per-user slot,
                                 and start again with the same age
```

The per-user semaphore sends "give way" when an interactive request starts
waiting there. It signals every permit of that person's that hasn't started its
generation and hasn't aged
([`_notify_background_permits`](../../src/audrey/routes/inflight.py#L130)).
Aged work keeps its place, which stops a busy person's chats from sending their
own agent back to the start forever.

The GPU grant and the "give way" signal can arrive together.
[`mark_running`](../../src/audrey/routes/inflight.py#L60) settles it before the
model call starts: If "give way" has already fired, the call gives the GPU slot
back and retries, so a call whose permit was taken back never runs. Once a
permit is marked running, its generation finishes uninterrupted.

### 2.6 Cancellation: giving back what you hold

A request can stop at any moment: Someone presses Stop or closes the tab, a
timeout fires, or the process shuts down. Whatever the moment, both queues have
to end up with the right number of free slots. There are three cases:

```text
cancelled while generating          the finally block hands the slot on
cancelled while waiting             the waiter leaves its queue; a release
                                    that runs first sweeps it out anyway (§2.2)
cancelled just after being picked   the slot is already the waiter's, so it
                                    must hand the slot on, or nobody will
```

The third case is the subtle one. The gate picks a waiter by setting its
future's result, but the waiting task notices only when the event loop next runs
it. If the task is cancelled in between, it wakes with the cancellation instead
of the slot. The gate has already counted the slot as given, and the
`async with` body never started, so its `finally` block will never run. The
slot would be lost for good, and with one slot, every later local generation
would wait forever: Audrey would stop answering until a restart. The
cancellation handler checks for this
([`fair_gate.py:97-113`](../../src/audrey/pipeline/fair_gate.py#L97-L113)):

```python
except asyncio.CancelledError:
    # Cancellation of the queued await cancels its Future. If
    # dispatch already set a result, we own a slot and must
    # transfer it even though the context body never started.
    async with self._lock:
        was_granted = fut.done() and not fut.cancelled()
        ...
    if was_granted:
        await self._release()
    raise
```

The per-user semaphore makes the same check
([`inflight.py:102`](../../src/audrey/routes/inflight.py#L102 'except BaseException:')).
The background adapter runs its cleanup in a shielded task, so a second
cancellation, such as shutdown arriving during a timeout, can't stop it partway
([`background_admission.py:52`](../../src/audrey/background_admission.py#L52 'cleanup = asyncio.create_task(')).

Both queues live in memory, in Audrey's single backend process. A restart
empties them, and that's correct: Everything that was waiting stopped with the
process, and the startup recovery from Lesson 6a settles the runs it left
behind. The queues also depend on that single process. A second one would keep
queues of its own, and the two would share the GPU without knowing about each
other.

### 2.7 The settings

All four live in
[`config.yaml:1408-1424`](../../config.yaml#L1408-L1424 'gpu:'):

| Setting | Default | What it decides |
|---|---|---|
| `gpu.concurrency` | 1 | how many local generations run at once |
| `fairness.max_inflight_per_user` | 3 | how many of one person's requests are in flight at once |
| `fairness.background_aging_seconds` | 30 | how long background work waits before it competes with chat |
| `fairness.max_tracked_users` | 1024 | the soft limit on people the cap keeps track of |

Raising `gpu.concurrency` lets that many generations run side by side. The
rotation works the same way, but queuing starts one caller later. Whether it
helps depends on the hardware: Two different models may not fit in GPU memory
together, and Ollama swapping weights between them can cost more than running
in parallel saves.

The aging setting must be above 0 and at most 300 seconds. Startup checks it
([`LocalSchedulingSettings`](../../src/audrey/scheduling.py#L19)), so a value
outside that range stops Audrey from starting, rather than quietly changing how
the queues behave.

Readiness reports how busy both queues are as plain counts, such as the slots in
use and how many interactive and background calls are waiting
([`pressure_snapshot`](../../src/audrey/pipeline/fair_gate.py#L183)). It never
reports who is waiting.

## 3. Comprehension questions

**1. Alice already has three answers in progress, the default cap, and sends a
fourth question from another conversation. A moment later, Bob sends his first.
Who waits, where, and for what?**

Alice's fourth question waits at the per-user cap until one of her three
finishes. It isn't refused (§2.4). Bob's request goes straight through the door.
At the gate, his first local call goes ahead of Alice's waiting calls, because
the gate takes turns between people and Alice was served last (§2.2). He still
waits for the generation that's already running.

**2. Bob's agent is in a Work step, waiting for the GPU, when Bob sends a chat.
The per-user cap is set to 1. What happens to each?**

The chat starts waiting at Bob's per-user semaphore as interactive work, which
tells the agent's permit to give way. The agent leaves the GPU queue and hands
back the slot. The chat takes it, and at the gate it goes ahead of unaged
background work (§2.3, §2.5). The agent tries again with its original age, so it
can't be pushed back forever. Had the agent's generation already started, the
chat would wait for it to finish: Nothing running is interrupted.

**3. Several people chat steadily all afternoon. Does a page watch's check ever
get the GPU?**

Yes. Once the check has waited `background_aging_seconds`, it has interactive
priority and takes its turn in the rotation like a person (§2.3). Its age counts
from its first request for a slot and survives give-way retries, so the chats
can't keep resetting it.

**4. Alice presses Stop at the instant the gate picks her waiting call. What
could go wrong, and what prevents it?**

Her call is cancelled after the gate gave it the slot but before her task
resumed. The `async with` body never starts, so its `finally` never runs, and
the slot would count as in use forever. With one GPU slot, every later local
generation would wait for good. The cancellation handler sees that its future
already has a result, and hands the slot on (§2.6).

**5. A bot sends a burst of requests with Bob's token, all to a cloud model.
What limits it, and whose chats does it slow down?**

The per-user cap: Only a few of Bob's requests are in flight at once, and the
rest wait at the door (§2.4). Cloud calls skip the GPU gate, so the burst takes
no GPU turns from anyone (§2.1). The only person it slows is Bob, whose own chats
now wait for per-user slots behind his bot's requests.

## When you're ready for the next lesson

[Lesson 9 — Deep mode and research mode](lesson-09-deep-mode.md) opens the deep
path that Lesson 8 routes into: the planner, the panel of workers that take
turns at this lesson's gate, synthesis and reflect, and research mode, which
checks claims before one writer answers.
