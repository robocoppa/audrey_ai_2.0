# Lesson 5 — Reading a function's signature

**Estimated time:** 20–30 minutes.

**Goal:** To learn to read a function's **signature** — the `: list[dict]` and
`-> int` annotations on its first line — and know what it eats and returns
before reading a single line of its body. Last lesson you finished reading how
`complexity.py` handles *values*; this one is about reading its *functions* from
the header down. By the end you'll read a whole function you've never seen,
cold, using only the habits these lessons have built.

We're still inside
[`src/audrey/pipeline/complexity.py`](../../src/audrey/pipeline/complexity.py).
Keep the REPL from last lesson open (or start a fresh one: `cd` into the project
folder and run `uv run python`).

---

## 1. Annotations: the shape, told to you up front

Go back and look at just the *first lines* — the signatures — of three
functions, ignoring their bodies:

```python
def count_tokens(messages: list[dict]) -> int:
  # line 72
def count_tokens_by_role(messages: list[dict]) -> dict[str, int]:
  # line 78
def is_complex(messages: list[dict], *, threshold: int) -> tuple[bool, int]:
  # line 122
```

Those `: list[dict]` and `-> int` parts are **type annotations** — and a
codebase that takes them seriously, as Audrey does, is making you a promise with
every one of them: *the signature tells the truth about what the function takes
and what it gives back.* `-> int` isn't a hint that it "probably" returns an
integer; it's a claim the code is held to. So the bet pays off like this: you can
read the first line and know the types going in and coming out — no need to open
the body to find out — and trust that answer, because the type checker (and
Cursor, as you write) flags any function whose body breaks the promise its header
made. That trust is the whole point. It's what lets you navigate a big codebase
by reading headers instead of bodies. Watch:

- `count_tokens(...) -> int` — give it a list of message dicts, get back one
  integer. A total.
- `count_tokens_by_role(...) -> dict[str, int]` — get back a dict mapping
  strings to ints. Role name → token count. You knew that already from last
  lesson, but the signature *told* you before you read the loop.
- `is_complex(...) -> tuple[bool, int]` — get back a pair: a `bool` and an
  `int`. Look at the body and it's `return n >= threshold, n` — "is it complex?"
  and "how many tokens?" bundled together so the caller gets both in one call.

Read the return type, know what comes back. That's the skill. The `messages:
list[dict]` on the input side does the same job in reverse — it tells you, and
the editor, what the function expects to be handed.

> **Concept spotlight.** Annotations don't change what the code
> *does* — Python still runs `count_tokens` the same whether or not the `-> int`
> is there. What they buy you is (1) documentation that can't drift out of the
> function's own header, and (2) the editor checking your work: if you try to
> pass a string where a `list[dict]` is expected, Cursor will underline it
> before you ever run the code. There's real depth to the type system —
> `list[dict]` vs. `list[dict[str, int]]`, what happens when a value could be
> "a string *or* `None`," and so on — and that gets its own lesson later. For
> now, the one move to internalize: **glance at the signature first; let it set
> your expectation for the body.**

There's also the `*` in `is_complex(messages, *, threshold)`. That bare star
means *"everything after me must be passed by name."* So callers write
`is_complex(msgs, threshold=500)`, never `is_complex(msgs, 500)`. The star
forces the call site to be self-documenting — a stray `500` floating in an
argument list tells you nothing; `threshold=500` tells you everything. Small
courtesy, big readability win, and now you'll recognize it when you see it.

Finally, the bottom of the file ([complexity.py:154](../../src/audrey/pipeline/complexity.py#L154)):

```python
__all__ = [
    "count_last_user_tokens",
    "count_tokens",
    "count_tokens_by_role",
    "has_deep_intent",
    "is_complex",
    "is_owui_task_request",
]
```

The same `__all__` you met in `scheduling.py`, doing the same job at a larger
scale: these names are the module's public surface. Notice what's *absent* —
`_count_message_tokens`, `_encoder`, `_strip_audrey_markup`. The leading
underscore on those names is the convention for "internal, don't import me from
outside," and `__all__` makes it official. Reading `__all__` is the fastest way
to learn what a module is *for*, without reading the whole thing.

---

## 2. Reading a whole function cold

You now have every piece. To see it click, read one function you *haven't* been
walked through —
`is_owui_task_request` ([complexity.py:111-119](../../src/audrey/pipeline/complexity.py#L111-L119)).
Its signature alone tells you a lot: `(messages: list[dict]) -> bool` — hand it
a conversation, get back a yes/no. The body walks the messages backward to find
the latest `"user"` one, then asks whether that message's text starts with a
particular header string. Notice it reaches for the **same `_iter_text_parts`
helper** from earlier: `next(_iter_text_parts(...), None)` grabs the *first*
text piece (the whole string for plain content, or the first text part of a
list), or `None` if there's no text at all. That's the payoff of pulling the
shape-handling into one place — a second function that needs "the text of a
message" just calls the helper instead of re-deriving the str-or-list dance.

Watch it run. Make the function importable (paste once), then try each call —
predict the result before you press enter:

```python
from audrey.pipeline.complexity import is_owui_task_request
```

An empty conversation, no user message to find:

```python
is_owui_task_request([])
```

Prints `False` — the loop never finds a `"user"` message, so the function falls
through to its final `return False`. A real utility request, whose text opens
with the `### Task:` header:

```python
is_owui_task_request([{"role": "user", "content": "### Task:\nsummarize"}])
```

Prints `True`. And an ordinary question:

```python
is_owui_task_request([{"role": "user", "content": "what's the weather?"}])
```

Prints `False`. (That `### Task:` prefix is how Audrey recognizes a
behind-the-scenes utility request from the client and routes it to the fast
path — substance over ceremony.) Three small habits did all the work there:
read the signature first, recognize the `_iter_text_parts` shape-handling, and
trace `None`/`return False` as the "nothing to do" signal. That's the whole
lesson, applied to a function nobody walked you through.

---

## 3. Why this mattered

With signatures, you can now read `complexity.py` end to end — and you've got
the move that does the most work in unfamiliar code: read the header first, and
let it set your expectation before the body. Three things that header gives you:

- **The contract: types in, types out.** `-> int`, `-> dict[str, int]`,
  `-> tuple[bool, int]` tell you what comes back; `messages: list[dict]` tells
  you what to hand in — all before you read a line of the body, and the type
  checker holds the body to it.
- **The call shape.** A bare `*` says which arguments must be passed by name, so
  a call like `is_complex(msgs, threshold=500)` reads itself — no guessing what
  a stray `500` means.
- **The public surface.** `__all__` names what a module is *for*; the
  leading-underscore names it leaves out are the internals you can ignore from
  outside. Reading it is the fastest way to size up a file you've never opened.

That's the skill these last lessons were building toward: you no longer read a
file top to bottom hoping it adds up. You read the signature, form an
expectation, and check the body against it — the difference between staring at a
wall of plausible lines and *reading*.

---

## 4. Where we're headed next

You've now read two whole files — `scheduling.py` and `complexity.py` — line by
line. The next step is to stop reading single files and start reading the
*project*: how the dozens of files under `src/audrey/` are organized, how a name
like `audrey.pipeline.complexity` maps to a folder and a file on disk, and how
to find the code behind any behavior without being handed the line number. In
the next lesson we'll navigate the real repository — the last "get in and
oriented" stop before we start meeting the bigger language features.
