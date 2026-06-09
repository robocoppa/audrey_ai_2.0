# Lesson 4 — Reading a function's signature

**Estimated time:** 25–35 minutes.

**Goal:** finish reading `complexity.py`. Last lesson you learned to handle a
value whose *type* can vary; now you'll learn to read a function's **signature**
— the `: list[dict]` and `-> int` annotations on its first line — and know what
it eats and returns before reading a single line of its body. By the end you'll
read a whole function you've never seen, cold, using only the habits this file
has taught.

We're still inside
[`src/audrey/pipeline/complexity.py`](../../src/audrey/pipeline/complexity.py).
Keep the REPL from last lesson open (or start a fresh one: `cd` into the project
folder and run `uv run python`).

---

## 1. `None` as a signal, and the `or "other"` idiom

Look at `count_tokens_by_role` ([complexity.py:78-85](../../src/audrey/pipeline/complexity.py#L78-L85)):

```python
def count_tokens_by_role(messages: list[dict]) -> dict[str, int]:
    """Per-role token sums; unknown/missing roles bucket under `"other"`."""
    enc = _encoder()
    totals: dict[str, int] = {}
    for m in messages:
        role = m.get("role") or "other"
        totals[role] = totals.get(role, 0) + _count_message_tokens(m, enc)
    return totals
```

`m.get("role") or "other"` is one of the most common idioms in Python, and now
you have everything to read it. `m.get("role")` returns the role string if it's
there — `"user"`, `"assistant"`, `"system"` — or `None` if the key is missing.
The `or` then does its job: in Python, `a or b` evaluates to `a` if `a` is
"truthy," otherwise `b`. `None` is falsy, and so is an empty string `""`. So
the whole expression means: *"use the role if there is a real one; otherwise
fall back to `"other"`."* It's the "missing → default" pattern in a single
line, and you'll see it everywhere.

This is **`None` used as a signal.** `None` isn't an error here; it's
information — "this message didn't say what role it is." The code reads that
signal and responds by bucketing the tokens under `"other"` rather than
crashing or dropping the message. Notice the contrast with last lesson: there,
an unreadable *content* meant "count nothing" (the helper yields no text); here,
a missing *role* means "bucket it under other" ("count it, just not under a
known role"). `None` shows up in both, and each function decides for itself what
the absence should mean — the same "missing value" gets two different,
intentional treatments.

> The line after it, `totals.get(role, 0) + ...`, uses `.get`'s *second*
> argument: `totals.get(role, 0)` means "the running total for this role, or
> `0` if we haven't seen this role yet." Same `.get` you already know, now with
> an explicit default instead of `None`. It's the standard way to accumulate
> into a dict without a "have I seen this key before?" check every time.

---

## 2. Annotations: the shape, told to you up front

Go back and look at just the *first lines* — the signatures — of three
functions, ignoring their bodies:

```python
def count_tokens(messages: list[dict]) -> int:                  # line 72
def count_tokens_by_role(messages: list[dict]) -> dict[str, int]:  # line 78
def is_complex(messages: list[dict], *, threshold: int) -> tuple[bool, int]:  # line 122
```

Those `: list[dict]` and `-> int` parts are **type annotations**, and the bet
of this whole codebase is that *you can read the signature and know the shape
before you read a single line of the body.* Watch:

- `count_tokens(...) -> int` — give it a list of message dicts, get back one
  integer. A total.
- `count_tokens_by_role(...) -> dict[str, int]` — get back a dict mapping
  strings to ints. Role name → token count. You knew that already from §1, but
  the signature *told* you before you read the loop.
- `is_complex(...) -> tuple[bool, int]` — get back a pair: a `bool` and an
  `int`. Look at the body and it's `return n >= threshold, n` — "is it complex?"
  and "how many tokens?" bundled together so the caller gets both in one call.

Read the return type, know what comes back. That's the skill. The `messages:
list[dict]` on the input side does the same job in reverse — it tells you, and
the editor, what the function expects to be handed.

> **Concept spotlight, lightly.** Annotations don't change what the code
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

Finally, the bottom of the file ([complexity.py:128](../../src/audrey/pipeline/complexity.py#L128)):

```python
__all__ = [
    "count_last_user_tokens",
    "count_tokens",
    "count_tokens_by_role",
    "is_complex",
    "is_owui_task_request",
]
```

The same `__all__` you met in `scheduling.py`, doing the same job at a larger
scale: these five are the module's public surface. Notice what's *absent* —
`_count_message_tokens`, `_encoder`, `_strip_audrey_markup`. The leading
underscore on those names is the convention for "internal, don't import me from
outside," and `__all__` makes it official. Reading `__all__` is the fastest way
to learn what a module is *for*, without reading the whole thing.

---

## 3. Reading a whole function cold

You now have every piece. To see it click, read one function you *haven't* been
walked through —
`is_owui_task_request` ([complexity.py:111-119](../../src/audrey/pipeline/complexity.py#L111-L119)).
Its signature alone tells you a lot: `(messages: list[dict]) -> bool` — hand it
a conversation, get back a yes/no. The body walks the messages backward to find
the latest `"user"` one, then asks whether that message's text starts with a
particular header string. Notice it reaches for the **same `_iter_text_parts`
helper** from last lesson: `next(_iter_text_parts(...), None)` grabs the *first*
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

## 4. Why this mattered

You can now read `complexity.py` end to end — and the three habits you built
across these lessons are the ones that carry to every file in the project:

- **A value can have more than one type, and good code asks rather than
  assumes.** `isinstance` is that question. The moment you see it, you know the
  code is handling a value whose shape isn't guaranteed — and the branches tell
  you exactly which shapes it expects.
- **`None` is information, not just "nothing."** Whether an absence means
  "count zero," "bucket under other," or "stop and return" is a decision each
  function makes on purpose. Reading that decision is reading the code's intent.
- **A signature is a contract you can read first.** `-> int`, `-> dict[str,
  int]`, `-> tuple[bool, int]` tell you what comes back before you read a line
  of the body; `messages: list[dict]` tells you what to hand in; a bare `*`
  tells you which arguments must be named. Glancing at the header before the
  body is the single fastest way to orient in unfamiliar code.

Put together, that's the difference between staring at a wall of plausible
lines and *reading* — knowing, at a glance, what each name holds and what each
function promises.

---

## 5. Where we're headed next

So far every value we've read has been loose — dicts with stringly-typed keys,
content that could be one of two shapes. That's how data looks at the *edge* of
the system, where it arrives from elsewhere. Deeper in, Audrey wants something
sturdier: values where the fields are fixed, named, and can't be misspelled. In
the next lesson we'll meet the tool for that — classes, and the lightweight
`@dataclass` Audrey leans on — by reading a real one (`AuthedUser`, the little
bundle that says who's making a request).

Keep the REPL handy. We'll keep poking at the real objects, live.
