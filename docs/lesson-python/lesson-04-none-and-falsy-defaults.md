# Lesson 4 — `None`, `or`, and falsy defaults

**Estimated time:** 20–30 minutes.

**Goal:** To read the small, deliberate decisions about *missing values* that
production code is full of. Last lesson you learned to handle a value whose
*type* can vary; this one is about the value that isn't there at all. You'll
learn the `m.get(key) or default` idiom and the sharp edge hiding inside it,
why `.get(key, default)` is a different tool for a different job, and how `None`
gets used not as an error but as a *signal* the code reads on purpose.

We're still inside
[`src/audrey/pipeline/complexity.py`](../../src/audrey/pipeline/complexity.py).
Keep the REPL from last lesson open (or start a fresh one: `cd` into the project
folder and run `uv run python`).

---

## 1. The `or "other"` idiom

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

First, one piece of background that makes the rest make sense.
`count_tokens_by_role` is a **diagnostic** — a measuring tape pulled out only
during debugging. It runs when an operator turns on a debug flag and emits a
single log line, something like `complexity.breakdown: user=412 assistant=88
system=30`, so that when you're asking "why did this request take the slow
path?" you can see exactly *where* a conversation's tokens came from. And it
measures whatever `messages` actually arrived — from Open WebUI, other apps,
scripts — none of which Audrey can assume are perfectly formed. A buggy client
or a half-built request can hand it a message with no `"role"` key at all. That
possibility is the whole reason the next line is shaped the way it is.

Now the idiom. `m.get("role") or "other"` pairs two everyday Python moves, and
now you have the background to read it. `m.get("role")` returns the role string
if it is there — usually `"user"`, `"assistant"`, `"system"`, or sometimes
`"tool"` in tool-using conversations — or `None` if the key is missing.
Then the `... or "other"` half takes over, and *that* half is one of
the most common fallbacks in Python: `a or b` evaluates to `a` when `a` is
"truthy," otherwise to `b`. So the whole expression means: *"use the role if
there is a real one; otherwise fall back to `"other"`."* It's the "missing →
default" pattern in a single line, and you'll see it everywhere.

Why is `or` the tool for that, rather than an `if`? Because the alternative is
three lines for something that's really one thought:

```python
role = m.get("role")          # might be None
if role is None:
    role = "other"
```

`role = m.get("role") or "other"` says the same thing in one line, and reads
left-to-right the way you'd say it out loud: *the role, or "other" if there
isn't one.* That compression is exactly why it's everywhere — once your eye
learns the shape, `x or default` reads as a single unit meaning "x, with a
fallback," and you stop parsing it as two separate operations.

---

## 2. The one case `or` gets wrong — and how Audrey guards for it

`m.get("role") or "other"` is the right tool, and you should reach for `x or
default` freely — but it has one blind spot worth knowing, because Audrey has a
line written specifically to dodge it. The blind spot comes from what `or`
actually checks: **it doesn't test whether a value is *there*, it tests whether
the value is *falsy*.** And several perfectly real values are falsy — not just
`None`, but also `0`, `""` (empty string), `False`, and `[]` (empty list). So
`or` can't tell "the value is missing" apart from "the value is present, and it
happens to be `0` or empty." It sends both to the fallback.

Almost always, that's fine — because almost always, an empty value *means* the
same thing as a missing one. A blank-string role is no more useful than no role,
so `count_tokens_by_role` is happy to send both to `"other"`. Reaching for `or`
there is correct, and reaching for something fussier would just be noise.

The idiom only breaks when an empty value is a **real, deliberate choice** rather
than "nothing was given." Audrey lets a request specify a **temperature** — the
dial controlling how random the model's wording is. Turn it up and the model
gets creative; set it to `0` and the model is fully deterministic, picking the
most likely word every time. A temperature of `0` isn't "no temperature." It's a
user saying, on purpose, *give me the same answer every time.* That's the trap
condition: a falsy value (`0`) that the user meant.

Watch what the idiom would do to it:

```python
temperature = request.get("temperature") or 0.7   # looks fine, isn't
```

Read aloud, that says "use the requested temperature, or `0.7` if none was
given" — and for almost every request it's correct. But the user who explicitly
asked for `0` gets `0.7` instead, because `0` is falsy and `or` can't tell "they
asked for zero" from "they asked for nothing." Their deterministic request
silently becomes a creative one. No crash, no warning — the code just quietly
overrode a deliberate setting. *This* is the case worth recognizing: the idiom
works for every request except the one that matters most here.

So Audrey doesn't use `or` for this. The real line that applies a request's
temperature ([graph.py:552](../../src/audrey/pipeline/graph.py#L552)) steps off
the idiom and tests for *absence* directly:

```python
if (t := state.get("temperature")) is not None:
    options["temperature"] = t
```

`is not None` asks the exact question the situation needs — "was a temperature
actually given?" — so a real `0.0` passes straight through and only a genuinely
missing value is skipped. That `is not None` is the considered exception to a
good default: the author reached past the clean idiom *because* this is the one
field where a falsy value is meaningful, and left this slightly longer line as
the marker of that decision.

The rule that falls out: keep reaching for `or` — it's right almost every time.
But when a value's `0`, `""`, or `False` could be a real answer, reach for the
precise tool instead: an explicit `is not None`, or `.get(key, default)`, which
substitutes **only** when the key is truly absent. You've already seen the second
one, one line below the role idiom — `totals.get(role, 0)` defaults to `0` only
for a role not yet seen. Two adjacent lines, each using the tool whose rule fits
the value it guards.

---

## 3. `None` as a signal

Step back to that missing role one more time, because it's an example of
**`None` used as a signal.** `None` isn't an error here; it's information —
"this message didn't say what role it is." So why bucket it under
`"other"` rather than crash, or just skip it? Because of what this function is
for. Crashing is out of the question — a debug log line should never be able to
take something down. But *skipping* the message would be quietly worse: the
per-role numbers would stop adding up to the true total, so the diagnostic would
lie to the very person leaning on it to debug. Bucketing under `"other"` keeps
the books balanced — every message's tokens land *somewhere* — and a nonzero
`other=` in the logs becomes its own clue: *something is sending us messages
with no role.* That's the real-world payoff. The breakdown you're squinting at
while a request misbehaves is trustworthy, and that odd bucket is a lead, not a
hole in the data.

Notice the contrast with last lesson: there, an unreadable *content* meant
"count nothing" (the helper yields no text); here, a missing *role* means
"bucket it under other." `None` shows up in both, and each function decides for
itself what the absence should mean — the same "missing value" gets two
different, intentional treatments, each fitting its own job.

---

## 4. Why this mattered

Missing values are everywhere in code that talks to the outside world, and how
a function handles them is one of the clearest windows into what it's *for*.
Three habits carry out of this lesson:

- **`x or default` is the one-line "missing → default."** When you see it, read
  it as a single unit: "x, with a fallback." It's the most common way Python
  fills in a value that might not be there.
- **Reach for `or` freely — but know its one blind spot.** `or` tests *falsy*,
  not *missing*: it can't tell an absent value from a real `0`, `""`, or `False`.
  Almost always that's fine; it's a bug only when a falsy value is a deliberate
  choice (a requested `temperature` of `0`). For those, test for absence directly
  — `is not None`, or `.get(key, default)`.
- **`None` is information, not just "nothing."** Whether an absence means "count
  zero," "bucket under other," or "stop and return" is a decision the function
  makes on purpose. Reading that decision is reading the code's intent.

Put together, that's the difference between glancing past a `.get(...) or ...`
as boilerplate and *reading* it — seeing the deliberate choice the author made
about what to do when the data doesn't cooperate.

---

## 5. Where we're headed next

You've now read how `complexity.py` handles values, all the way down to the
ones that aren't there. The other half of reading code fluently is reading its
*functions* — and you can do a surprising amount of that from the first line
alone, before the body. In the next lesson we'll stay in this same file and
learn to read a function's **signature**: the `: list[dict]` and `-> int`
annotations that tell you what a function eats and returns, the bare `*` that
forces arguments to be named, and the `__all__` list that says which names a
module means for the outside world. By the end you'll read a whole function
you've never seen, cold.
