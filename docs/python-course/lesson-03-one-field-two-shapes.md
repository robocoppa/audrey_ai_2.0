# Lesson 3 — When a value has more than one shape

**Estimated time:** 25–35 minutes.

**Goal:** in the last lesson you learned that a message is a dict and that
`.get` is a deliberate "this might be missing" choice. Now we meet a value whose
*type isn't fixed* — sometimes a string, sometimes a list — and the way real
code copes with that: it checks the type, then handles each shape on its own
terms. By the end you'll understand exactly why one small helper in
`complexity.py` is shaped the way it is, and recognize the same move anywhere
you see it.

We're still inside
[`src/audrey/pipeline/complexity.py`](../../src/audrey/pipeline/complexity.py),
picking up exactly where Lesson 2 left off. Keep the REPL from last lesson open
(or start a fresh one: `cd` into the project folder and run `uv run python`).

---

## 1. One field, two shapes

Here's the part that separates "I know what a dict is" from "I can read this
function." In Lesson 2 we read `_count_message_tokens`, whose loop walked over
`_iter_text_parts(m.get("content"))` — and we deferred the helper. This is it.
That helper is where the real shape-handling lives
([complexity.py:43-59](../../src/audrey/pipeline/complexity.py#L43-L59)):

```python
def _iter_text_parts(content: object) -> Iterable[str]:
    """Yield the text of a message's `content`, in order."""
    if isinstance(content, str):
        yield content
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                yield part["text"]
```

The key realization, and the reason this helper exists at all: **`content` does
not have one fixed type.** Sometimes it's a plain string — an ordinary text
message. But it can also arrive as a *list of parts*, where each part is its own
little dict like `{"type": "text", "text": "describe this"}`. One field, two
possible shapes.

Where does that second shape come from in the real world? Audrey speaks the
**OpenAI Chat Completions format** — the de-facto standard that chat clients
(Open WebUI, and the apps and bots that talk to Audrey) send and that the whole
LLM ecosystem agrees on. In that format every message has a `content` field,
and originally it was always a string. Then images happened: people wanted to
send a picture *and* a question about it in the same turn ("what's in this
photo?"). The standard couldn't just add a separate `image` field without
breaking every text-only client, and it couldn't make `content` two different
types depending on the day. So it widened `content` to *either* a plain string
(the common case) *or* a list of typed parts — a `{"type": "text", ...}` part
for the words, an image part for the picture. The shape you're looking at is the
fingerprint of a real product decision: **one field that had to keep working
for plain text while also growing to carry an image the user attached in the
chat box.**

That history is exactly why a helper like this earns its place. The caller —
`_count_message_tokens` from Lesson 2 — just wants *the text*, to count tokens.
It shouldn't have to know or care which `content` shape the client happened to
send; it needs one clean stream of strings either way. And guessing wrong would
break things: treat a `list` as if it were a string and you'd crash; count an
image part's raw bytes as text and you'd get a meaningless number. So
`_iter_text_parts` is the *seam* where that variability gets absorbed — whatever
shape came in, only real text comes out, one piece at a time — and every
function past it gets to pretend `content` was simple. Absorbing the outside
world's irregularity at one chokepoint, so the rest of the code stays clean, is
the job of a great many small helpers in a real codebase.

(One honest note, because reading real code means noticing these: today
Audrey's request schema actually pins `content` to a plain string, so the
list-of-parts branch below isn't exercised on the live path *yet*. It's written
to the full standard on purpose — the day Audrey accepts image messages, this
helper already copes, and nothing downstream has to change. Writing to the
contract, not just to today's input, is a deliberate move you'll see often.)

---

## 2. How the code asks: `isinstance`

To pull only the text out, the helper first has to figure out which shape it's
looking at — and a value whose type can vary is normal in real code; you can't
just assume. So the code **asks**:

> **Concept spotlight: `isinstance`.** `isinstance(content, str)` asks "is
> `content` a string?" and answers `True`/`False`. `isinstance(content, list)`
> asks "is it a list?" You branch on the answer. This pattern — *check the type,
> then handle each shape on its own terms* — is called **type-narrowing**:
> after the `if isinstance(content, str):` line, inside that block, you and the
> reader (and the editor) all know `content` is definitely a string, so using
> it as a string is safe. The helper handles three cases: it's a string → hand
> it back as the one piece of text; it's a list → walk the parts and hand back
> each text one; it's *neither* → fall off the end having handed back nothing.

Read the list branch slowly, because it stacks the same idea twice. The outer
`isinstance(content, list)` narrows `content` to a list. Then, for each `part`,
the inner `isinstance(part, dict) and isinstance(part.get("text"), str)`
narrows *again*: only take this part if it's a dict **and** its `"text"` key
holds a string. Anything that doesn't fit that shape — an image part, a
malformed part — is silently skipped. Image bytes have no meaningful token
count here, so they contribute nothing.

> **What's `yield`?** You'll have noticed `yield` where you expected `return`.
> The short version for now: `yield` hands a value back to the loop that's
> walking this helper, one at a time, then picks up where it left off to hand
> back the next — which is exactly why the caller could write `for text in
> _iter_text_parts(...)`. A function that yields like this is a *generator*, and
> it gets a proper lesson of its own later. For today, read it as "this helper
> produces a sequence of text pieces," and don't sweat the mechanics.

And the case where `content` is *neither* a string nor a list — say it was
missing entirely and `.get` gave back `None`: the helper simply yields nothing,
so the counting loop in `_count_message_tokens` runs zero times and the count
stays `0`. The function doesn't crash, doesn't guess — it treats an unreadable
message as "contributes nothing to the count." That's a deliberate, defensive
default, and it's only safe *because* this is a rough gate, not an exact
accounting.

---

## 3. Try it

Paste these one at a time and watch the type checks answer. Is a string a
string?

```python
isinstance("hello", str)
```

Prints `True`. Is that same string a list?

```python
isinstance("hello", list)
```

Prints `False`. Now build a text part like the ones inside a multimodal
`content` list, and run the exact check the helper uses on it:

```python
part = {"type": "text", "text": "describe this"}
```

(Silent — it's an assignment.)

```python
isinstance(part, dict) and isinstance(part.get("text"), str)
```

Prints `True` — it's a dict, and its `"text"` holds a string. Now an image
part, which has no `"text"` key:

```python
bad = {"type": "image", "url": "..."}
```

```python
isinstance(bad, dict) and isinstance(bad.get("text"), str)
```

Prints `False`. That last `False` is exactly how `_iter_text_parts` skips an
image part: `.get("text")` returns `None`, `None` is not a `str`, so the `if`
is false and that part is never yielded.

---

## 4. Why this mattered

One small helper, but it taught the move you'll use most when reading real code:

- **A value can have more than one type, and good code asks rather than
  assumes.** `isinstance` is that question. The moment you see it, you know the
  code is handling a value whose shape isn't guaranteed — and the branches tell
  you exactly which shapes it expects.
- **Irregularity gets absorbed at a seam.** `_iter_text_parts` exists so the
  messy "string-or-list" reality of `content` is dealt with in *one* place, and
  every function past it gets to pretend `content` was simple. When you meet a
  small, oddly-specific helper, ask what irregularity it's there to hide — the
  answer usually explains the whole design.

Hold onto that `isinstance`-then-branch habit; the next lesson uses it again
while it adds the other half of reading a function fluently — its signature.

---

## 5. Where we're headed next

You've been reading function *bodies* to figure out what they do. The next
lesson shows you the shortcut: a function's **signature** — the `: list[dict]`
and `-> int` annotations on its very first line — often tells you what it eats
and returns before you read a single line inside. We stay in `complexity.py`,
look at how `None` becomes a deliberate fallback (`m.get("role") or "other"`),
read the annotations on the functions you've already met, and finish by reading
a whole function you've *never* seen, cold.

Keep the REPL handy. We'll keep poking at the real objects, live.
