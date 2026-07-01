# Lesson 2 — How a conversation becomes data

**Estimated time:** 25–35 minutes.

**Goal:** open your first real Audrey files and learn to read them fluently.
You already know what an `int`, a `str`, and a `dict` are. The thing that
actually makes a production codebase hard to read isn't the types — it's the
*conventions*: which type was reached for, what was left out, and what a missing
value is meant to signal. This lesson starts with the most important shape in
the whole project — the chat message — and the small, deliberate decisions in
how the code pulls data out of it.

Keep a REPL open the whole way through. We'll poke at the very objects we're
reading about instead of taking the prose's word for it. To start one now:

1. In Cursor's terminal, make sure you're in the project folder — the
   `audrey_ai_2.0` directory you cloned in Lesson 1. (`cd` into it if you're
   not.)
2. Run `uv run python`.

You'll get a `>>>` prompt — that's the REPL, waiting for you to type Python.
Leave it running; every "Try it" box below means "type this at that `>>>`
prompt." (To leave the REPL later, type `exit()` or press `Ctrl+D`.)

---

## 1. The question

Before any of the code makes sense, you need one piece of background: **what a
conversation with an AI actually looks like inside the program.**

When you chat with an AI in a browser, it feels like a back-and-forth. In code,
it's much plainer. The whole conversation — everything said so far — is sent to
the server as a **list of messages**, and each message is a small **dict** (a
Python dictionary: a bag of labeled values, `{label: value, ...}`). A single
message looks like this:

```python
{"role": "user", "content": "what's the weather?"}
```

Two labels — Python calls them *keys* — carry the whole thing:

- **`"role"`** — *who said it.* Most messages are `"user"` (you, the
  human), `"assistant"` (the AI's past replies), or `"system"` (hidden setup
  instructions the app sends before you ever type). Tool-using turns can also
  include `"tool"` messages, which hold the result of a tool call. The role is
  how the model tells each kind of message apart.
- **`"content"`** — *what was said*: the actual text, like `"what's the
  weather?"`.

A whole conversation is just those dicts in a list, oldest first:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "what's the weather?"},
    {"role": "assistant", "content": "I can't check live weather, but…"},
    {"role": "user", "content": "ok, tell me a joke instead"},
]
```

That list — `messages` — is the single most important data shape in the whole
project. It's what arrives from the browser, what gets passed from function to
function, and what eventually goes to the AI model. Nearly every function you
read today takes a `messages` list and pulls things out of it. So when you see
a lone `m` in the code, picture one of those little `{"role": ..., "content":
...}` dicts — a single line of the conversation.

With that in hand, here's a real line from Audrey you'll read in a minute. `m`
is one message dict from that list:

```python
role = m.get("role") or "other"
```

You know the pieces — `m` is a dict, `.get` looks something up, `or` is `or`.
And yet, read cold, it raises questions a syntax tutorial never answers: *Why
`.get("role")` instead of `m["role"]`? What's the `or "other"` guarding
against? Why is a chat message a bare dict at all, instead of some fancier
`Message` object?*

That gap — knowing the syntax but not the *intent* — is what this lesson
closes. Every one of those choices encodes a decision about what the data is
and how it can go wrong. Learn to read the decision, and the
code stops being a wall of plausible-looking lines and starts telling you what
it expects.

We'll work mostly inside one file, [`src/audrey/pipeline/complexity.py`](../../src/audrey/pipeline/complexity.py). It's
short, it has no exotic machinery, and by the end of the lesson
you'll understand all of it.

---

## 2. The smallest real value: a typed constant

Open [`src/audrey/scheduling.py`](../../src/audrey/scheduling.py). The whole file:

```python
"""Shared scheduling constants."""

from __future__ import annotations

ANON_USER_BUCKET = "__anon__"

__all__ = ["ANON_USER_BUCKET"]
```

You imported `ANON_USER_BUCKET` blind in Lesson 1; here it is. It's just a
string. But notice the decisions packed into a four-line file:

- **It's a plain `str`, and it lives alone in its own module.** Two different
  parts of Audrey (the fair-GPU gate and the per-user request cap) need to
  bucket anonymous traffic under one shared key. If each defined its own
  `"__anon__"`, a typo in one would silently split the bucket. Defining it
  *once*, here, is a deliberate "single source of truth" move. The value being
  a humble string is the point — it doesn't need to be anything fancier.
- **The leading underscores in `"__anon__"`** are a convention, not syntax: a
  real user's id would never look like that, so the sentinel can't collide
  with a genuine value. You'll see this trick a lot — a deliberately
  unnatural-looking string used as a "this isn't real data" marker.
- **`__all__`** is a list of the names this module is willing to hand out. It's
  the module's public face: when something does `from audrey.scheduling import
  *`, only the names in `__all__` come along. It's documentation as much as
  enforcement — *"the one thing worth importing from here is
  `ANON_USER_BUCKET`."*

> **`from __future__ import annotations`** sits at the top of nearly every
> Audrey file. For now, read it as boilerplate that makes type annotations
> (the `: str` and `-> int` bits we're about to meet) cheaper and more
> flexible. We'll come back to *why* it's there when annotations get their own
> lesson; you don't need it to read today's code.

That's the gentlest possible real file. Now to one that does some work.

---

## 3. A message is a `dict` with known keys

Open [`src/audrey/pipeline/complexity.py`](../../src/audrey/pipeline/complexity.py). Its job: decide whether an incoming
request is "complex" enough to deserve Audrey's slow, thorough path instead of
the fast one. It decides by counting **tokens**.

> **What's a token, and why count them?** A language model doesn't read text
> letter by letter — it chops it into **tokens**, the small chunks it actually
> processes. A token is roughly a word-piece: short common words are one token
> (`"the"`), longer or rarer words split into several (`"tokenization"` →
> `token` + `ization`). As a rule of thumb, one token is about four characters
> of English. **Counting matters because tokens are the unit of everything that
> costs:** a model can only hold so many tokens at once (its "context window"),
> and the more tokens you send, the slower and pricier the response. So token
> count is Audrey's best cheap estimate of *how much work a request is*. This
> file uses it as a gate: a short prompt is probably a quick question (fast
> path); a long one — a big paste, a sprawling conversation — likely needs the
> slower, more thorough path. It's a rough proxy for "how much is being asked
> here," computed before any expensive model is involved.

Start at `_count_message_tokens` ([complexity.py:62-69](../../src/audrey/pipeline/complexity.py#L62-L69)):

```python
def _count_message_tokens(m: dict, enc: tiktoken.Encoding) -> int:
    role = m.get("role")
    n = 0
    for text in _iter_text_parts(m.get("content")):
        if role == "assistant":
            text = _strip_audrey_markup(text)
        n += len(enc.encode(text))
    return n
```

That `enc.encode(text)` is the tokenizer from the box above doing its work —
turning a string into the list of tokens, whose length is the count. Don't
worry about the loop or `_iter_text_parts` yet — that helper is the first thing
we open in the next lesson. For now, focus on the two `m.get(...)` calls, where
this function pulls the `"role"` and `"content"` out of one message dict —
exactly the shape from §1.

Why is a message a bare `dict` and not some custom object? Because Audrey gets
these messages as **JSON** from the outside world (the browser, other clients),
and a JSON object arrives in Python as a dict, for free. Keeping it a dict —
with string keys like `"role"` and `"content"` — means the code maps
one-to-one onto the data that came in over the network. There's no translation
layer to build or keep in sync. (`JSON` is just the standard text format
programs use to send structured data to each other; for now, read "JSON
object" as "a dict that travelled over the network.")

The interesting choice is `.get`. Two ways exist to pull a key out of a dict:

```python
m["content"]      # raises KeyError if "content" is missing
m.get("content")  # returns None if "content" is missing
```

`.get` was chosen here, and that choice is itself a statement: *"a message
might not have a `content` key, and that's not a crash-worthy event — the code
will handle the absence itself."* Square brackets would say the opposite:
*"this key must exist; if it doesn't, something is deeply wrong, blow up."*
Reading `.get` tells you the code is braced for messy input. That's almost
always the right posture at the edge of a system, where the data comes from
someone else.

**Try it.** Paste these into your REPL one at a time and watch what each does.
First, build a message dict to work with:

```python
m = {"role": "user", "content": "hi"}
```

Nothing prints — assigning a value is silent. Now read a key that exists:

```python
m.get("content")
```

Prints `'hi'`. And the other key:

```python
m.get("role")
```

Prints `'user'`. Now ask for a key that *isn't* there:

```python
m.get("nonexistent")
```

Nothing prints at all. That's because the result is `None`, and the REPL
doesn't echo `None`. Force it into view by printing it:

```python
print(m.get("nonexistent"))
```

Now you see `None`. That silence-on-`None` trips everyone once — get used to
it. Finally, the contrast: ask for the missing key with square brackets
instead of `.get`:

```python
m["nonexistent"]
```

This one *crashes*, on purpose, with a `KeyError: 'nonexistent'` and a few
lines of traceback. That's the whole difference: `.get` hands back `None` for a
missing key; `[]` raises. (Your `m` is unharmed — re-run any of the lines above
and it still works.)

---

## 4. Why this mattered

The idea under this lesson is load-bearing for everything that follows:

- **A chat with an AI is, in code, just a list of message dicts.** Once you
  picture `messages` — `{"role": ..., "content": ...}` dicts in a list — most
  of Audrey's pipeline stops being mysterious. Nearly every function you'll
  read takes that list and pulls things out of it.
- **How you read a key is a decision, not a detail.** `.get` says "this might
  be missing, and I'll cope"; `[]` says "this must exist, or crash." At the
  edge of a system, where data comes from someone else, coping is usually
  right — and now you can read that intent straight off the code.

Hold onto the picture of `messages`; the next lesson opens the same file's
trickiest function and shows what happens when a value's *type* — not just a
key — can't be taken for granted.

---

## 5. Where we're headed next

We left one thing on the table: that `_iter_text_parts` helper the token
counter looped over. It's the next thing we open, and it's there for a reason
you'll recognize immediately — a message's `content` isn't always a plain
string. The next lesson stays in `complexity.py` and tackles values whose
*shape* can vary, the `isinstance` checks that handle them, and how to read a
function's type annotations to know what it eats and returns before you read a
line of its body.
