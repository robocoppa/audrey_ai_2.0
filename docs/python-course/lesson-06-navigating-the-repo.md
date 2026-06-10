# Lesson 6 — Reading your first real files: navigating the repo

**Estimated time:** 20–30 minutes.

**Goal:** To stop reading single files and start reading the *project*. So far
you've read two files — `scheduling.py` and `complexity.py` — line by line. This
lesson zooms out: you'll learn to turn a name like `audrey.pipeline.complexity`
into a file on disk, read a folder of code as a map of what the app does, and
find the code behind any behavior yourself — without anyone handing you a line
number.

Keep the REPL from last lesson open. You'll also want Cursor's file explorer
visible (the sidebar showing the folder tree) — this lesson moves between the two.

---

## 1. Folders are packages, files are modules

Open the `src/audrey/` folder in the explorer. You'll see a mix of loose `.py`
files and folders:

```text
src/audrey/
├── __init__.py
├── main.py
├── config.py
├── auth.py
├── scheduling.py
├── metrics.py
├── kb/
├── models/
├── pipeline/
├── routes/
└── tools/
```

Two words cover everything here. A **module** is a single `.py` file — `auth.py`
is a module, `scheduling.py` (which you already read) is a module. A **package**
is a folder of modules grouped because they belong together — `pipeline/` holds
the orchestration code, `routes/` holds the web endpoints, `kb/` holds the
knowledge-base machinery. That's the whole organizing idea: related modules live
in a folder, and the folder's name tells you what they're about.

There's one piece of bookkeeping that makes a folder count as a package. Look
inside any of those folders and you'll find a file named `__init__.py`:

```text
src/audrey/pipeline/
├── __init__.py     ← this marks the folder as a package
├── complexity.py
├── graph.py
├── classify.py
└── …
```

Here's the thing worth noticing: in Audrey, those `__init__.py` files are
**empty**. Every one of them — `pipeline/__init__.py`, `routes/__init__.py`, and
the rest — is a zero-byte file. They're empty *on purpose*. The file doesn't need
to contain anything; its mere presence is the signal that says "this folder is a
package, not just a folder of loose scripts." Think of it as a flag planted in
the folder. *Why* Python needs that flag, and the more advanced things an
`__init__.py` can do when it isn't empty, is a topic of its own — we'll get to it
in the modules-and-imports lesson. For now: empty `__init__.py` = "this folder is
a package," and that's all you need.

The one exception is the package's *top-level* `__init__.py`
([\_\_init\_\_.py:1](../../src/audrey/__init__.py#L1)), which holds a single line:

```python
__version__ = "7.0.0"
```

That's Audrey's version number, kept in one place so the rest of the code can
read it. You'll see exactly who reads it in a minute.

---

## 2. A dotted name is a path

Back in Lesson 5 you ran this line in the REPL:

```python
from audrey.pipeline.complexity import is_owui_task_request
```

You used it without thinking about where `audrey.pipeline.complexity` *is*. It's
worth slowing down, because that dotted name is one of the most useful things to
be able to read at a glance. **The dots are folders, and the last name is the
file.** Walk it left to right:

- `audrey` → the package folder `src/audrey/`
- `.pipeline` → the subfolder `pipeline/`
- `.complexity` → the file `complexity.py` inside it

So `audrey.pipeline.complexity` *is* `src/audrey/pipeline/complexity.py` — the
file you read across the last two lessons. The dotted name and the file path are
the same address written two ways, and translating between them is mechanical.
Once you can do it, any import line in the codebase tells you exactly which file
to open.

You don't have to take that on faith — Python will tell you the file behind any
imported name. Try it:

```python
import audrey.pipeline.complexity as c
print(c.__file__)
```

It prints the full path to `complexity.py` on your disk. Every module carries a
`__file__` attribute holding its own location, so when you're unsure where an
import actually lands, importing it and printing `__file__` settles it
immediately.

---

## 3. `main.py`'s imports are a map of the whole app

Now point that skill at the file where Audrey starts: `main.py`. Open it and look
only at the block of imports near the top
([main.py:20-41](../../src/audrey/main.py#L20-L41)). A sample:

```python
from audrey import __version__
from audrey.config import get_config
from audrey.kb.qdrant import QdrantKB
from audrey.models.registry import ModelRegistry
from audrey.pipeline.graph import build_graph
from audrey.routes.openai import router as openai_router
from audrey.tools.discovery import ToolRegistry, discover_all
```

You can now read that as a map. Each line names a package, and every package in
the project shows up: `config` (settings), `kb` (the knowledge base), `models`
(the model layer), `pipeline` (the orchestration), `routes` (the web endpoints),
`tools` (the tool registry). Before reading a single function body, the import
block has told you what pieces the app is built from and roughly what each one
does. That first line — `from audrey import __version__` — is who reads the
version string you saw in §1.

This is the payoff of §2: each of those names is a file you can open. `from
audrey.kb.qdrant import QdrantKB` says "there's a class called `QdrantKB`, and it
lives in `src/audrey/kb/qdrant.py`." When you want to know how Audrey talks to its
vector database, you now know exactly which file to open — the import told you.

You don't need to understand what any of these *do* yet; each gets its own lesson.
The point is narrower and more useful: **`main.py`'s import block is a table of
contents for the whole codebase**, and you can read it.

---

## 4. Finding code without a line number

So far every file you've read came with a pointer — "open `complexity.py`,
line 78." Real work doesn't. You notice a *behavior* and have to find the *code*.
Two tools cover almost all of it, and you've just met the first.

**Follow the import trail.** When you can read a dotted name as a path, you can
chase any name to its source. You see `build_graph` used in `main.py`, you read
its import (`from audrey.pipeline.graph import build_graph`), and you know to open
`src/audrey/pipeline/graph.py`. One hop, no guessing.

**Search the text.** When you don't even have a name yet — just a string you saw,
or a word for the behavior — search the code. Say you remember that Audrey treats
a message starting with `### Task:` specially (you met this in Lesson 5's
`is_owui_task_request`), and you want to find where that prefix actually lives.
Search for the literal text across the source:

```text
grep -rn "Task:" src/audrey/
```

(Or use Cursor's project-wide search — the magnifying glass in the sidebar — and
type `Task:`. Same result, nicer to read.) It points you straight at
`pipeline/complexity.py`, where the prefix is defined as a constant. You started
from a behavior you remembered and landed on the exact line, with no one handing
you a location.

Those two moves — chase a name through its import, or search for a word — are how
you navigate a codebase you don't have memorized. Which is every codebase, at
first.

---

## 5. Why this mattered

You can now get around the whole project, not just read one file at a time:

- **A dotted name is a file path in disguise.** `audrey.pipeline.complexity` is
  `src/audrey/pipeline/complexity.py` — dots are folders, the last name is the
  file. Translating between them is mechanical, never a guess.
- **`main.py`'s imports are a map.** The import block names every package and
  what it's for, so you can read the shape of the app before any function body —
  and each name points at the file to open next.
- **Finding code is a skill, not a handout.** Follow an import trail to a name's
  source, or search the text for a word or string. Between the two, you can find
  the code behind any behavior on your own.

That's the last piece of getting oriented. You've gone from "I can run a script"
to "I can open this real project, find my way to any part of it, and read what's
there."

---

## 6. Where we're headed next

That closes Part 1. You have a working setup, the Python reading habits from
`scheduling.py` and `complexity.py`, and now a map of the whole repo. Part 2
turns to the *shapes* production code is built from — the next tier of Python you
came here for. It opens with **classes and dataclasses**: the tool Audrey reaches
for when it wants a value with fixed, named fields instead of a loose dict. We'll
meet it by reading a real, small one — `AuthedUser`, the little bundle that says
who's making a request — and you'll finally see what's behind a line like
`me.email` that's been waiting in the wings.
