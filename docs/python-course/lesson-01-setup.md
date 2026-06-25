# Lesson 1 — Your environment and the project

**Estimated time:** 20–30 minutes, almost all of it one-time setup.

**Goal:** stand up a clean, modern Python development environment — the same
one a professional would use — and point it at the real Audrey codebase, so
that by the end of this lesson you can run Audrey's own test suite on your
machine. You already know how to write Python; this lesson is about the
tooling that the rest of the course (and real work) assumes.

**One path, on purpose.** There are a dozen ways to set up Python. To keep
the course consistent we commit to one good combination — **Cursor** as the
editor, **uv** for Python and packages — and walk it exactly. If you already
have a setup you like, adapt freely; the only hard requirement is Python 3.12+
and the ability to run `pytest`. The lesson branches only where your OS forces
a different command.

---

## 1. Cursor

We use **Cursor**, a free editor built on VS Code, so the course can give you
exact menus and shortcuts instead of vague "somewhere in your editor"
directions. If you're already reading this in Cursor, or you prefer VS Code,
you're set — everything here works identically in both.

Install it from <https://cursor.com/> (`.exe` on Windows, `.dmg` on macOS,
`.AppImage` on Linux). Then add the **Microsoft Python extension**: open the
Extensions panel (`Ctrl+Shift+X`, or `Cmd+Shift+X` on Mac), search `Python`,
install the one from Microsoft. It gives you the inline error squiggles and
"go to definition" that make reading a real codebase bearable.

From here, everything happens inside Cursor — including the terminal. Open it
with **View → Terminal** or ``Ctrl+` `` (``Cmd+` `` on Mac).

---

## 2. uv

> **What is uv?** A single, fast tool that does the two jobs every Python
> project needs: it **installs and pins Python versions**, and it **manages a
> project's dependencies in an isolated environment** so one project's
> packages can't break another's. It replaces the old `pyenv` + `venv` + `pip`
> + `pip-tools` juggling act with one command that's fast enough you stop
> thinking about it. We use it because Audrey itself is a uv project — its
> `pyproject.toml` and `uv.lock` are what uv reads — so the commands you learn
> here are the exact ones the project ships with.

Install uv. Paste the line for your OS into Cursor's terminal:

- **macOS / Linux:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Windows** (PowerShell):
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

Close and reopen the terminal (the installer edits your `PATH`, which only a
fresh shell picks up), then confirm and install the Python this project
targets:

```bash
uv --version
uv python install 3.12
```

Audrey requires Python 3.12+. uv keeps that version isolated to the project,
so it won't disturb whatever Python your system already has.

---

## 3. Get the project running

Now the part that makes this course different from a syntax tutorial: you'll
run the actual server's test suite. First, decide *where* the project should
live and move there — `git clone` drops the repo into whatever directory your
terminal is currently in, so pick a parent folder you'll remember rather than
cloning blind:

```bash
cd ~/code          # or wherever you keep projects; make it first if needed
git clone https://github.com/robocoppa/audrey_ai_2.0.git
cd audrey_ai_2.0
uv sync --extra dev
```

(Heads-up: `uv sync` pulls PyTorch and the CUDA libraries, so expect a one-time
download of several GB — no GPU required to install or test.)

(If `~/code` doesn't exist yet, `mkdir -p ~/code` first; on Windows use a path
like `~\code`.) After cloning, point Cursor at the repo: **File → Open
Folder…** → `audrey_ai_2.0`, so the editor and its terminal both work inside
the project from here on.

`uv sync` reads `pyproject.toml` + `uv.lock` and installs *exactly* the
dependency versions the project was built and tested against, into a project
-local `.venv/`. The `--extra dev` part pulls in the development-only extras —
chiefly `pytest` (the test runner) and `ruff` (the linter) — which aren't
needed to *run* Audrey but are needed to *work on* it.

When it finishes, run the suite:

```bash
.venv/bin/pytest tests/ -q
```

(On Windows the path is `.venv\Scripts\pytest`.) You should see a wall of
dots and a green summary line — every test passing, in well under a second.

That green line is the whole point of this lesson. The tests are **hermetic**:
they exercise Audrey's real logic — fair scheduling, auth, the routing gate —
without needing the GPU, the models, or any network. So your laptop is now a
place where you can change Audrey's code and instantly know whether you broke
anything. That feedback loop is what the back half of this course is built on.

> **If the tests fail to *collect*** (errors before any test runs), it's
> almost always the environment, not the code — usually a Python older than
> 3.12 or an incomplete `uv sync`. Re-run `uv sync --extra dev` and confirm
> `uv run python --version` reports 3.12+.

---

## 4. The two ways you'll run code

Throughout the course you'll run Python two ways, and it's worth being
deliberate about which:

- **A script** — `uv run python some_file.py`. The `uv run` prefix guarantees
  the code executes inside the project's `.venv`, so Audrey's dependencies are
  importable. Reach for this when you're building or running something real.
- **The REPL** — `uv run python` with no file drops you at an interactive
  prompt in that same environment. Reach for this to *interrogate* the
  codebase: import a module, call a function, inspect what it returns. We'll
  use it that way often once we're inside Audrey — being able to poke a real
  function live is one of the fastest ways to understand it.

Quick sanity check that your environment can see the project — from the repo
root:

```bash
uv run python -c "import audrey; print('audrey importable')"
```

If that prints the message instead of an `ImportError`, your environment is
wired correctly and pointed at the real package.

---

## 5. Practice the workflow

1. **Run the suite and read the summary.** Run `.venv/bin/pytest tests/ -q`
   and note the count and the time. Then run it again with `-v` instead of
   `-q` — see how the output changes from dots to named tests. Which would you
   want while debugging one failing test, and which for a quick "is anything
   broken?" check?

2. **Use the REPL to interrogate, not to compute.** Start `uv run python` and
   import something real:
   ```python
   >>> from audrey.scheduling import ANON_USER_BUCKET
   >>> ANON_USER_BUCKET
   ```
   You don't know what that constant is *for* yet — that's fine. The point is
   that the REPL reaches straight into the live codebase. (We'll meet this
   exact value when we cover fair scheduling.)

3. **Make a test fail on purpose, then fix it.** Open any file under `tests/`,
   find an `assert` line, and change an expected value so it's wrong. Re-run
   the suite. Read how `pytest` reports the mismatch — the file, the line, the
   expected-vs-actual diff. Then undo your change and confirm green again.
   Getting comfortable reading a *failing* test report is worth more than any
   amount of reading passing ones.

---

## 6. Where we're headed next

You now have a professional Python environment that can run and test a real
application — and an editor and REPL pointed straight at its code. That's the
platform the rest of the course stands on.

Next we start *reading*. Lesson 2 opens the smallest real files in Audrey and
looks at how its data is actually shaped — the values and types that flow
through the system. You already know what an `int`, a `str`, and a `dict` are,
so we won't dwell on that. The interesting part is how a production codebase
*uses* them deliberately: why a message is a `dict` with particular keys, why
a constant like `ANON_USER_BUCKET` is a plain string, where `None` shows up
and what it signals, and how type annotations turn those choices into
documentation the editor can check. By the end of it you'll be reading real
Audrey functions and knowing, at a glance, what kind of thing each name holds.

Keep the REPL handy — we'll use it to poke at the very objects we're reading
about, live, instead of taking the prose's word for it.

Next: **Lesson 2 — Values and types, as the codebase uses them.**
