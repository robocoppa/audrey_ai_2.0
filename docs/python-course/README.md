# Learn Python by Building a Real AI Server

A course for people who can already write basic Python and want to cross the
gap into reading and changing **real** code — taught by reading and extending
**Audrey**, a self-hosted AI server that routes chat requests across a stable
of large language models.

Most tutorials leave you stuck at a wall: you can write a script, and then you
open a real codebase and understand nothing. This course is built to get you
over that wall. Instead of toy examples, every new concept is introduced
*inside* real production code — the actual files Audrey runs on — so the leap
from "I can write a script" to "I can work in a real application" happens by
practice, not hand-waving.

By the last lesson you'll be able to open any file in this project, read
it, and make a small change with confidence.

## Who this is for

You can already write basic Python. You know variables, `if`/`else`, loops,
and functions, you've run a script, and an error message doesn't frighten
you. What you *don't* yet have is the next tier — classes, type hints,
decorators, context managers, `async` — and, more importantly, the ability
to open a real production codebase and understand what you're looking at.

That gap is exactly what this course closes. It takes you from **"I can write
a script" to "I can read and extend a real application."** We don't re-teach
`if` statements; we move fast through what you know and spend our time where
the real difficulty lives.

If you've genuinely never coded, start with any free "intro to Python"
resource first — this course assumes you're past day one.

## How the course is shaped

The course is in four parts. It opens the real Audrey codebase almost
immediately — after one brisk lesson that sharpens the Python you already
know, you start reading actual production files. From there, every new
concept is taught by pointing at where Audrey uses it, building up to the
point where you can trace a whole request through the system and change it.

- **Part 1 — Get in and oriented (Lessons 1–6).** Stand up a real dev
  environment, sharpen the fundamentals you already have, and read your first
  real Audrey files — small ones, fully understood.
- **Part 2 — The shapes production code is made of (Lessons 7–12).** The next
  tier you don't have yet, each taught through Audrey's own code: classes and
  dataclasses, type hints, modules and imports, error handling, comprehensions
  and iteration patterns, decorators and caching.
- **Part 3 — The hard, important parts (Lessons 13–16).** Context managers,
  `async`/`await`, the libraries Audrey leans on (FastAPI, Pydantic, httpx),
  and how its data flows from an HTTP request into typed pipeline state.
- **Part 4 — Read it, test it, change it (Lessons 17–19).** Trace one request
  end-to-end through the system, learn the test-and-lint workflow, and make a
  real change to Audrey as a capstone.

Each lesson follows the same rhythm:

1. **The question** — what are we trying to do, in plain language.
2. **In the wild** — the idea shown directly in real Audrey code, with
   paste-along REPL snippets so you're running things as you read.
3. **Concept spotlight** — the Python feature behind it, sharpened.
4. **Why this mattered** — what the lesson bought you, in a few lines.
5. **Where we're headed next** — a short lead-in to the following lesson.

## Lesson index

**Part 1 — Get in and oriented**
1. [Your environment and the project](lesson-01-setup.md)
2. [How a conversation becomes data](lesson-02-values-and-types.md)
3. [When a value has more than one shape](lesson-03-one-field-two-shapes.md)
4. [`None`, `or`, and falsy defaults](lesson-04-none-and-falsy-defaults.md)
5. [Reading a function's signature](lesson-05-reading-a-signature.md)
6. [Reading your first real files (navigating the repo)](lesson-06-navigating-the-repo.md)

**Part 2 — The shapes production code is made of**
7. Classes and dataclasses
8. Type hints for real
9. Modules, imports, and how a package fits together
10. Errors and exceptions: failing on purpose
11. Comprehensions and iteration patterns
12. Decorators and caching

**Part 3 — The hard, important parts**
13. Context managers and the `with` statement
14. `async` and `await`
15. The libraries Audrey is built on
16. From HTTP request to typed pipeline state

**Part 4 — Read it, test it, change it**
17. Reading one request end-to-end
18. The test-and-lint workflow
19. Make a real change (capstone)

## What is Audrey, briefly?

Audrey is a program that sits between a chat box (in your browser) and a
set of AI models, and decides — for every message — *which* model should
answer, whether it needs to search the web or a knowledge base first, and
whether the answer is good enough to send back. It runs on one computer at
home.

You won't run the *full* system (it needs GPUs and models). But from Lesson 1
you'll run its real test suite on your own machine, and throughout the course
you'll read, run, and modify its actual code. You'll learn what each part does
as we reach it. For now, just know it's **real** — deployed and in daily use —
which is exactly why it's worth learning from.

## Prerequisites

- A computer (Windows, macOS, or Linux).
- Willingness to type code yourself instead of copy-pasting. You learn by
  typing.

Everything else — installing Python, the editor, the tools — is covered in
Lesson 1.

## Where to start

Begin with [Lesson 1 — Your environment and the project](lesson-01-setup.md).
Work through the lessons in order; each one ends with a pointer to the next.
