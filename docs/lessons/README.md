# Audrey AI 2.0 — Codebase Lessons

A 13-lesson course that walks the entire Audrey codebase end-to-end,
starting from "what tools is this codebase built with" and ending at
"I could extend this myself."

This is a **personal learning track** — written for the codebase
author (Bart) to consolidate knowledge after shipping Phases 1-31.
It is published openly because the explanations may help anyone trying
to understand a similar multi-model orchestration pattern.

## How the course is structured

Lesson 1 is a **foundations primer** — what async, FastAPI, Pydantic,
LangGraph, and the other core libraries are, in the abstract. Read it
first so the rest of the course can refer back rather than
re-explain.

From Lesson 2 onward, each lesson is one focused area of the codebase
with four parts:

1. **Context** — why this piece exists, what problem it solves, where it
   sits in the system.
2. **Read-along** — the relevant files with line-number callouts to the
   load-bearing parts. Not a full code reproduction; the lesson points at
   what to notice.
3. **Audit notes** — anything I'd flag for change, with severity tag
   (`nit`, `consider`, `should-fix`, `bug`). The author decides what to
   act on; nothing changes without approval.
4. **Comprehension questions** — 3-5 per lesson. Some are warm-ups; some
   surface gaps.

Python proficiency assumed: "I can read def, classes, and control
flow." Anything more advanced (async, type hints, decorators, context
managers) gets introduced in Lesson 1.

## Prerequisites

- The Audrey repo (you're in it)
- Python 3.12 + `uv sync --extra dev` (so you can run `pytest`)
- Comfort with reading code and running shell commands

## Lessons

### Part I — Foundations

1. [Foundations: the tools you'll meet in this codebase](lesson-01-foundations.md) —
   `async`/`await`, FastAPI, Pydantic, LangGraph, type hints,
   context managers, httpx, vector search, Prometheus, pytest. The
   shape of each, why it exists, where you'll see it in Audrey.
2. [The request lifecycle, end-to-end](lesson-02-request-lifecycle.md) —
   one request from OWUI to the user, all the way through. Touches
   every major component without going deep on any.
3. **Configuration + startup** *(coming next)* — `main.py`, `config.py`,
   `compose.yaml`. How the app boots and what gets read where.
4. **The model layer** *(coming)* — `models/`. The Ollama client, the
   model registry, the health tracker.

### Part II — The pipeline

5. **Classify + complexity** *(coming)*
6. **Memory recall + datetime injection** *(coming)*
7. **Fast path + ReAct** *(coming)*
8. **Deep panel + synthesis** *(coming)*
9. **Reflection + retry** *(coming)*
10. **Fair scheduling + in-flight cap** *(coming)*

### Part III — Knowledge base + tools

11. **KB ingest + storage** *(coming)*
12. **Tools dispatch + custom-tools server** *(coming)*

### Part IV — Routes, observability, ops

13. **Routes + auth + metrics** *(coming)*

## What you'll know at the end

- Every line of every Python file in `src/audrey/` and `tools-server/`,
  in context. ~9,800 LOC total.
- How async + FastAPI + LangGraph fit together in this specific
  codebase. (Not a general tutorial on those — concrete-first.)
- How the streaming SSE protocol works for OpenAI-compatible APIs.
- How Qdrant indexes vectors and how CLIP + nomic-embed-text get used.
- How fairness is enforced across users on a single GPU.
- Where to look first when something breaks.

## What this course will NOT cover

- Container infra (Unraid, Docker, Cloudflare Tunnel) — see
  [`../campaign-1/`](../campaign-1/) phase docs for that.
- Frontend integration (Open WebUI) — black box from Audrey's side.
- Model training, embedding theory, etc. — out of scope.
