# Audrey AI 2.0 — Codebase Lessons

A 15-lesson course that walks the entire Audrey codebase end-to-end,
starting from "what tools is this codebase built with" and ending at
"I could extend this myself."

This is a **personal learning track** — written for the codebase
author (Bart) to consolidate knowledge after shipping Phases 1-31.
It is published openly because the explanations may help anyone trying
to understand a similar multi-model orchestration pattern.

## How the course is structured

Lessons 1, 2, and 3 are **foundations primers** — Lesson 1 covers
Python language features (async, context managers, type hints,
dataclasses, TypedDict); Lesson 2 covers the orchestration stack
(FastAPI, Pydantic, LangGraph); Lesson 3 covers the satellite
libraries (httpx, Qdrant + embeddings, Prometheus, pytest). Each
section includes a "Why Audrey needs this" subsection that anchors
the abstract feature to a concrete Audrey reality. Read all three
first so the rest of the course can refer back rather than re-explain.

From Lesson 4 onward, each lesson is one focused area of the codebase
with three parts:

1. **Context** — why this piece exists, what problem it solves, where it
   sits in the system.
2. **Read-along** — the relevant files with line-number callouts to the
   load-bearing parts. Not a full code reproduction; the lesson points at
   what to notice.
3. **Comprehension questions** — 3-5 per lesson. Some are warm-ups; some
   surface gaps.

Python proficiency assumed: "I can read def, classes, and control
flow." Anything more advanced (async, type hints, decorators, context
managers) gets introduced in Lessons 1, 2, and 3.

## Prerequisites

- The Audrey repo (you're in it)
- Python 3.12 + `uv sync --extra dev` (so you can run `pytest`)
- Comfort with reading code and running shell commands

## Lessons

### Before you start

0. [Introduction — what is Audrey, and what is this course?](lesson-00-introduction.md) —
   a short orientation: what the project is, what it does for the
   user, and the high-level shape of the system. Read first so the
   foundations lessons have something to attach to.

### Part I — Foundations

1. [Foundations I: Python language features](lesson-01-foundations.md) —
   `async`/`await`, context managers, type hints + dataclasses,
   TypedDict. The language layer; why each feature exists, why
   Audrey needs it specifically, where you'll see it.
2. [Foundations II: the orchestration stack](lesson-02-foundations-libraries.md) —
   FastAPI, Pydantic, LangGraph. The libraries that *shape* Audrey's
   HTTP surface and pipeline.
3. [Foundations III: the satellite libraries](lesson-03-foundations-satellites.md) —
   httpx, Qdrant + embeddings, Prometheus, pytest. The libraries
   Audrey *calls out to*.
4. [The request lifecycle, end-to-end](lesson-04-request-lifecycle.md) —
   one request from OWUI to the user, all the way through. Touches
   every major component without going deep on any.
5. **Configuration + startup** *(coming next)* — `main.py`, `config.py`,
   `compose.yaml`. How the app boots and what gets read where.
6. **The model layer** *(coming)* — `models/`. The Ollama client, the
   model registry, the health tracker.

### Part II — The pipeline

7. **Classify + complexity** *(coming)*
8. **Memory recall + datetime injection** *(coming)*
9. **Fast path + ReAct** *(coming)*
10. **Deep panel + synthesis** *(coming)*
11. **Reflection + retry** *(coming)*
12. **Fair scheduling + in-flight cap** *(coming)*

### Part III — Knowledge base + tools

13. **KB ingest + storage** *(coming)*
14. **Tools dispatch + custom-tools server** *(coming)*

### Part IV — Routes, observability, ops

15. **Routes + auth + metrics** *(coming)*

## What you'll know at the end

- Every line of every Python file in `src/audrey/` and `tools-server/`,
  in context.
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
