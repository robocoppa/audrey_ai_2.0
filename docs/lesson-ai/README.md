# Audrey AI 2.0 — Codebase Lessons

A guided walk through the Audrey codebase end-to-end, written for
the project author to consolidate knowledge after shipping the
system, and published openly because the patterns may help others
building similar multi-model orchestrators on a home server.

The course starts with foundations primers (Python language features
+ the libraries Audrey is built on) and then walks the codebase one
focused area at a time, until every Python file in `src/audrey/` and
`tools-server/` has a referent in your head.

## Where to start

Begin with [Lesson 0 — Introduction](lesson-00-introduction.md). It's
a short orientation that explains what Audrey is, how the request
pipeline is shaped, and what this course is for. From there, work
through the lesson files in this directory in numeric order; each
one ends with a pointer to the next.

## Lessons published so far

Lessons are written one at a time, and the course is being extended to
cover the application around the pipeline. Lesson 0 describes the full
plan.

**Foundations and the system map**

- [Lesson 0 — Introduction](lesson-00-introduction.md)
- [Lesson 1 — Foundations I: Python language features](lesson-01-foundations.md)
- [Lesson 2 — Foundations II: the orchestration stack](lesson-02-foundations-libraries.md)
- [Lesson 3 — Foundations III: the satellite libraries](lesson-03-foundations-satellites.md)
- [Lesson 4 — The platform and the request lifecycle](lesson-04-request-lifecycle.md)
- [Lesson 5 — Configuration and startup](lesson-05-configuration-and-startup.md)

**The answer pipeline**

- [Lesson 6 — The model layer](lesson-06-the-model-layer.md)
- [Lesson 7 — Classification and routing](lesson-07-classification-and-routing.md)
- [Lesson 8 — Deep mode: planner, panel, synthesizer, reflect](lesson-08-deep-mode.md)
- [Lesson 9 — Tool use and the ReAct loop](lesson-09-tool-use-and-react.md)
- [Lesson 10 — How function calling actually works](lesson-10-how-function-calling-works.md)
- [Lesson 11 — The knowledge base: ingest and search](lesson-11-kb-ingest-and-search.md)
- [Lesson 12 — The KB lifecycle: watcher, reconcile, uploads](lesson-12-kb-lifecycle.md)
- [Lesson 13 — Per-user context: identity, memory, and history](lesson-13-memory-and-context-injection.md)
- [Lesson 14 — Fair scheduling: how Audrey shares one GPU](lesson-14-fair-scheduling.md)
- [Lesson 15 — The OpenAI routes, end-to-end](lesson-15-openai-routes.md)
- [Lesson 16 — The custom-tools sidecar](lesson-16-custom-tools-sidecar.md)
- [Lesson 17 — Research mode](lesson-17-research-mode.md)

Lessons 6–17 are being brought up to date one at a time. Until a lesson's
refresh lands, parts of it may still describe the earlier frontend, Open
WebUI, as the client.

## Prerequisites

- The Audrey repo (you're in it).
- Python 3.12 + `uv sync --extra dev` (so you can run `pytest`).
- Comfort with reading code and running shell commands. Comfort with
  modern Python (async, type hints, decorators) is *not* assumed —
  the foundations lessons cover those.

## Out of scope

- Container infra (Unraid, Docker, Cloudflare Tunnel) beyond the
  trust boundaries the lessons need — see
  [`../campaign-1/`](../campaign-1/) phase docs for that.
- React internals; the course covers the browser boundary.
- Model training, embedding theory, etc.
