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

## Prerequisites

- The Audrey repo (you're in it).
- Python 3.12 + `uv sync --extra dev` (so you can run `pytest`).
- Comfort with reading code and running shell commands. Comfort with
  modern Python (async, type hints, decorators) is *not* assumed —
  the foundations lessons cover those.

## Out of scope

- Container infra (Unraid, Docker, Cloudflare Tunnel) — see
  [`../campaign-1/`](../campaign-1/) phase docs for that.
- Frontend integration (Open WebUI) — black box from Audrey's side.
- Model training, embedding theory, etc.
