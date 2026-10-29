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

**Foundations and the platform**

- [Lesson 0 — Introduction](lesson-00-introduction.md)
- [Lesson 1 — Foundations I: Python language features](lesson-01-foundations.md)
- [Lesson 2 — Foundations II: the orchestration stack](lesson-02-foundations-libraries.md)
- [Lesson 3 — Foundations III: the satellite libraries](lesson-03-foundations-satellites.md)
- [Lesson 4 — The platform and the request lifecycle](lesson-04-request-lifecycle.md)
- [Lesson 5 — Configuration and startup](lesson-05-configuration-and-startup.md)
- [Lesson 6a — The application database](lesson-06a-application-database.md)

**The answer pipeline**

- [Lesson 6 — The model layer I: choosing a model](lesson-06-the-model-layer.md)
- [Lesson 7 — The model layer II: calling models](lesson-07-calling-models.md)
- [Lesson 8 — Classification and routing](lesson-08-classification-and-routing.md)
- [Lesson 9 — Deep mode and research mode](lesson-09-deep-mode.md)
- [Lesson 10 — Tool use and the ReAct loop](lesson-10-tool-use-and-react.md)
- [Lesson 11 — How function calling actually works](lesson-11-how-function-calling-works.md)
- [Lesson 12 — The knowledge base: ingest and search](lesson-12-kb-ingest-and-search.md)
- [Lesson 13 — The KB lifecycle: watcher, reconcile, uploads](lesson-13-kb-lifecycle.md)
- [Lesson 14 — Per-user context: identity, memory, and history](lesson-14-memory-and-context-injection.md)
- [Lesson 15 — Fair scheduling: how Audrey shares one GPU](lesson-15-fair-scheduling.md)
- [Lesson 16 — The OpenAI routes, end-to-end](lesson-16-openai-routes.md)
- [Lesson 17 — The custom-tools sidecar](lesson-17-custom-tools-sidecar.md)

Lessons 11–16 are being brought up to date one at a time. Until a
lesson's refresh lands, parts of it may describe an earlier version of Audrey.

**Lesson numbers changed on 2026-10-23,** when the model layer became two
lessons and research mode joined deep mode. Old Lessons 7–16 are now 8–17
(old Lesson *N* is now Lesson *N*+1), Lesson 7 is new, and research mode, old
Lesson 17, is now a section of Lesson 9. Older notes elsewhere in the
repository may use the old numbers.

**The course structure was revised on 2026-10-23:** 27 lessons in seven parts,
with the pipeline taught in the order a request meets it, and identity and the
application database moved into the platform part. The
[plan](../plans/lesson-catch-up-plan.md#the-course-structure-revised-2026-10-23)
has the table. The application database lesson is published as Lesson 6a, read
between Lessons 5 and 6, so the other files keep their current numbers until the
identity lesson is added. That renumber gives both lessons their numbers and puts
the pipeline lessons in their new order.

## Prerequisites

- The Audrey repo (you're in it), so you can open the files each lesson
  cites. Nothing needs installing or running: the lessons explain the
  code, and never ask you to run it.
- Comfort with reading code. Comfort with modern Python (async, type
  hints, decorators) is *not* assumed — the foundations lessons cover
  those.

## Out of scope

- Container infra (Unraid, Docker, Cloudflare Tunnel) beyond the
  trust boundaries the lessons need — see
  [`../campaign-1/`](../campaign-1/) phase docs for that.
- React internals; the course covers the browser boundary.
- Model training, embedding theory, etc.
