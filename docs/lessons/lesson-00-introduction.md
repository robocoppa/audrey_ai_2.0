# Introduction — what is Audrey, and what is this course?

**Audrey** is a self-hosted AI assistant that sits between a chat
interface and a stable of large language models. It runs on a single
home server with two consumer GPUs, accepts chat requests from
[Open WebUI](https://github.com/open-webui/open-webui) (a browser
frontend), routes each request to the right combination of models and
tools, and streams the answer back. From the user's side it looks
like ChatGPT — type a question, get a streamed response, sometimes
with a "thinking…" banner while the model works. From the inside,
each request fans out into a small pipeline of decisions about
*which* model, *how many* models, *whether to use tools*, and *whether
the answer is good enough to ship*.

The pipeline is what makes Audrey more than a thin proxy in front of
[Ollama](https://ollama.com/). A short prompt ("what time is it?")
flows through the **fast path** — one model, one shot, streamed
straight back. A harder prompt ("compare BTRFS and ZFS for a 5-bay
NAS") flows through the **deep panel** — several models run in
parallel, each producing a draft, then a synthesizer model reads all
the drafts and writes a single grounded answer. A reflection step
checks the result before it ships and, if the answer looks broken,
re-runs the panel with a stronger model. Tools are wired in at every
step: the panel can call out to a web search, a knowledge-base
vector lookup, an image search, or a per-user memory store, then
fold those results back into its answer.

Underneath the pipeline are the unglamorous pieces that make a
home-hosted system actually usable for more than one person. **Fair
scheduling** caps how many concurrent generations each user can run
so one person can't starve everyone else off the GPU. **Per-user
memory and uploads** keep each person's facts and files private to
them. A **knowledge base** of indexed text and image chunks
lives on disk, ingested by a watcher that picks up new files
automatically. **Authentication** runs every request through OWUI's
session check, so the public URL can't be hit by drive-by traffic.
**Prometheus + Grafana** record latency, throughput, fairness, and
tool-call rates so problems are visible before users notice them.

The whole system was built in **discrete phases**, each landing one
self-contained feature: an HTTP route, a pipeline node, an auth
boundary, a metrics surface, a knowledge-base capability. Every
phase had a deploy doc that walked the change from "edit these
files" through "rebuild the container" to a numbered list of smoke
tests run against the live server, and a phase wasn't considered
done until those tests passed against actual traffic. The phase
docs live on as build history in [`docs/campaign-1/`](../campaign-1/) —
they aren't a tutorial path, but they capture what was built when
and why each decision went the way it did. Coding was done with an
AI pair-programmer (Claude Code) on the laptop while the author
ran all git operations and deploy steps manually on the server.
That workflow shaped a lot of the codebase's tone: the comments,
naming, and docstrings were written for the *next* reader, because
the next reader was usually Claude looking at the file with no
memory of why it existed. Late in the campaign a hermetic test
suite went in as a
regression-guard layer — pytest tests that cover the load-bearing
behavior (fair scheduling, auth, SSRF guards, the reflection
short-circuit) and run in well under a second, so the cost of
checking before every commit is essentially zero.

This course walks the codebase from the libraries it's built on up
through the request pipeline, the storage layer, the tool surface,
and the operational glue. It's written for the codebase author
(learning his own Python codebase after shipping it) and published
openly because the patterns may help anyone else building a
multi-model orchestrator on a home box.

Start with [Lesson 1 — Foundations I: Python language features](lesson-01-foundations.md).
