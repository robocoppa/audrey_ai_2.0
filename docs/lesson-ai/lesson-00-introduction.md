# Introduction — what is Audrey, and what is this course?

**Audrey** is a self-hosted AI application that runs on a single server.
People reach it in two ways. In a **browser**, they sign in, chat, attach files,
and keep related conversations together in Projects. **Bots and scripts** call
its OpenAI-compatible API with a personal access token, the same way they would
call a commercial model provider. Behind both sits one **pipeline** that decides,
for every turn, how the answer gets made: which model, how many models, whether
to use tools, and whether the result is good enough to send.

From the user's side it looks like any modern chat assistant: type a question,
watch the answer stream in, sometimes with a progress line while the models
work. From the inside, each turn fans out into a small series of decisions,
carried out by local models on the server's GPU and by cloud models, both
reached through the same model server, [Ollama](https://ollama.com/).

The pipeline is what makes Audrey more than a thin proxy in front of Ollama. A
short prompt ("what time is it?") flows through the **fast path**: one model,
one shot, streamed straight back. A harder prompt ("compare BTRFS and ZFS for a
5-bay NAS") flows through the **deep panel**: several models draft answers in
parallel, a synthesizer model reads all the drafts and writes one grounded
answer, and a reflection step checks the result before it ships. **Research
mode** goes further for questions that need sources: researchers gather
evidence, separate stages check it, and a writer produces the answer, with a
Sources list that code builds rather than the model. Tools are available
throughout: web search, opening a web page, the shared knowledge base, a
person's own files, their saved memories and their past conversations.

Underneath the pipeline are the pieces that make a home-hosted system usable by
more than one person, and trustworthy with their data:

- **Fair scheduling** shares the GPU, so one person's burst of requests can't
  starve everyone else.
- **Identity is Audrey's own.** Cloudflare Access proves who is at the browser
  and personal access tokens identify bots, but Audrey assigns each account its
  stable id, its role and its private storage.
- **Conversations belong to the server.** Every message, and every answer still
  being written, is recorded in Audrey's database. Closing the tab doesn't lose
  an answer, and a reload shows exactly what was saved.
- **Files and media become searchable text.** Uploaded documents are indexed
  for search; videos, audio and scanned PDFs are first turned into text in an
  isolated worker container. A shared knowledge base of documents and images
  sits alongside, kept current by a watcher.
- **Side effects are durable.** Work that must happen after a turn, such as
  archiving it for later search or finishing a deletion, is written down first
  and retried until it's done, even across a system restart.
- **Operations are visible.** One readiness report says what's up and what's
  degraded, and Prometheus and Grafana record latency, throughput, fairness,
  model usage and tool calls.

The system was built in small, self-contained steps. Each one landed a single
feature (a route, a pipeline stage, an identity boundary, a worker) and wasn't
considered done until it had been deployed and checked against the live server.
The build history and design notes are indexed in the [docs README](../README.md).
They aren't a tutorial path, but they record what was built when, and why each
decision went the way it did. Most of the code was written with an AI
pair-programmer (Claude Code and Codex) on a laptop, while every git operation and deploy
step was run by hand on the server. That workflow shaped the codebase's tone:
comments, names and docstrings are written for the *next* reader, because the
next reader was usually an assistant opening the file with no memory of why it
existed. A hermetic test suite guards the load-bearing behavior, such as fair
scheduling, identity, ownership checks and the guards on outbound requests, and
runs on the laptop without any live service.

## How the course is organized

The course walks the codebase from the language features and libraries it's
built on, through the request pipeline, to the application around it. It's published openly because the patterns may help anyone building a
multi-model application on a home server.

The lessons are grouped in five parts:

1. **Foundations and the system map.** The Python features and libraries Audrey
   relies on, then the platform as a whole: the containers, the life of a
   request, configuration and startup.
2. **The answer pipeline.** Models, routing, deep mode, tools, the knowledge
   base, the application database and identity, per-user context, fair
   scheduling, the OpenAI-compatible routes, the tools sidecar, and research
   mode.
3. **The native application.** Runs that outlive the browser connection, the
   event stream behind every answer, the browser side, and side effects that
   survive a restart.
4. **Files and media.** Files as durable objects, background jobs, turning media
   into text, reaching outside safely, and asking questions about your own
   files.
5. **Shaping a turn, and the second API.** Skills, Projects, and the Responses
   API.

Start with [Lesson 1 — Foundations I: Python language features](lesson-01-foundations.md).
