# Campaign 2 Phase 40 — uploads in the chat surface

Video ingest works. Phases 32-38 built the transport, the job lifecycle, the
worker, the transcript, the visual pass, the summary and the cost work, and
phase 39 made it all findable. What none of them built is a way to *use* it
from the place people actually sit, which is an OWUI chat window.

**Status: steps 1-2 done 2026-08-05 (step 1 as a banner, not a sidebar link —
see below). Steps 3-4 not built. Nothing deployed yet.**

---

## The gap

The retrieval half already works, and that is worth stating precisely because
it bounds the phase. `kb_search` is in `_USER_SCOPED_TOOLS`
([`tools/dispatch.py`](../../src/audrey/tools/dispatch.py)), so the dispatcher
**overwrites** whatever `user` argument the model supplies with the real
pipeline user id. A chat turn reaches the caller's own video chunks and
structurally cannot reach anyone else's. That is not what needs fixing.

What is missing is everything around it:

1. **Nothing links to the upload page.** `/upload` exists and the only way to
   reach it is knowing the URL.
2. **A question cannot be scoped to one video.** `TextQuery` carries `query`,
   `top_k` and `user`. With ten videos uploaded, "what did they say about the
   handover" searches all of them, and there is no way to say "in this one".
3. **Chat cannot say what you have.** The tool surface is `web_search`,
   `web_fetch`, `kb_search`, `kb_image_search`, three `memory_*` and
   `chat_history_search`. Nothing lists your files, so "what videos have I
   uploaded?" is unanswerable — even though `GET /v1/files` knows exactly.
4. **A 458-second job reports nothing while it runs.** The row goes
   `pending` → `processing` → `ready`, and the page only refreshes when you
   act on it.

**2 and 3 are one feature, not two.** A filename filter the model cannot
discover is a filter it will never use correctly — it would have to guess file
names. Build the listing first and the filter second, or neither is usable.

## What is NOT in scope

- **No `audrey_video` virtual model.** A virtual model is a
  `/v1/chat/completions` endpoint; video ingest is a file-transport plus
  background-job problem, and they do not meet. OWUI's own file uploads go to
  OWUI's storage and RAG, never to Audrey's `/v1/files`, so such a model would
  have no way to receive the bytes — and a chat-completions body still faces
  the 100 MB edge cap that [phase 32](phase-32-video-upload-transport.md)
  exists to work around. As a *retrieval* path it earns even less: `kb_search`
  already reaches the chunks and phase 39 already finds them by quote.
- **No worker heartbeat protocol.** See step 4 below — elapsed time is
  derivable from data the row already holds, and a mid-job progress channel is
  a much larger change for a much smaller gain.
- **No changes to the fairness model or the ingest pipeline.** This phase is
  entirely about the surfaces around it.

---

## Step 1 — a banner linking to the upload page (no code) ✅ DONE

**There is no sidebar-links feature in Open WebUI.** The first version of this
plan, and the `PROJECT_STATE.md` followup it inherited, both named
`Admin → Settings → Interface → Sidebar Links`. That menu does not exist. It is
absent from the settings docs; v0.11.0 reorganized the admin panel into
Settings and rebuilt the sidebar without adding it; and "custom pages /
integrations" is still an open upstream feature request. Nobody had opened the
menu before writing it down as a five-minute task.

What works instead is a **banner**. Banner content is HTML — markdown is *not*
rendered — so it takes an anchor:

```html
<a href="https://<audrey-host>/upload" target="_blank">Upload a video</a>
```

Where to find it depends on version, since v0.11+ merged the admin panel into
Settings:

- **v0.11+** — Settings → Admin → System → General → Banners
- **older** — Admin Panel → Settings → Interface → Banners

**Do not reach for `WEBUI_BANNERS` on a running instance.** It is a
PersistentConfig variable: it seeds the database on first launch only, and
after that the stored value wins and the environment variable is ignored
silently. On this box that means an Unraid container-template edit does nothing
whatsoever, with no error to say so.

Set `dismissible: false` unless the banner is meant to be temporary. A
dismissible banner disappears per-user on first click and the discovery problem
it exists to solve comes straight back.

**This is weaker than the step it replaces, and that re-ranks the phase.** A
banner is a strip above the chat pane, not a permanent navigation entry, and
there is nothing to navigate back to once it is gone. Discovery has to move
into the chat itself instead — which makes steps 2 and 3 the phase's discovery
story rather than its second half.

## Step 2 — a tool that lists your own files ✅ BUILT 2026-08-05

**Shipped as shape (b), a separate service-token route.** What landed:

| file | change |
|---|---|
| `src/audrey/routes/files.py` | `POST /v1/files/list` — `require_service`, user in the body, returns `ModelFileRow` |
| `tools-server/app.py` | `list_my_files` tool, proxying that route |
| `src/audrey/tools/dispatch.py` | `list_my_files` in `_USER_SCOPED_TOOLS`; new `audit_user_scoping()` |
| `src/audrey/main.py` | the audit runs on all three discovery paths — boot, background retry, `/v1/tools/rediscover` |
| `tests/test_files_service_list.py` | new; route auth, isolation, response shape |
| `tests/test_dispatch.py` | the cross-user overwrite, plus the audit |

1304 hermetic tests pass, ruff clean on every touched file, cite drift for these
files repaired (22 remaining, all pre-existing).

**One correction to the analysis below.** The choice between (a) and (b) is more
lopsided than it reads. `resolve_kb_caller` returns
`KBCaller(email=None, is_service=True)` — there is **no act-as field on the
caller object at all**. `/v1/kb/query` gets its target user from `req.user` in
the request *body* and picks with
`effective_user = req.user if caller.is_service else caller.email`. So (a) was
never "extend a dependency"; it was "give a `GET` route a request body, or put
an email in a query string". (b) it is.

**`require_service`, not `resolve_kb_caller`, on the new route.** The latter
also accepts a user JWT, and this route names its target in the body — so any
logged-in user could have read anyone's file list by typing a different address.
Same reasoning the job routes already use.

**The startup check was worth building.** `audit_user_scoping()` walks the
discovered registry, and warns for any tool whose request schema declares a
`user` property that is not in `_USER_SCOPED_TOOLS`. A warning, not a boot
failure: `user` is an ordinary word and a false positive that stops the stack
would just get the check deleted. The test that earns its keep runs the **real**
tools-server OpenAPI through the **real** discovery path — the three tests
either side of it audit synthetic specs and only prove the function works.

**Not done here:** the tool description promises nothing about scoping a search,
because step 3 does not exist yet. Add that clause when it does.

---

**Check the auth before designing anything.** `GET /v1/files` depends on
`require_user`, which wants a real OWUI bearer. The tools-server holds
`KB_SERVICE_TOKEN` and cannot obtain a user JWT, so a naive proxy gets a 401.

**This is the fourth phase to walk into that trap** — phase 36 planned around
a service-token act-as on `/v1/chat/completions` that does not exist, phase 37
inherited the same assumption, and the phase-36 plan's transport had to be
rebuilt as `POST /v1/media/describe`. `resolve_kb_caller` (service token plus
an act-as user) lives **only on the KB query routes**. Confirm what a route
actually depends on before writing a line.

Two viable shapes, and the choice should be made deliberately:

- **(a) Extend `resolve_kb_caller` to the file-list route.** Smallest
  diff. The route becomes reachable by a service token naming any user, which
  is the same trust already granted on `/v1/kb/query` — so it widens an
  existing boundary rather than opening a new one.
- **(b) A separate service-token route**, mirroring `/v1/media/describe`.
  More code, but keeps the user-facing list route untouched and lets the
  tool's response shape be shaped for a model (fewer fields, no byte counts)
  rather than for a UI.

**(b) is the recommendation**, for the reason phase 36 chose it: the tool wants
a different response than the page does, and a route with two audiences drifts
toward serving neither. A model does not need `bytes`, `collection` or
`source_freed_at`; it needs filename, status, duration and summary.

Then, in order:

1. `tools-server/app.py` — the route and its `user` request-body field.
2. **`_USER_SCOPED_TOOLS` in [`tools/dispatch.py`](../../src/audrey/tools/dispatch.py).**
   That file carries an explicit warning about this: edit the tools-server
   without editing the set, and the dispatcher passes the *model-supplied*
   `user` straight through — other users' file lists become reachable from a
   prompt. There is no startup check for it today.
3. Consider adding that startup check while you are here. The warning says to
   look for a `user` property in `ToolSpec.parameters` and complain when the
   name is absent from the set. It would have to be a warning rather than a
   hard failure, since a tool may legitimately take an unrelated `user` field.

**The tool description is load-bearing.** A model calls a tool because of what
its description says. It should say plainly that this lists the caller's own
uploaded files and returns their filenames — because the next step depends on
the model having those filenames to hand.

## Step 3 — scope a search to one file

`TextQuery` gains an optional filter. `file_id` is the precise key, but a model
will only ever have a **filename**, so the filter should accept a filename and
resolve it. Decide where that resolution happens — resolving in the route means
one place, and means a filename that matches nothing can return a clear empty
rather than a silent full-corpus search.

**The trap that will bite: the filter must apply to BOTH retrievers.** Phase 39
made KB search hybrid — `_search_text_hybrid` runs `search_text` (dense) and
`search_lexical` (BM25) and fuses them by reciprocal rank. Neither takes a
filter today. Filter only the dense side and the lexical side keeps returning
hits from every other file, fusion mixes them, and the result looks *almost*
right — which is the worst failure mode available, because nobody checks a
result that looks plausible.

Qdrant filtering is already in use elsewhere in
[`kb/qdrant.py`](../../src/audrey/kb/qdrant.py) (`FieldCondition` +
`MatchValue` on the delete paths), so the mechanism is proven; it is the
plumbing through both search methods and the fusion that is new.

Payloads already carry everything needed: `file_id`, `filename`, `user`,
`source`, `chunk_idx`, and — usefully — `artifact`, which is `"transcript"`,
`"visual"` or `"summary"`. An artifact filter is nearly free once a file filter
exists, and "what was *on screen*" versus "what was *said*" is a real
distinction a user will want.

**Then decide whether the model can be trusted to use it.** A filter the model
applies wrongly is worse than none — scoping to the wrong video answers
confidently from the wrong source. Worth testing whether it scopes only when
the user named a file, or whenever any file is mentioned in the conversation.

## Step 4 — say what is happening during a job

The cheap, honest version needs no new protocol. The row already records
`leased_at` when a worker claims it, so the page can render "processing for
4m12s" from data it can already fetch. Add a poll while anything is `pending`
or `processing`, and stop when nothing is.

**`leased_at` is not currently exposed**, and adding it is the standing
three-places lesson: the column exists in the schema, but `_list_user_sync`
selects an explicit column list and `FileRow` is a separate model. Miss the
SELECT and `GET /v1/files` returns a 500 for every user — that exact bug
shipped on 2026-08-04 when `summary` was added to two places out of three.

Resist a real progress protocol. The worker would have to report stage
transitions over HTTP, `ingest_result` would need a sibling endpoint, and the
lease logic would have to tolerate partial updates — a lot of surface for
"whisper is done, frames are next". Elapsed time against a known typical
duration (~458s for a nine-minute video) tells the user what they actually
want, which is whether to wait or come back.

---

## Which model to ask with

Relevant to verification, and worth writing down because it is not obvious.

**`audrey_auto`.** A short question classifies to the fast path, which runs a
ReAct loop *only when the picked model is in `fast_path.tool_capable_models`*.
The `general` pool's priority-100 entry is `qwen3.6:35b`, which is on that
list, so tools are active and `kb_search` fires with the dispatcher-injected
user id. It is also local, so it costs no cloud credit.

Three ways to get this wrong:

- **`audrey_passthrough` has no tools at all** — no classifier, no gate, no
  tool loop. It will answer about a video from nothing, confidently.
- **The deep pools work but are overkill**, and `audrey_cloud` puts a
  retrieval question on paid inference.
- **Phrasing can force deep mode.** `complexity.deep_intent_phrases` includes
  "thorough", "in depth", "comprehensive" and "step by step", matched as
  case-insensitive substrings. "Give me a thorough summary of that video"
  routes to the deep panel regardless of length.

## Verification

**1. The banner renders as a real link** and opens the upload page from inside
OWUI, as a logged-in user. Banner content is HTML-only, so a markdown link
renders as literal `[text](url)` characters rather than failing visibly —
confirm it is an anchor, not text that looks like one.

**2. The listing tool returns only the caller's files.** ✅ **Hermetic, done.**
The negative test is the one that matters —
`test_dispatch_one_overwrites_user_for_list_my_files` names
`victim@example.com` and asserts `alice@example.com` goes on the wire. The
route's own half (`TestIsolation` in `tests/test_files_service_list.py`) pins
that a user JWT cannot reach it at all and that a second user's rows never
appear. **Still worth one pass by hand on the box**, because nothing hermetic
proves the deployed tools-server and the deployed Audrey agree on the route
path.

**3. A scoped question searches one file.** Ask about something that appears in
two uploaded videos, once unscoped and once scoped, and confirm the sources
differ. **Check the lexical side specifically** — a quote that appears verbatim
in the *other* file is the case that catches a filter applied to only one
retriever.

**4. A filename that matches nothing returns empty**, not a silent full-corpus
search.

**5. The model uses the tools unprompted.** No prompt should name them —
phase 29 established that OpenAPI discovery is enough and that naming a tool in
a prompt is how you get it called when it should not be. If adoption is poor,
measure it before prompt-steering, per the A-B-A rule.

**6. The file list still returns 200 after adding `leased_at`.** The
three-places check, and there is an existing test pinning `FileRow` against
what `list_user` actually returns.

**7. A processing row shows elapsed time and stops polling when done.**

### Rollback

Steps are independent. The sidebar link is config. The tool is additive — an
undiscovered tool is simply never called. The filter is an optional field, so
omitting it restores current behaviour exactly, which also makes it the safest
thing to ship first and measure.

## What this unblocks

Video stops being a side door. The natural follow-ons are the same treatment
for non-video uploads (a PDF summary is a reasonable want and a different
phase), and answering "what do I have" well enough that the upload page becomes
optional rather than mandatory.
