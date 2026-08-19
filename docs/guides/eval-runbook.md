# Eval runbook — on-box cheat sheet

Everything here runs **on the box**, from `/mnt/user/appdata/audrey_ai_2.0`
(with the `_2.0` suffix — `/mnt/user/appdata/audrey` does not exist).

Tracked, so a `git pull` puts it on the box next to the commands it describes.

---

## Launch

Always `nohup … &`. `eval-onbox.sh` starts the eval as a **detached container**
and then waits on it only to send the Telegram ping — so an SSH drop can never
kill a run, but it does kill the notification.

```bash
cd /mnt/user/appdata/audrey_ai_2.0
```

| variable | default | notes |
|---|---|---|
| `MODEL` | `audrey_research` | virtual model, or `$1` |
| `CASES` | `eval_prompts_protocol.json` | or `$2` |
| `LABEL` | `$MODEL` | names the output file — **set it** when the model is ambiguous (code/topics/deep all run `audrey_deep`) |
| `ARGS` | — | forwarded to the harness verbatim (`--only`, `--repeat`, `--verbose`) |
| `MODELS` | — | comma-separated sweep; also writes a results JSON for `eval_compare.py` |

Output lands in `testing-out/<date>-<HHMMSS>-<LABEL>-onbox-answers.md`. The
timestamp means re-running the same LABEL on one day no longer overwrites.

### One case, many times — the only way to tell a real change from variance

n=1 cannot resolve this suite's ±5 swing.

```bash
MODEL=audrey_auto CASES=eval_prompts_video.json LABEL=paging \
  ARGS='--only video-long-transcript-paging --repeat 5' \
  nohup scripts/eval-onbox.sh > testing-out/paging.log 2>&1 &
```

### Suites, and roughly how long

Research/deep turns run 2–4 min each; fast/video turns 15–60 s.

⚠️ **The `executed` column is the one to read when the question is "is this
model any good".** Those cases extract the answer's code, run it, and pass on
exit 0 — a real verdict. Every other check in this harness is structural or
phrase-matched: they catch a model that wandered off task, fabricated, or lost
its fence, but they cannot tell a correct answer from a plausible one. A suite
with `0 executed` measures shape, not accuracy.

| cases file | n | executed | typical model | rough time |
|---|---|---|---|---|
| `eval_prompts_protocol.json` | 10 | 0 | `audrey_research` | **20–40 min** |
| `eval_prompts_deep.json` | 18 | 2 | `audrey_deep` | long |
| `eval_prompts_topics.json` | 13 | 0 | `audrey_deep` | long |
| `eval_prompts_video.json` | 12 | 0 | `audrey_auto` | ~10 min |
| `eval_prompts_fast.json` | 12 | 0 | `audrey_fast` | short |
| `eval_prompts_code.json` | 10 | 7 | `audrey_deep` | plumbing gate |
| `eval_prompts_code_hard.json` | 5 | **5** | `audrey_deep` | the discriminating tier |
| `eval_prompts_writer_ab.json` | 5 | 0 | `audrey_research` | writer A-B |
| `eval_prompts_local_models.json` | 12 | 2 | sweep | local bake-off (SMOKE TEST) |
| `eval_prompts_code_models.json` | 6 | **6** | sweep | coding lineup |
| `eval_prompts_code_hard_models.json` | 5 | **5** | sweep | hardest, unpinned |
| `eval_prompts_models_ab.json` | 9 | 1 | sweep | lineup A-B |

```bash
# video
nohup env MODEL=audrey_auto CASES=eval_prompts_video.json LABEL=video \
  scripts/eval-onbox.sh > testing-out/last-video-run.log 2>&1 &

# coding — plumbing gate, then the tier that actually discriminates
nohup env MODEL=audrey_deep CASES=eval_prompts_code.json LABEL=code \
  scripts/eval-onbox.sh > testing-out/last-code-run.log 2>&1 &
nohup env MODEL=audrey_deep CASES=eval_prompts_code_hard.json LABEL=code-hard \
  scripts/eval-onbox.sh > testing-out/last-code-hard-run.log 2>&1 &

# topics (reasoning / science / writing / general knowledge)
nohup env MODEL=audrey_deep CASES=eval_prompts_topics.json LABEL=topics \
  scripts/eval-onbox.sh > testing-out/last-topics-run.log 2>&1 &

# research protocol (the default)
nohup env MODEL=audrey_research CASES=eval_prompts_protocol.json LABEL=protocol \
  scripts/eval-onbox.sh > testing-out/last-research-run.log 2>&1 &
```

### Per-model sweeps

`MODELS=` runs **every case once per model** and writes a results JSON beside
the answers file — feed that to `scripts/eval_compare.py`.

```bash
# local bake-off
nohup env CASES=eval_prompts_local_models.json LABEL=local-bakeoff \
  MODELS='audrey_passthrough/nemotron-3.5-lightning:latest,audrey_passthrough/muse-glimmer:latest,audrey_passthrough/qwen3.8:latest' \
  scripts/eval-onbox.sh > testing-out/last-run.log 2>&1 &

# coding lineup
nohup env CASES=eval_prompts_code_models.json LABEL=code-sweep \
  MODELS='audrey_passthrough/qwen3.8:latest,audrey_passthrough/nemotron-3.5-lightning:latest,audrey_passthrough/kimi-k2.7-code:cloud,audrey_passthrough/deepseek-v4-pro:cloud' \
  scripts/eval-onbox.sh > testing-out/last-code-sweep.log 2>&1 &

# general lineup A-B
nohup env CASES=eval_prompts_models_ab.json LABEL=models-ab \
  MODELS='audrey_passthrough/qwen3.8:latest,audrey_passthrough/muse-glimmer:latest,audrey_passthrough/deepseek-v4-pro:cloud,audrey_passthrough/kimi-k2.6:cloud,audrey_passthrough/glm-5.2:cloud' \
  scripts/eval-onbox.sh > testing-out/last-models-ab-run.log 2>&1 &

# virtual-model A-B (task role on vs off)
nohup env MODEL=audrey_video CASES=eval_prompts_video.json LABEL=video-ab \
  MODELS='audrey_video,audrey_auto' \
  scripts/eval-onbox.sh > testing-out/last-video-ab-run.log 2>&1 &
```

⚠️ A sweep multiplies cloud spend by the model count. Cloud credits are a hard
budget.


### "How good is this model?" — the capability battery

Reach for this when a NEW model lands and the question is capability, not
lineup. `eval_prompts_local_models.json` does **not** answer it: its own header
calls it a smoke test, and only 2 of its 12 cases are executed. It tells you a
model can hold a format and disclaim a gap — nothing about whether its answers
are right.

Three suites, twenty cases, **twelve of them executed**. Run them in this order;
the first is the sharpest signal per minute.

```bash
M=audrey_passthrough/<model>:latest

# 5 cases, ALL executed — LRU-with-TTL, async debugging, a tokenizer,
# topological sort, duration parsing. The tier that discriminates.
nohup env CASES=eval_prompts_code_hard_models.json LABEL=cap-hard \
  MODELS="$M" scripts/eval-onbox.sh > testing-out/cap-hard.log 2>&1 &

# 6 cases, ALL executed — LRU cache, merge intervals, two debugging cases,
# word frequency, flatten. Broader and easier; separates "cannot code" from
# "cannot do the hard ones".
nohup env CASES=eval_prompts_code_models.json LABEL=cap-code \
  MODELS="$M" scripts/eval-onbox.sh > testing-out/cap-code.log 2>&1 &

# 9 cases — reasoning, science explanation, writing, general knowledge.
# Only 1 executed, so the score is a floor: it catches wandering and
# fabrication, not weak prose. ▶ READ THE ANSWERS FILE for these.
nohup env CASES=eval_prompts_models_ab.json LABEL=cap-ab \
  MODELS="$M" scripts/eval-onbox.sh > testing-out/cap-ab.log 2>&1 &
```

Then the same three against whatever the candidate would displace — a score
with no baseline is not a result. Models that fit in VRAM together can share one
sweep:

```bash
nohup env CASES=eval_prompts_code_hard_models.json LABEL=cap-hard-base \
  MODELS='audrey_passthrough/qwen3.8:latest,audrey_passthrough/nemotron-3.5-lightning:latest' \
  scripts/eval-onbox.sh > testing-out/cap-hard-base.log 2>&1 &
```

⚠️ **A model that cannot be resident gets its OWN run, never a sweep.**
`_expand_sweep` groups by model so each loads once, which is enough when the set
fits in 48 GB and useless when it does not — the big one is evicted and reloaded
from disk on every alternation. 2026-08-18: `laguna-s-2.1` is 96 GB against
48 GB of VRAM and ran at 4–6 tok/s, so its latency column is a memory-bandwidth
reading, not a measurement of the model. Judge such a model on ACCURACY here and
take its latency from a box where it fits.

⚠️ **Quantization is part of the answer, not a detail.** `ollama show` reports
it, and a Q8_0 build is twice the bytes per weight of Q4_K_M — the same
parameter count can be resident or not depending purely on which tag was pulled.
Check `ollama show <model>` for `parameters`, `quantization` and `embedding
length` before drawing conclusions from a size on disk. ▶ An embedding length
far too small for the parameter count (3072 at 117.6B) means a sparse MoE: the
file is large, the active fraction is not, and it will outrun what its footprint
suggests.

**Before the first run**, a new model needs all three of these or every case
fails identically:

1. `passthrough.allowed_models` in `config.yaml` — the only gate
   (`routes/openai/passthrough.py`). No `model_registry` entry is needed until
   it earns a production role.
2. `scripts/pull-models.sh` — pinned by
   `test_every_model_the_config_names_is_pulled_by_the_script`, so a rebuilt box
   cannot come up missing a name the config mentions.
3. **`docker restart open-webui`** after `up -d --build audrey-ai`. OWUI reads
   `/v1/models` once and holds it, and the harness talks to OWUI, never to
   Audrey. Skipping this fails every case with
   `HTTP 400 {"detail":"Model not found"}` — OWUI's string, absent from `src/`,
   and it reads exactly like a bad model name.

⚠️ **Thinking is a separate arm, not a setting to leave on.** `PASSTHROUGH_THINK`
is global: Claudette, Hermes and OpenClaw all get whatever it says. Set it, run
the arm, take it back out. The completion log now carries `think=` alongside
`thinking_len` and `chars_per_tok`, so what was asked for and what came back are
both on the record — check them rather than inferring thinking from prose shape.
---

## Watch a run

```bash
docker ps --filter name=audrey-eval          # running?
docker logs -f audrey-eval                   # follow
docker logs --tail 5 audrey-eval             # where is it now
ls -lt testing-out/*onbox-answers.md | head -3
```

The container name is fixed (`audrey-eval`) and the script `docker rm -f`s any
previous one, so **only one eval runs at a time** — a second launch kills the
first. Sequential also matters on its own: `GPU_CONCURRENCY=1` means concurrent
runs just queue at the gate and every latency number becomes meaningless.

### "Did it hang?"

Almost always no. Check in this order:

1. **`docker ps -a --filter name=audrey-eval`** — `Up N minutes` inside the
   suite's expected window is a healthy run, not a stall. A 10-case research
   protocol at 29 minutes is normal.
2. **`Exited (0)` + the answers file on disk** — it finished. The container
   never depended on your laptop either way.
3. **A suspiciously small answers file** — see the trap below.

### No Telegram ping

⚠️ **SSH dropping is not the explanation** when you launched with `nohup … &` —
that is the whole point of `nohup`, and the notify block runs after
`docker wait` in a shell that survives SIGHUP. Read the log you redirected to;
every failure path there prints a `WARN:` and none of them are fatal:

```bash
tail -20 testing-out/<your>.log
```

- `>> waiting for audrey-eval to finish…` and nothing after it → still running.
- `WARN: <path>/fleet-watchdog/.env not found` → creds file moved or the
  watchdog stack is down.
- `WARN: WATCHDOG_TOKEN/CHAT_ID not in …` → the file exists but is empty.
- `WARN: Telegram summary send failed` → the box could not reach Telegram.

The answers file is always on the box regardless — a failed send never masks a
completed run.

---

## Traps

**A run can fail every case and still write a complete-looking file.** The tell
is **size**: 10 research answers is tens of KB, and ~2.6 KB is ten error stubs.
Ten research cases finishing in ~3 minutes is the same signal — real research
turns take 2–4 minutes *each*.

```bash
head -3 testing-out/<file>.md               # "N cases, 0 passed" = failed early
grep -m3 "error:" testing-out/<file>.md     # WHY — this is the line that matters
```

⚠️ **`audrey-ai` being healthy is not sufficient.** Two independent things must
be ready, and only one of them is what `docker compose ps` reports:

```bash
docker compose ps audrey-ai                      # Audrey up
curl -s localhost:8000/v1/models | grep audrey_  # Audrey OFFERING the model
```

OWUI keeps its **own** model-list cache and can serve a stale one for a long
time after Audrey restarts — that bit this campaign once already, with a
47-hour-stale list. An eval whose every case 404s on the virtual model looks
exactly like an eval that ran against a dead Audrey.

**Three clocks.** Host is PDT, container logs and eval *filenames* are MDT
(your wall clock), `docker inspect` is UTC. A file stamped `202937` with an
mtime of `19:30` is the same moment, not a discrepancy.

**Never `docker image prune -a`.** `audrey-eval` is build-only, so it is always
"unused" and always deleted, and the next run dies with `image audrey-eval:latest
not built`. Plain `docker image prune -f` (dangling only) is safe.

```bash
docker compose --profile eval build audrey-eval   # only when the IMAGE changes
```

The harness and case files are **mounted from `./scripts` at run time** — edit
or `git pull` them and the next run picks them up, no rebuild. But that also
means a harness change on the laptop does nothing until it is **committed,
pushed and pulled** on the box.

**`✅`/`❌` in the footer counts errors only.** A search that returned 200 with
zero results shows as `✅`. Thin grounding does not look like failure.

---

## Debug flags

All three are env-overridable — **use `.env`, not `config.yaml`**. `config.yaml`
is tracked and bind-mounted: editing it leaves a diff every later `git pull`
has to work around, and an on-box `sed -i` gives the container a stale file
handle that a plain restart will not clear.

```bash
echo 'DEBUG_RESEARCH_TRACE=1' >> .env
docker compose up -d --force-recreate audrey-ai
docker compose logs audrey-ai | grep "ENV OVERRIDE"     # confirm it landed
```

| flag | what it adds | visible to |
|---|---|---|
| `DEBUG_CONTEXT_TRACE=1` | one `context-trace:` line per ReAct round + `FINAL`: live tool results with sizes, how many compaction stubbed, total `convo_chars` | **log only** |
| `DEBUG_PANEL_DRAFTS=1` | every worker's full draft — **and the only way to see a fast turn that escalated to a deep panel** | client |
| `DEBUG_RESEARCH_TRACE=1` | researcher notes, ledger, fact-check verdicts, writer guidance | client |
| `COMPLEXITY_LOG_BREAKDOWN=1` | why a turn routed fast vs deep | log only |

⚠️ The last two are **client-visible** — fine in an eval artifact, noise in
OWUI. Comment them out afterwards.

If you must edit `config.yaml` instead, a force-recreate is mandatory:

```bash
sed -i 's/^\(\s*\)debug_research_trace:.*/\1debug_research_trace: true/' config.yaml
grep -n "debug_research_trace" config.yaml
docker compose up -d --force-recreate audrey-ai
```

⚠️ `config.yaml` is also `COPY`d into the image, so anything the app reads at
**boot** (e.g. `passthrough.allowed_models`) needs `up -d --build`, not just a
recreate.

---

## Useful greps

```bash
# compaction: what the model was actually looking at
docker compose logs audrey-ai | grep "context-trace:" | grep -E "ANSWERED|FINAL"

# a fast turn that was re-run through the deep panel (costs a 3-worker panel,
# two of them cloud, while reporting itself as `fast`)
docker compose logs audrey-ai | grep -c "escalate: fast→deep"
docker compose logs audrey-ai | grep "escalate: fast→deep" | tail -20

# the catalogue guard fetching a file the model tried to describe unread
docker compose logs audrey-ai | grep catalogue-guard

# a researcher naming an authority it never fetched ("Herodotus, Histories",
# "Meta Llama 4 Family Announcement") — demoted so it can no longer make a
# claim read as confident. Expect a handful per research run; ZERO means the
# demotion is not deployed, not that the ledgers were clean.
docker compose logs audrey-ai | grep -c "ledger: demoting url-less"
docker compose logs audrey-ai | grep "ledger: demoting url-less" | tail -20

# the linkage shape of every structuring call. READ THE TRIPLE — it separates
# two different faults that both end in unsourced claims:
#   claims=48 linked=46 dangling=0  sources=11  healthy
#   claims=41 linked=0  dangling=0  sources=5   linkage LOST — cited URLs never
#                                   reached the claims; grounded work gets
#                                   soft-pedalled
#   claims=21 linked=0  dangling=21 sources=3   the model INVENTED citation ids
#                                   (`w2_src_corrode` vs a ledger numbering its
#                                   sources `w2_s2`). Nothing resolves. This is
#                                   the pressure a required `source_ids` creates.
#   claims=16 linked=1  dangling=0  sources=1   the RESEARCHER wrote no
#                                   `SOURCES:` block. Empty source_ids are
#                                   CORRECT here; the fix is upstream.
#   claims=N  linked=0  dangling=0  sources=0   found nothing — grounding, not
#                                   linkage
docker compose logs audrey-ai | grep "research: structured"

# ▶▶ The same line now ends `catalogue=N dropped=D notes_only=M`. The catalogue
# is the ONLY id authority since 2026-08-14, so these three read as:
#   catalogue=N  real `(title, url)` rows the worker retrieved, handed to the
#                structurer as `s1..sN`. ⚠️ `catalogue=0` on a TOOL-USING worker
#                means `_retrieved_sources` parsed nothing — a tools-server
#                response shape changed and the whole mechanism is inert.
#   dropped=D    claim `source_ids` outside the catalogue, discarded. This is the
#                old bug, now visible instead of silent: run `220557` had 37 such
#                cites landing on wrong sources. A steady low D is the model
#                fumbling one id; a spike means it has gone back to inventing a
#                numbering scheme, and the prompt rule needs re-measuring.
#   notes_only=M sources the model named whose URL is nowhere in the catalogue —
#                a `memory_recall` hit, or a page it saw a snippet of and never
#                fetched. These are DROPPED, and M is the price of the trade.
#                ⚠️ If M runs high the fix is a SECOND id namespace, never
#                relaxing the single-authority rule.
docker compose logs audrey-ai | grep -o "catalogue=[0-9]* dropped=[0-9]* notes_only=[0-9]*"

# only the middle case above. ⚠️ Wants ZERO, and unlike the demotion grep a zero
# here is meaningful on its own — the `research: structured` lines prove the
# instrument is live.
docker compose logs audrey-ai | grep -c "UNLINKED-LEDGER"

# ⚠️ The one video shape that produces an authoritative summary built on almost
# nothing: 0 transcript segments AND only 1-2 keyframe descriptions. Zero
# instances so far — the case that prompted the check had 10 descriptions and was
# fine — so this is a watch, not a known bug.
docker compose logs audrey-ai | grep "summarise:" | grep "(0 segments"

# which env overrides are actually live
docker compose logs audrey-ai | grep "ENV OVERRIDE"

# per-turn summary: task, confidence, mode, model, tool calls
docker compose logs audrey-ai | grep "chat.completions model="
```

⚠️ The Unraid shell has **no python3** — use `jq` or plain `grep`, never pipe
to `python3`.

---

## Compare two runs

On the LAPTOP, against results you have pulled down:

```bash
uv run scripts/eval_compare.py testing-out/<a>-results.json testing-out/<b>-results.json
```

⚠️ **On the BOX there is no `python3`**, so it has to run inside a container.
The eval image has Python and both directories are already bind-mountable:

```bash
docker run --rm --entrypoint sh \
  -v /mnt/user/appdata/audrey_ai_2.0/testing-out:/out \
  -v /mnt/user/appdata/audrey_ai_2.0/scripts:/eval:ro \
  audrey-eval:latest -c \
  'python /eval/eval_compare.py /out/*-lag-s-hard-*-results.json /out/*-base-hard-*-results.json'
```

⚠️ **Entrypoint `sh -c`, not `python`, whenever the paths carry a glob.** `/out`
does not exist on the HOST, so the host shell finds nothing to expand and passes
the pattern through as a literal — Python then reports
`results file not found: /out/*-…-results.json`, which reads like a missing file
rather than an unexpanded wildcard. Letting the CONTAINER's shell expand it is
the fix; naming both files explicitly also works.

A results JSON is written for **every** run, not only sweeps — a single-model
re-run used to leave the human-readable `.md` alone, so run-over-run comparison
was eyeball-only.

⚠️ **A run's score is not a quality trend line.** It moves with what the checks
can see at least as much as with the answers. Before reading a delta, confirm
both runs used the same harness commit — a check added between them changes the
number without anything changing in Audrey.

---

## Harness reference

Moved out of `docs/PROJECT_STATE.md` 2026-08-13: this is stable
reference about the checks themselves, not current state.

**The inventory.** Opt-in: `names_files` (⚠️ NOT `answer_contains` — one
filename is a substring of the other, so a contains-check on both is
satisfied by naming only the longer), `continuation`, `disclaims`,
`not_contains`. Always-on: `not_truncated`, `not_misattributed`,
`no_reasoning_leak`. Per-corpus: `no_fiction`. Automatic but rarely
applicable: `grounded`. Informational, never a gate:
`_reports_degraded_context`.

⚠️⚠️ **`no_false_limit` shipped as a gate on 2026-08-11 and was RETRACTED
the next morning.** It failed an answer for blaming a limit while its footer
showed ✅, on the theory the limit must be invented; the compaction work that
followed says those answers are most likely telling the TRUTH about a thinned
context. **A harness that cannot see the input must not call the output a
lie.**

⚠️ **Two text normalisations, both found by a false FAIL.** A check reading
the END of an answer must call `_prose_region` (the footer opens
`\n\n---\n>`, not the banner separator; it also folds curly quotes). A
check matching a PHRASE must call `_unemphasised` too — `has **no**
transcript` defeats a regex needing its words adjacent. ⚠️ Both are kept
away from `_filenames_named` (`*` is a **delimiter** there) and
`_looks_truncated` (`**Summary:**` ends on a colon once stripped).

⚠️ **`no_fiction` ground truth**, needed whenever `_CORPUS_FICTIONS` is
touched: the Gracie clip is **visual only** — grappling, a pin, a
scoreboard, IBJJF signage — with **no result of any kind**; Carlsen plays
**White** and plays the London himself (3272 vs CM Shuvalov 2707, 3-minute
blitz, he wins). Everything else about them is invented. ⚠️ Update
`_KNOWN_UPLOADS` when the box's uploads change.

⚠️ **Three cases carry no case-SPECIFIC check on purpose** —
`unnamed-reference`, `two-file-compare`, `control-named-scoped`;
`test_every_video_case_is_checked_for_something_behavioural` pins the set.

### Writing a check

**Writing an eval check: match SHAPE, not wording, and measure it against the
answers archive BEFORE wiring it in.** Three own-goals in one week — a
substring blacklist of observed phrasings that both models walked around on
the very next run (paraphrase space is infinite); an ASCII-only `don'?t` that
false-failed a textbook disclaimer (models write `don’t`); an opt-in check
that covered only the case I predicted while the behaviour moved to another.
**A check that false-fails a good answer is as damaging as one that misses a
bad one — it is the one that gets deleted.** So: describe the property
(`continuation`, `not_misattributed`), not the sentence; run the candidate
over `~/Downloads/Telegram Desktop/*onbox-answers.md` — 55 files, every suite
— and count true vs false positives first; make it ALWAYS-ON with an opt-out
when the behaviour could show up in any case. ⚠️ The one legitimate
blacklist is FALSE FACTS about a fixed corpus: the files never change, so the
set of untruths is bounded. Phrasings are not. ⚠️ And attach that blacklist
to the CORPUS, not to the case where you last saw the invention — five times
running, the next one landed somewhere else (`no_fiction`, `"corpus":
"video"`). Ground truth for it is the artifact summaries on the OWUI upload
page, which is what the model is actually given.
⚠️ **Normalise before matching a phrase; never let a check depend on
formatting.** Curly apostrophes and markdown emphasis have each cost several
runs of silent false fails — `don’t` and `has **no** transcript` both defeat
a regex that wants its words adjacent. `_prose_region` folds the quotes,
`_unemphasised` drops the asterisks, and both stay away from filename
extraction, where `*` is a delimiter.
⚠️ **One clean archive flip does not justify a widening.** `you would like me
to` was adopted on a single genuine flip, was redundant on the very answer it
was added for, and passed the worst answer in the next run.
⚠️⚠️ **Never conclude "the model made this up" from the OUTPUT alone.** The
answer cannot distinguish an invention from an accurate report of a degraded
context, and a check built on that distinction will punish honesty — it did,
for one day, as `no_false_limit`. Log what the model RECEIVED
(`agentic.debug_context_trace`) before writing any check that calls an
answer false.
⚠️ **Correlate failures against tool-call count before theorising.** It cost
one query and overturned fourteen runs of conclusions.

