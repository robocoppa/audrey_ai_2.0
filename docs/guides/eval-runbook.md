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

| cases file | n | typical model | rough time |
|---|---|---|---|
| `eval_prompts_protocol.json` | 10 | `audrey_research` | **20–40 min** |
| `eval_prompts_deep.json` | 18 | `audrey_deep` | long |
| `eval_prompts_topics.json` | 13 | `audrey_deep` | long |
| `eval_prompts_video.json` | 12 | `audrey_auto` | ~10 min |
| `eval_prompts_fast.json` | 12 | `audrey_fast` | short |
| `eval_prompts_code.json` | 10 | `audrey_deep` | plumbing gate |
| `eval_prompts_code_hard.json` | 5 | `audrey_deep` | the discriminating tier |
| `eval_prompts_writer_ab.json` | 5 | `audrey_research` | writer A-B |
| `eval_prompts_local_models.json` | 12 | sweep | local bake-off |
| `eval_prompts_models_ab.json` | 9 | sweep | lineup A-B |

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
  MODELS='audrey_passthrough/nemotron-3.5-lightning:latest,audrey_passthrough/muse-glimmer:latest,audrey_passthrough/qwen3.6:35b' \
  scripts/eval-onbox.sh > testing-out/last-run.log 2>&1 &

# coding lineup
nohup env CASES=eval_prompts_code_models.json LABEL=code-sweep \
  MODELS='audrey_passthrough/qwen3-coder-next:latest,audrey_passthrough/qwen2.5-coder:32b,audrey_passthrough/qwen3.6:35b,audrey_passthrough/kimi-k2.7-code:cloud,audrey_passthrough/deepseek-v4-pro:cloud,audrey_passthrough/minimax-m3:cloud' \
  scripts/eval-onbox.sh > testing-out/last-code-sweep.log 2>&1 &

# general lineup A-B
nohup env CASES=eval_prompts_models_ab.json LABEL=models-ab \
  MODELS='audrey_passthrough/qwen3.6:35b,audrey_passthrough/deepseek-r1:32b,audrey_passthrough/qwen3-coder-next:latest,audrey_passthrough/deepseek-v4-pro:cloud,audrey_passthrough/kimi-k2.6:cloud,audrey_passthrough/glm-5.2:cloud' \
  scripts/eval-onbox.sh > testing-out/last-models-ab-run.log 2>&1 &

# virtual-model A-B (task role on vs off)
nohup env MODEL=audrey_video CASES=eval_prompts_video.json LABEL=video-ab \
  MODELS='audrey_video,audrey_auto' \
  scripts/eval-onbox.sh > testing-out/last-video-ab-run.log 2>&1 &
```

⚠️ A sweep multiplies cloud spend by the model count. Cloud credits are a hard
budget.

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

# which env overrides are actually live
docker compose logs audrey-ai | grep "ENV OVERRIDE"

# per-turn summary: task, confidence, mode, model, tool calls
docker compose logs audrey-ai | grep "chat.completions model="
```

⚠️ The Unraid shell has **no python3** — use `jq` or plain `grep`, never pipe
to `python3`.

---

## Compare two runs

```bash
uv run scripts/eval_compare.py testing-out/<a>-results.json testing-out/<b>-results.json
```

Only sweeps (`MODELS=`) write the results JSON.

⚠️ **A run's score is not a quality trend line.** It moves with what the checks
can see at least as much as with the answers. Before reading a delta, confirm
both runs used the same harness commit — a check added between them changes the
number without anything changing in Audrey.
