# Harness command reference

Every eval/probe harness in `scripts/`, where it runs, the exact command, and how
to tell running-vs-stalled. Two environments:

- **Laptop** (this repo, `.venv`) — hermetic harnesses, no box needed.
- **Box** (`root@Tower`, `/mnt/user/appdata/audrey_ai_2.0`) — anything that hits the
  live stack (`audrey-ai:8000`, the KB, the research pipeline). The box has its own
  git checkout; `git pull` there first, and note newly-pulled scripts are NOT inside
  a running container until it's rebuilt (mount them into a throwaway instead).

---

## 1. Research eval — `eval-onbox.sh` → `eval_research.py`  (BOX)

Full `audrey_research` pipeline over a case set. SLOW: ~70–280s per case, so a
5-case run is ~10–20 min. Runs detached, Telegram-pings on completion.

```bash
cd /mnt/user/appdata/audrey_ai_2.0 && git pull
# rebuild the eval image only if the case file / harness changed since last build:
docker compose --profile eval build audrey-eval

# the trace-diagnostic set (attention/plate-tectonics/mrna + reasoning + gk):
CASES=eval_prompts_writer_ab.json LABEL=research-trace-diag nohup bash scripts/eval-onbox.sh \
  > testing-out/last-research-run.log 2>&1 &

# full 10-case protocol set instead:
CASES=eval_prompts_protocol.json LABEL=protocol nohup bash scripts/eval-onbox.sh \
  > testing-out/last-research-run.log 2>&1 &
```
- MODEL defaults to `audrey_research`; override with `MODEL=audrey_deep` etc.
- Run from the repo root and use `bash scripts/…` (avoids exec-bit / PATH `Exit 127`).
- Case files live in `scripts/eval_prompts*.json` and are BAKED into the eval image
  — that's why a case-file change needs the `--profile eval build`.

**Running vs stalled:**
```bash
docker ps --filter name=audrey-eval --format '{{.Names}}\t{{.Status}}'   # Up = alive; gone = done/crashed
docker logs -f audrey-eval                                               # live per-case banners
cat testing-out/last-research-run.log                                    # launch errors (Exit 127 lives here)
```
- Normal: a case shows no new output for 1–4 min (it's grinding stages).
- Stalled: one case frozen >6–7 min (past the 360s `deep_worker` timeout), OR the
  container vanished with no answers file.
- Backend check: `docker run --rm --network ollama-net curlimages/curl:latest -s -o /dev/null -w "%{http_code}\n" http://audrey-ai:8000/health` → want `200`.

**Output:** `testing-out/<stamp>-<LABEL>-onbox-answers.md` (+ `-results.json`).

---

## 2. Eval compare — `eval_compare.py`  (LAPTOP or BOX)

Builds a case-by-model table from one or more eval `--save-json` result files. Fast.
```bash
.venv/bin/python scripts/eval_compare.py testing-out/<a>-results.json testing-out/<b>-results.json --out compare.md
```
No live services — pure JSON crunching. Done when it exits.

---

## 3. KB score probe — `kb_score_probe.py`  (BOX)

Probes `/v1/kb/query` with labeled on/off-domain queries; reports score
distributions + the safe-floor window (for tuning `kb.min_score`). Fast: 22 queries,
each capped at 30s, healthy run <2 min.

The script isn't inside the running container, so mount the host scripts dir into a
throwaway on `ollama-net` (has httpx, resolves `audrey-ai`):
```bash
docker run --rm --network ollama-net \
  -v /mnt/user/appdata/audrey_ai_2.0/scripts:/s \
  audrey-custom-tools \
  python3 /s/kb_score_probe.py --base-url http://audrey-ai:8000
# machine-readable:  … python3 /s/kb_score_probe.py --save-json /s/../testing-out/kb-scores.json
```
Args: `--base-url --queries --top-k --timeout --save-json`. Query set:
`scripts/kb_probe_queries.json` (edit to add queries after a corpus change).

**Running vs stalled:** prints one line per query as each completes. No query blocks
>30s (per-query timeout). No new line for >30s = stalled; else just slow. If it looks
frozen, health-check `audrey-ai:8000/health` — a down backend times out every query.

---

## 4. Sources-block probe — `sources_block_probe.py`  (LAPTOP, hermetic)

Replays research ledgers through the REAL `_render_sources_block` to catch
Sources-rendering regressions. No box, no network.
```bash
.venv/bin/python scripts/sources_block_probe.py                       # built-in fixtures (want 4/4)
.venv/bin/python scripts/sources_block_probe.py --ledger dump.json    # replay a captured ledger dict
.venv/bin/python scripts/sources_block_probe.py --ledger dump.json --expect-sources   # exit 1 if empty
```
Instant — it's a pure function over dicts. Exit 0 / "N/N passed" = done.

---

## 5. Other probes (LAPTOP, hermetic)

- **`probe_complexity_gate.py`** — exercises the fast/deep complexity gate.
  `.venv/bin/python scripts/probe_complexity_gate.py`
- **`analyze_draft_sizes.py`** — draft-size stats from saved answers.
  `.venv/bin/python scripts/analyze_draft_sizes.py <answers.md>`
- **`measure_chunk_tails.py`** — KB chunk-tail measurement over a docs tree.
  `.venv/bin/python scripts/measure_chunk_tails.py docs`
- **`check-lesson-links.py` / `check-lesson-conventions.py`** — lesson cite-drift +
  convention checks. `.venv/bin/python scripts/check-lesson-links.py [file …]`
- **`run_all_evals.sh`** — batch-runs the eval suites (see the script header).

All of these run to completion in seconds and print a result; there's no
"stalled" state to worry about — if it hasn't printed and exited, it errored.

---

## The one universal stall check

Anything that hits the box's live stack (research eval, KB probe) ultimately depends
on `audrey-ai` being up. When in doubt:
```bash
docker run --rm --network ollama-net curlimages/curl:latest \
  -s -o /dev/null -w "%{http_code}\n" http://audrey-ai:8000/health
```
`200` → backend fine, the harness is just working. Anything else → the stack is down
and every call is timing out, which *looks* like a stall but is a backend outage.
