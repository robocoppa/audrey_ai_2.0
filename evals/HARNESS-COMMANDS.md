# Harness command reference

Every eval/probe harness in `evals/` and `scripts/`, where it runs, the exact command, and how
to tell running-vs-stalled. Two environments:

- **Laptop** (this repo, `.venv`) — hermetic checks and normal live Audrey API
  evals over `http://192.168.1.11:8000/v1`; save reports in `evals/results/`.
- **Box** (`root@Tower`, `/mnt/user/appdata/audrey_ai_2.0`) — direct Ollama
  probes requiring Docker DNS/GPU controls, or optional long detached API evals
  with Telegram notification. Saved full reports need a laptop copy step;
  compact terminal output may instead provide all necessary findings.

---

## 1. Research eval — laptop normally; optional `eval-onbox.sh` on Tower

The laptop harness is `evals/eval_research.py`, using the PAT in
`.env.test.local` and the working LAN/WARP route. Its `--save-file` and
`--save-json` destinations belong under local `evals/results/`. The following
Tower workflow remains available for a long run that must survive disconnects.

Full `audrey_research` pipeline over a case set. SLOW: ~70–280s per case, so a
5-case run is ~10–20 min. Runs detached, Telegram-pings on completion.

Update the Tower checkout first. The existing trace-diagnostic run remains:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
CASES=eval_prompts_writer_ab.json LABEL=research-trace-diag nohup bash evals/eval-onbox.sh \
  > testing-out/last-research-run.log 2>&1 &
```

For the ten-case research protocol instead:

```bash
CASES=eval_prompts_protocol.json LABEL=protocol nohup bash evals/eval-onbox.sh \
  > testing-out/last-research-run.log 2>&1 &
```

- MODEL defaults to `audrey_research`; override with `MODEL=audrey_deep` etc.
- Run from the repo root and use `bash evals/eval-onbox.sh`.
- Harness and case files are read-only runtime mounts from `evals/`; edits to
  these files need no image rebuild. Build only when image dependencies change.

**Running vs stalled:**
```bash
docker ps --filter name=audrey-eval --format '{{.Names}}\t{{.Status}}'   # Up = alive; gone = done/crashed
docker logs -f audrey-eval                                               # live per-case banners
cat testing-out/last-research-run.log                                    # launch errors (Exit 127 lives here)
```
- Normal: a case shows no new output for 1–4 min (it's grinding stages).
- Stalled: one case frozen >6–7 min (past the 360s `deep_worker` timeout), OR the
  container vanished with no answers file.
- Backend check: `docker run --rm --network ollama-net curlimages/curl:latest -s -o /dev/null -w "%{http_code}\n" http://audrey:8000/health` → want `200`.

**Output:** `testing-out/<stamp>-<LABEL>-onbox-answers.md` (+ `-results.json`).

---

## 2. Eval compare — `eval_compare.py`  (LAPTOP or BOX)

Builds a case-by-model table from one or more eval `--save-json` result files. Fast.
```bash
.venv/bin/python evals/eval_compare.py testing-out/<a>-results.json testing-out/<b>-results.json --out compare.md
```
No live services — pure JSON crunching. Done when it exits.

---

## 3. KB score probe — `kb_score_probe.py`  (BOX)

Probes `/v1/kb/query` with labeled on/off-domain queries; reports score
distributions + the safe-floor window (for tuning `kb.min_score`). Fast: 22 queries,
each capped at 30s, healthy run <2 min.

The script isn't inside the running container, so mount the host scripts dir into a
throwaway on `ollama-net` (has httpx, resolves `audrey`):
```bash
docker run --rm --network ollama-net \
  -v /mnt/user/appdata/audrey_ai_2.0/scripts/probes:/s \
  audrey-custom-tools \
  python3 /s/kb_score_probe.py --base-url http://audrey:8000
# machine-readable:  … python3 /s/kb_score_probe.py --save-json /s/../testing-out/kb-scores.json
```
Args: `--base-url --queries --top-k --timeout --save-json`. Query set:
`scripts/probes/kb_probe_queries.json` (edit to add queries after a corpus change).

**Running vs stalled:** prints one line per query as each completes. No query blocks
>30s (per-query timeout). No new line for >30s = stalled; else just slow. If it looks
frozen, health-check `audrey:8000/health` — a down backend times out every query.

---

## 4. Sources-block probe — `sources_block_probe.py`  (LAPTOP, hermetic)

Replays research ledgers through the REAL `_render_sources_block` to catch
Sources-rendering regressions. No box, no network.
```bash
.venv/bin/python scripts/probes/sources_block_probe.py                       # built-in fixtures (want 4/4)
.venv/bin/python scripts/probes/sources_block_probe.py --ledger dump.json    # replay a captured ledger dict
.venv/bin/python scripts/probes/sources_block_probe.py --ledger dump.json --expect-sources   # exit 1 if empty
```
Instant — it's a pure function over dicts. Exit 0 / "N/N passed" = done.

---

## 5. Other probes (LAPTOP, hermetic)

- **`probe_complexity_gate.py`** — exercises the fast/deep complexity gate.
  `.venv/bin/python scripts/probes/probe_complexity_gate.py`
- **`analyze_draft_sizes.py`** — draft-size stats from saved answers.
  `.venv/bin/python scripts/analysis/analyze_draft_sizes.py <answers.md>`
- **`measure_chunk_tails.py`** — KB chunk-tail measurement over a docs tree.
  `.venv/bin/python scripts/probes/measure_chunk_tails.py docs`
- **`check-lesson-links.py` / `check-lesson-conventions.py`** — lesson cite-drift +
  convention checks. `.venv/bin/python scripts/lessons/check-lesson-links.py [file …]`
- **`run_all_evals.sh`** — batch-runs the eval suites (see the script header).

All of these run to completion in seconds and print a result; there's no
"stalled" state to worry about — if it hasn't printed and exited, it errored.

---

## The one universal stall check

Anything that hits the box's live stack (research eval, KB probe) ultimately depends
on `audrey` being up. When in doubt:
```bash
docker run --rm --network ollama-net curlimages/curl:latest \
  -s -o /dev/null -w "%{http_code}\n" http://audrey:8000/health
```
`200` → backend fine, the harness is just working. Anything else → the stack is down
and every call is timing out, which *looks* like a stall but is a backend outage.
