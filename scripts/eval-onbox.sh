#!/usr/bin/env bash
#
# eval-onbox.sh — run the live eval ON THE BOX and Telegram-notify on completion,
# in one command. See docs/campaign-2/phase-27-eval-on-box.md.
#
# Why this exists: the bare `docker run -d …` + a separately-pasted notify command
# is easy to half-forget (you get the run but no ping). This wraps both so the
# notification fires EVERY time, automatically — you only ever call this script.
#
# It:
#   - runs the audrey-eval image detached on ollama-net (immune to laptop net),
#   - waits for it to finish,
#   - pushes a Telegram message via the fleet-watchdog watcher bot,
#   - leaves the answers file in the mounted testing-out dir.
#
# Secrets are SOURCED at runtime, never baked in:
#   - the OWUI key from the eval env-file (EVAL_ENV below),
#   - the Telegram token/chat-id from the fleet-watchdog hub .env (WATCHDOG_ENV).
#
# USAGE (on the box):
#   scripts/eval-onbox.sh                                   # research protocol (default)
#   scripts/eval-onbox.sh audrey_deep eval_prompts_deep.json
#   MODEL=audrey_fast CASES=eval_prompts_fast.json scripts/eval-onbox.sh
#
#   # Code / topics protocols (LABEL de-collides the answers file from a plain
#   # same-day deep run — without it both would be <date>-audrey_deep-onbox-…):
#   MODEL=audrey_deep CASES=eval_prompts_code.json LABEL=code scripts/eval-onbox.sh
#   MODEL=audrey_deep CASES=eval_prompts_topics.json LABEL=topics scripts/eval-onbox.sh
#
#   # ARGS= forwards any extra harness flag. Sampling ONE case many times is
#   # how a real change gets told apart from run-to-run variance — a 24-case
#   # run at n=1 cannot resolve the ±5 swing this suite has:
#   MODEL=audrey_auto CASES=eval_prompts_video.json LABEL=paging \
#     ARGS='--only video-long-transcript-paging --repeat 5' \
#     scripts/eval-onbox.sh
#
#   # Per-model sweep (MODELS= adds --models + a results JSON next to the
#   # answers file; feed that JSON to scripts/eval_compare.py afterwards):
#   CASES=eval_prompts_code_models.json LABEL=code-sweep \
#     MODELS='audrey_passthrough/qwen3.8:latest,audrey_passthrough/kimi-k2.7-code:cloud' \
#     scripts/eval-onbox.sh
#
# NOTE: harness + case files are MOUNTED from ./scripts at run time, so editing
# or pulling them needs no rebuild. Rebuild only when the image's own contents
# change (the httpx pin, the base image):
#   docker compose --profile eval build audrey-eval
#
# ⚠️ NEVER `docker image prune -a` on this box. `-a` removes every image with
# no RUNNING container, and audrey-eval is build-only — so it is always
# "unused" and is always deleted, leaving `ERROR: image audrey-eval:latest not
# built` on the next run. Plain `docker image prune -f` (dangling only) is
# safe and is what compose.yaml's one-shot recipe uses.
#
# Run it detached so it survives a disconnect and still notifies:
#   nohup scripts/eval-onbox.sh >/mnt/user/appdata/audrey_ai_2.0/testing-out/last-run.log 2>&1 &
#
set -uo pipefail

# ── config (override via env) ───────────────────────────────────────────────
APPDATA="${APPDATA:-/mnt/user/appdata/audrey_ai_2.0}"
IMAGE="${IMAGE:-audrey-eval:latest}"
NETWORK="${NETWORK:-ollama-net}"
EVAL_ENV="${EVAL_ENV:-${APPDATA}/eval.env}"                 # OWUI base-url + sk- key
WATCHDOG_ENV="${WATCHDOG_ENV:-/mnt/user/appdata/fleet-watchdog/.env}"  # Telegram creds
OUT_DIR="${OUT_DIR:-${APPDATA}/testing-out}"
CONTAINER="${CONTAINER:-audrey-eval}"

# model + cases: first two positional args, or env, or the research defaults
MODEL="${MODEL:-${1:-audrey_research}}"
CASES="${CASES:-${2:-eval_prompts_protocol.json}}"
# LABEL names the output files (default: the model — the historical naming).
# Set it when the model alone is ambiguous (code/topics both run audrey_deep).
LABEL="${LABEL:-${MODEL}}"
# STAMP is date + time (YYYY-MM-DD-HHMMSS) so re-running the same protocol/LABEL
# on the same day no longer overwrites the previous answers file (the old
# date-only stamp collided — e.g. three writer-glm-hedgefix runs in one day all
# wrote the same filename). Still date-first, so files sort chronologically and
# the paired <date>-<desc>-report.md convention still matches on the date prefix.
#
# ⚠️ Taken from the CONTAINER's clock, not the host's. There are three clocks
# on this box: the Unraid host is PDT, the containers log MDT, and `docker
# inspect` reports UTC. A host-stamped filename therefore read about an hour
# EARLIER than the log lines from its own run, so correlating an answers file
# against `docker logs` meant doing the arithmetic every time. Asking the
# container what time it is needs no hardcoded zone and cannot drift if either
# clock is changed later. Falls back to the host clock, loudly, if it is down.
STAMP_FROM="${STAMP_FROM:-audrey-ai}"
STAMP="$(docker exec "${STAMP_FROM}" date +%F-%H%M%S 2>/dev/null || true)"
if [[ -z "${STAMP}" ]]; then
  STAMP="$(date +%F-%H%M%S)"
  echo "WARN: could not read ${STAMP_FROM}'s clock; stamping with the host's" \
       "(PDT — reads ~1h early against container logs)." >&2
fi
SAVE_FILE="${STAMP}-${LABEL}-onbox-answers.md"

# The per-case results JSON (machine-readable: checks, latency, source stats) is
# written for EVERY run, not just sweeps — it's the input for eval_compare.py and
# lets any two runs be diffed programmatically (single-model research re-runs used
# to leave only the human .md, so run-over-run comparison was eyeball-only).
RESULTS_FILE="${STAMP}-${LABEL}-onbox-results.json"

# MODELS (comma-separated) turns the run into a per-model sweep: the harness runs
# every case once per model. Empty = normal single-model run. Either way we emit
# --save-json below.
MODELS="${MODELS:-}"
SWEEP_ARGS=()
if [[ -n "${MODELS}" ]]; then
  SWEEP_ARGS=(--models "${MODELS}")
fi

# ARGS forwards extra harness flags verbatim. Added 2026-08-12: without it
# `--only` and `--repeat` were unreachable through this runner, so the
# diagnostic protocol they exist for (sample ONE case enough times to tell a
# real change from run-to-run variance) could not actually be run on the box.
#   ARGS='--only video-long-transcript-paging --repeat 5'
# Word-splitting is intentional — this is a flag list, not a single argument.
EXTRA_ARGS=()
if [[ -n "${ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=(${ARGS})
fi

# ── preflight ───────────────────────────────────────────────────────────────
# ⚠️ `die` EXITS BEFORE THE NOTIFY BLOCK AT THE END OF THIS SCRIPT. That was
# harmless while every `die` was an instant pre-flight check you would see in
# your own terminal — but the readiness gate below can fire after a 180s wait,
# on a run you launched with `nohup … &` and walked away from. On 2026-08-18
# that produced the worst possible outcome: no eval, no ping, and no signal at
# all until the models were noticed missing from OWUI by hand.
# ▶ So a setup failure notifies too, matching `probe-onbox.sh`, which has always
# reported exit 2 as "❌ setup error — nothing probed".
notify_setup_failure() {
  local msg="$1"
  [[ -f "${WATCHDOG_ENV}" ]] || return 0
  # shellcheck disable=SC1090
  set -a; . "${WATCHDOG_ENV}"; set +a
  [[ -n "${WATCHDOG_TOKEN:-}" && -n "${WATCHDOG_CHAT_ID:-}" ]] || return 0
  curl -s "https://api.telegram.org/bot${WATCHDOG_TOKEN}/sendMessage" \
    -d chat_id="${WATCHDOG_CHAT_ID}" \
    -d "text=❌ Audrey eval SETUP ERROR (exit 2) — nothing ran.
${msg}" >/dev/null || true
}

die() { echo "ERROR: $*" >&2; notify_setup_failure "$*"; exit 2; }
command -v docker >/dev/null || die "docker not found — run this on the box, not the laptop."
[[ -f "${EVAL_ENV}" ]] || die "eval env-file missing: ${EVAL_ENV} (see phase-27 step 2)."
docker image inspect "${IMAGE}" >/dev/null 2>&1 || die "image ${IMAGE} not built (see phase-27 step 1)."
mkdir -p "${OUT_DIR}"; chmod 700 "${OUT_DIR}"

# ── wait for the stack to be READY, not merely started ──────────────────────
# ⚠️ A CONTAINER THAT IS `Up` IS NOT A CONTAINER THAT IS LISTENING. On
# 2026-08-18 a run launched seconds after `up -d --force-recreate audrey-ai`
# fired all twelve cases at 14:50:32 while uvicorn finished binding at
# 14:50:33.9. Every case returned `HTTP 400 {"detail":"Open WebUI: Server
# Connection Error"}` and the answers file led with **"12 cases, 0 passed"** —
# a total verdict, in the harness's normal voice, on a suite that never reached
# a model. That is the failure this whole script exists to prevent: a run that
# looks like a result.
# ⚠️ The STAMP step above does NOT cover this. `docker exec … date` succeeds the
# instant the container starts, which is why the run was stamped 14:50:32 and
# still had nothing to talk to.
# Both containers are gated, for different reasons: `audrey-ai` is what was late
# here, and `open-webui` is what the harness actually talks to — an
# `allowed_models` change requires bouncing it, so it is routinely restarted
# moments before a run.
READY_TIMEOUT="${READY_TIMEOUT:-180}"
READY_CONTAINERS="${READY_CONTAINERS:-audrey-ai open-webui}"

wait_ready() {
  local name="$1" deadline=$(( SECONDS + READY_TIMEOUT )) state health
  if ! docker inspect "${name}" >/dev/null 2>&1; then
    echo ">> ready   : ${name} — no such container, skipping" >&2
    return 0
  fi
  while :; do
    state="$(docker inspect -f '{{.State.Status}}' "${name}" 2>/dev/null || echo missing)"
    # ⚠️ `.State.Health` is ABSENT unless the image declares a HEALTHCHECK, and
    # the template prints an empty string rather than failing — so an empty
    # health field means "no healthcheck", not "unhealthy". open-webui may or
    # may not declare one depending on how it was recreated on the Unraid UI,
    # so running is all that can be required of it.
    health="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{end}}' "${name}" 2>/dev/null || true)"
    if [[ "${state}" == "running" && ( -z "${health}" || "${health}" == "healthy" ) ]]; then
      echo ">> ready   : ${name} (${health:-no healthcheck})"
      return 0
    fi
    if (( SECONDS >= deadline )); then
      die "${name} not ready after ${READY_TIMEOUT}s (status=${state}, health=${health:-none}). Nothing ran — re-run when it is up, or SKIP_READY=1 to bypass."
    fi
    sleep 2
  done
}

if [[ "${SKIP_READY:-0}" == "1" ]]; then
  echo ">> ready   : SKIPPED (SKIP_READY=1) — a 0/N run may mean the stack, not the model"
else
  for _c in ${READY_CONTAINERS}; do wait_ready "${_c}"; done
fi

# Fixed container name → remove any stopped prior one so re-runs don't collide.
docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true

# ── the harness comes from the REPO, not the image ──────────────────────────
#
# `Dockerfile.eval` still COPYs the harness and case files in, so the image
# stays self-contained and runnable on its own. But mounting `scripts/` over
# `/eval` means an edit to `eval_research.py` or a case file is live on the
# next run with no rebuild — which is the whole reason `LIVE_SCRIPTS` exists.
#
# ⚠️ The failure this removes was SILENT. Case files are baked, so a pulled
# change that was not followed by `docker compose --profile eval build` ran
# the OLD suite and reported a clean pass — the new checks simply were not
# there. Same trap in a different coat as the 2026-08-08 COPY-list omission.
# A stale image now cannot happen; a missing repo dir fails loudly instead.
#
# LIVE_SCRIPTS=0 falls back to the baked copy — for reproducing an older run
# against the image as it was built.
SCRIPT_DIR="${SCRIPT_DIR:-${APPDATA}/scripts}"
SCRIPT_MOUNT=()
if [[ "${LIVE_SCRIPTS:-1}" == "1" ]]; then
  [[ -f "${SCRIPT_DIR}/eval_research.py" ]] \
    || die "LIVE_SCRIPTS=1 but ${SCRIPT_DIR}/eval_research.py is missing (set SCRIPT_DIR, or LIVE_SCRIPTS=0 to use the baked copy)."
  [[ -f "${SCRIPT_DIR}/${CASES}" ]] \
    || die "case file ${SCRIPT_DIR}/${CASES} not found — pulled it yet?"
  SCRIPT_MOUNT=(-v "${SCRIPT_DIR}:/eval:ro")
  echo ">> harness: ${SCRIPT_DIR} (live; LIVE_SCRIPTS=0 for the baked copy)"
else
  echo ">> harness: baked into ${IMAGE}"
fi

echo ">> running ${MODEL} (${CASES}) on the box → ${OUT_DIR}/${SAVE_FILE}"
docker run -d --name "${CONTAINER}" --network "${NETWORK}" \
  --env-file "${EVAL_ENV}" \
  -v "${OUT_DIR}:/out" \
  "${SCRIPT_MOUNT[@]}" \
  "${IMAGE}" \
    --model "${MODEL}" \
    --cases "/eval/${CASES}" \
    --save-file "/out/${SAVE_FILE}" \
    --save-json "/out/${RESULTS_FILE}" \
    "${SWEEP_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
  || die "docker run failed."

# ── wait, then notify (always) ──────────────────────────────────────────────
echo ">> waiting for ${CONTAINER} to finish…"
rc="$(docker wait "${CONTAINER}")"
echo ">> ${CONTAINER} exited: ${rc}"

# Telegram push via the fleet-watchdog watcher bot. Non-fatal throughout — the
# eval already ran and its answers are saved; a failed send shouldn't mask that.
#   1. a SUMMARY message (verdict + the harness's "N/N cases passed" line), and
#   2. the ANSWERS FILE as a document (the full output — text messages cap at
#      4096 chars, but the answers file is ~68 KB, so it must go as sendDocument,
#      which allows up to 50 MB).
if [[ -f "${WATCHDOG_ENV}" ]]; then
  # shellcheck disable=SC1090
  set -a; . "${WATCHDOG_ENV}"; set +a
  if [[ -n "${WATCHDOG_TOKEN:-}" && -n "${WATCHDOG_CHAT_ID:-}" ]]; then
    # ⚠️ Exit 1 is NOT an error — it is the harness reporting that a case
    # failed a check, which is the normal result of a suite that measures
    # something. Flattening every non-zero code to one ⚠️ made "the eval keeps
    # exiting 1" read as a broken run for two days. Codes: 0 clean, 1 failed
    # checks, 2 setup, 3 crashed (see eval_research.py's EXIT CODES).
    case "${rc}" in
      0) verdict="✅ all checks passed" ;;
      1) verdict="⚠️ completed — some checks FAILED (read the [FAIL] blocks)" ;;
      2) verdict="❌ SETUP ERROR (exit 2) — nothing ran" ;;
      3) verdict="❌ HARNESS CRASHED (exit 3) — traceback in docker logs" ;;
      *) verdict="❌ exit ${rc}" ;;
    esac
    # Pull the harness's own pass/fail summary line from the container logs.
    summary="$(docker logs "${CONTAINER}" 2>&1 | grep -E 'cases passed' | tail -1)"
    api="https://api.telegram.org/bot${WATCHDOG_TOKEN}"

    curl -s "${api}/sendMessage" \
      -d chat_id="${WATCHDOG_CHAT_ID}" \
      -d "text=${verdict} Audrey eval ${MODEL} finished (exit ${rc}) → ${SAVE_FILE}
${summary:-（summary unavailable）}" \
      >/dev/null || echo "WARN: Telegram summary send failed." >&2

    # Attach the full answers file as a document.
    if [[ -f "${OUT_DIR}/${SAVE_FILE}" ]]; then
      curl -s "${api}/sendDocument" \
        -F chat_id="${WATCHDOG_CHAT_ID}" \
        -F document=@"${OUT_DIR}/${SAVE_FILE}" \
        >/dev/null || echo "WARN: Telegram document send failed (file still on box)." >&2
    fi
  else
    echo "WARN: WATCHDOG_TOKEN/CHAT_ID not in ${WATCHDOG_ENV}; skipped notify." >&2
  fi
else
  echo "WARN: ${WATCHDOG_ENV} not found; skipped Telegram notify." >&2
fi

echo ">> answers: ${OUT_DIR}/${SAVE_FILE}"
echo ">> results: ${OUT_DIR}/${RESULTS_FILE}  (feed to scripts/eval_compare.py)"
exit "${rc}"
