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
#   # Per-model sweep (MODELS= adds --models + a results JSON next to the
#   # answers file; feed that JSON to scripts/eval_compare.py afterwards):
#   CASES=eval_prompts_code_models.json LABEL=code-sweep \
#     MODELS='audrey_passthrough/qwen3-coder-next:latest,audrey_passthrough/kimi-k2.7-code:cloud' \
#     scripts/eval-onbox.sh
#
# NOTE: the case files are BAKED into the audrey-eval image — after pulling new
# protocols (or harness changes), rebuild once:
#   docker compose --profile eval build audrey-eval
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
DATE="$(date +%F)"
SAVE_FILE="${DATE}-${LABEL}-onbox-answers.md"

# MODELS (comma-separated) turns the run into a per-model sweep: the harness
# runs every case once per model and we also save the per-case results JSON
# (the input for scripts/eval_compare.py). Empty = normal single-model run.
MODELS="${MODELS:-}"
SWEEP_ARGS=()
if [[ -n "${MODELS}" ]]; then
  SWEEP_ARGS=(--models "${MODELS}" --save-json "/out/${DATE}-${LABEL}-onbox-results.json")
fi

# ── preflight ───────────────────────────────────────────────────────────────
die() { echo "ERROR: $*" >&2; exit 2; }
command -v docker >/dev/null || die "docker not found — run this on the box, not the laptop."
[[ -f "${EVAL_ENV}" ]] || die "eval env-file missing: ${EVAL_ENV} (see phase-27 step 2)."
docker image inspect "${IMAGE}" >/dev/null 2>&1 || die "image ${IMAGE} not built (see phase-27 step 1)."
mkdir -p "${OUT_DIR}"; chmod 700 "${OUT_DIR}"

# Fixed container name → remove any stopped prior one so re-runs don't collide.
docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true

echo ">> running ${MODEL} (${CASES}) on the box → ${OUT_DIR}/${SAVE_FILE}"
docker run -d --name "${CONTAINER}" --network "${NETWORK}" \
  --env-file "${EVAL_ENV}" \
  -v "${OUT_DIR}:/out" \
  "${IMAGE}" \
    --model "${MODEL}" \
    --cases "/eval/${CASES}" \
    --save-file "/out/${SAVE_FILE}" \
    "${SWEEP_ARGS[@]}" \
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
    verdict="✅"; [[ "${rc}" != "0" ]] && verdict="⚠️"
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
exit "${rc}"
