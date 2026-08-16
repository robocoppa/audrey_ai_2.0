#!/usr/bin/env bash
#
# probe-onbox.sh — run any probe ON THE BOX, detached, and Telegram-notify on
# completion. Sibling of eval-onbox.sh, same conventions.
#
# WHY THIS EXISTS
#
# Every probe in scripts/ has the same three problems on this box:
#
#   1. ⚠️ **Unraid has no `python3`.** A probe cannot run on the host at all.
#      It has to run inside `audrey-ai`, which has Python, the `audrey`
#      package, and a route to Ollama over the compose network.
#   2. ⚠️ **The repo is NOT bind-mounted into `audrey-ai`** — only
#      `config.yaml`, `/data` and `/datasets`. So the probe has to be copied in
#      before it can run.
#   3. ⚠️ **A probe outlives the SSH session, and used not to.** Long runs died
#      to laptop-sleep process orphaning, which looked like a broken probe for
#      two days. `docker exec` without help ties the in-container process to
#      the exec client, so a dropped connection can take the run with it.
#
# ▶ **This script SELF-DETACHES.** You do not type `nohup … &` — it re-execs
# itself under `setsid nohup` on first call, prints where the log is, and hands
# your prompt straight back. Close the SSH session immediately; the run is
# owned by init, not by your shell. That is deliberate: an incantation you have
# to remember is one you will eventually forget, at the cost of a long run.
#
# ⚠️ Logs go to a BIND-MOUNTED path (`${OUT_DIR}` on the host), never to the
# container filesystem — a `docker compose up -d --build` would otherwise take
# the results with it.
#
# USAGE (on the box, from ${APPDATA}):
#
#   scripts/probe-onbox.sh router_probe.py MODEL=qwen3.5:4b BOTH=1 ROUNDS=3
#   scripts/probe-onbox.sh thinking_probe.py MODEL=qwen3.8:latest TOOLS=1
#   scripts/probe-onbox.sh check_model_inventory.py \
#       ARGS='--config /app/config.yaml --tags-url http://ollama:11434'
#
# Trailing KEY=VALUE pairs become environment for the probe. `ARGS=` is special:
# its contents are passed as command-line flags instead.
#
#   FOREGROUND=1 scripts/probe-onbox.sh …   # do not detach (for debugging)
#
set -uo pipefail

APPDATA="${APPDATA:-/mnt/user/appdata/audrey_ai_2.0}"
CONTAINER="${CONTAINER:-audrey-ai}"
OUT_DIR="${OUT_DIR:-${APPDATA}/testing-out/probes}"
WATCHDOG_ENV="${WATCHDOG_ENV:-/mnt/user/appdata/fleet-watchdog/.env}"

PROBE="${1:-}"
if [[ -z "${PROBE}" ]]; then
  echo "usage: $0 <probe-script.py> [KEY=VALUE …] [ARGS='--flags']" >&2
  echo "       probes live in ${APPDATA}/scripts/" >&2
  exit 2
fi
shift

PROBE_PATH="${APPDATA}/scripts/${PROBE}"
if [[ ! -f "${PROBE_PATH}" ]]; then
  echo "no such probe: ${PROBE_PATH}" >&2
  exit 2
fi

# ── self-detach ─────────────────────────────────────────────────────────────
# Re-exec under setsid+nohup so the run survives the SSH session. The marker
# env var stops the child from doing it again. `setsid` (not just nohup) so the
# process leaves the controlling terminal's session entirely — nohup alone only
# ignores SIGHUP, and a closed terminal can still take a process group with it.
STAMP="$(docker exec "${CONTAINER}" date +%Y-%m-%d-%H%M%S 2>/dev/null || date +%Y-%m-%d-%H%M%S)"
LABEL="${PROBE%.py}"
LOG="${OUT_DIR}/${STAMP}-${LABEL}.log"

if [[ -z "${_PROBE_DETACHED:-}" && -z "${FOREGROUND:-}" ]]; then
  mkdir -p "${OUT_DIR}"
  _PROBE_DETACHED=1 STAMP="${STAMP}" LOG="${LOG}" \
    setsid nohup "$0" "${PROBE}" "$@" >"${LOG}" 2>&1 &
  echo ">> ${LABEL} running detached (pid $!). Safe to close this session."
  echo ">> log: ${LOG}"
  echo ">>      tail -f ${LOG}"
  exit 0
fi

# ⚠️ The stamp is read from the CONTAINER clock, matching the eval answers
# files. This box runs three clocks — host PDT, container logs MDT, `docker
# inspect` UTC — so a host-stamped filename would not line up with the log
# lines inside the run it names.
mkdir -p "${OUT_DIR}"

# ── split trailing KEY=VALUE pairs into env, and pull ARGS= out ──────────────
ENV_FLAGS=()
PROBE_ARGS=""
for kv in "$@"; do
  case "${kv}" in
    ARGS=*) PROBE_ARGS="${kv#ARGS=}" ;;
    *=*)    ENV_FLAGS+=("-e" "${kv}") ;;
    *)      echo "WARN: ignoring '${kv}' — expected KEY=VALUE or ARGS='…'" >&2 ;;
  esac
done

echo ">> probe   : ${PROBE}"
echo ">> env     : ${ENV_FLAGS[*]:-（none）}"
echo ">> args    : ${PROBE_ARGS:-（none）}"
echo ">> started : $(date)"
echo

# ── copy the probe in and run it ────────────────────────────────────────────
# Copied fresh every run: the repo is not mounted, so a `git pull` on the host
# is invisible inside the container until this happens.
IN_CONTAINER="/tmp/probe-${STAMP}.py"
if ! docker cp "${PROBE_PATH}" "${CONTAINER}:${IN_CONTAINER}"; then
  echo "ERROR: could not copy ${PROBE} into ${CONTAINER} — is it running?" >&2
  exit 2
fi

# shellcheck disable=SC2086 — PROBE_ARGS is a deliberate word-split flag string
docker exec "${ENV_FLAGS[@]}" "${CONTAINER}" python3 "${IN_CONTAINER}" ${PROBE_ARGS}
rc=$?

docker exec "${CONTAINER}" rm -f "${IN_CONTAINER}" 2>/dev/null || true

echo
echo ">> finished: $(date)"
echo ">> exit    : ${rc}"

# ── notify (always; non-fatal) ──────────────────────────────────────────────
# The probe has already run and its log is saved — a failed send must not mask
# that. ⚠️ Non-zero is NOT necessarily an error: router_probe exits 1 for a
# DISQUALIFIED candidate and check_model_inventory exits 1 when config names a
# model Ollama does not have. Both are findings, which is the point.
if [[ -f "${WATCHDOG_ENV}" ]]; then
  # shellcheck disable=SC1090
  set -a; . "${WATCHDOG_ENV}"; set +a
  if [[ -n "${WATCHDOG_TOKEN:-}" && -n "${WATCHDOG_CHAT_ID:-}" ]]; then
    case "${rc}" in
      0) verdict="✅ clean" ;;
      1) verdict="⚠️ finished with FINDINGS (exit 1 — read the log, this is usually the point)" ;;
      2) verdict="❌ setup error (exit 2) — nothing probed" ;;
      *) verdict="❌ exit ${rc}" ;;
    esac
    # ⚠️ MESSAGE FORMAT IS eval-onbox.sh's, DELIBERATELY UNCHANGED — a one-line
    # verdict, then the probe's own summary line, then the full log as a
    # document. It reads well on a phone and is the shape already trusted; do
    # not "improve" it by inlining a log tail (tried 2026-08-15, reverted).
    # Text messages cap at 4096 chars anyway, which is why the log goes as
    # sendDocument rather than in the body.
    summary="$(grep -E 'parse rate|NOT ON THE BOX|every model named|reclaimable|DISQUALIFIED' \
                 "${LOG}" 2>/dev/null | tail -1)"
    [[ -z "${summary}" ]] && summary="$(grep -v '^[[:space:]]*$' "${LOG}" 2>/dev/null | tail -1)"
    api="https://api.telegram.org/bot${WATCHDOG_TOKEN}"

    curl -s "${api}/sendMessage" \
      -d chat_id="${WATCHDOG_CHAT_ID}" \
      -d "text=${verdict} Audrey probe ${LABEL} finished (exit ${rc}) → ${LOG}
${summary:-（summary unavailable）}" \
      >/dev/null || echo "WARN: Telegram summary send failed." >&2

    # Attach the full log as a document.
    if [[ -f "${LOG}" ]]; then
      curl -s "${api}/sendDocument" \
        -F chat_id="${WATCHDOG_CHAT_ID}" \
        -F document=@"${LOG}" \
        >/dev/null || echo "WARN: Telegram document send failed (log still on box)." >&2
    fi
  else
    echo "WARN: WATCHDOG_TOKEN/CHAT_ID not in ${WATCHDOG_ENV}; skipped notify." >&2
  fi
else
  echo "WARN: ${WATCHDOG_ENV} not found; skipped Telegram notify." >&2
fi

exit "${rc}"
