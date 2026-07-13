#!/usr/bin/env bash
#
# run_all_evals.sh — run the full live eval suite (research + deep + fast) in
# one shot, writing all three answers files into docs/testing/ under today's
# date. This is the one-command "exercise every mode against the live box"
# entry point; it just chains scripts/eval_research.py three times with the
# right --model / --cases / --save-file for each protocol.
#
# WHAT IT DOES NOT DO: it does not grade quality. Like the harness it wraps, it
# runs structural checks and saves the answers; the quality read is yours (write
# the paired <date>-<mode>-report.md by hand — see docs/testing/README.md).
#
# PREREQUISITES (same as the harness — see docs/testing/README.md):
#   - You are on the LAN/VPN and can reach the box (default 192.168.1.11:8080).
#   - .env.test.local exists at the repo root with AUDREY_EVAL_BASE_URL and
#     AUDREY_EVAL_API_KEY (the harness auto-loads it). This script does a
#     preflight check and refuses to start if it's missing.
#   - .venv exists (.venv/bin/python). Run from the repo root.
#
# USAGE
#   scripts/run_all_evals.sh                 # the default trio (research deep fast), today's date
#   scripts/run_all_evals.sh research deep   # only the named protocols, in order
#   scripts/run_all_evals.sh code code-hard  # opt-in protocols: code, code-hard, topics (not in the default trio)
#   DATE=2026-07-01 scripts/run_all_evals.sh # override the date stamp
#   ONLY=euclid scripts/run_all_evals.sh research   # pass --only through
#
# Each protocol runs to completion even if an earlier one failed its checks; the
# script's exit code is non-zero if ANY protocol had a failing case (so it can
# gate CI / a pre-deploy check). A short PASS/FAIL summary prints at the end.
#
# RUNTIME: the full suite is slow — deep/research cases are ~90–180s each, and
# the default trio is ~40 cases. Budget 60–90 minutes. Run it in the background
# and wait for completion; do not foreground-block on it.
#
set -uo pipefail

# --- locate repo root (this script lives in scripts/) -----------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PY="${REPO_ROOT}/.venv/bin/python"
HARNESS="${REPO_ROOT}/scripts/eval_research.py"
TESTING_DIR="${REPO_ROOT}/docs/testing"
DATE="${DATE:-$(date +%F)}"
ONLY_FLAG=()
[[ -n "${ONLY:-}" ]] && ONLY_FLAG=(--only "${ONLY}")

# --- protocol table: name -> model + cases file -----------------------------
# Keep in sync with docs/testing/README.md "The three protocols".
declare -A MODEL=(
  [research]="audrey_research"
  [deep]="audrey_deep"
  [fast]="audrey_fast"
  [code]="audrey_deep"
  [code-hard]="audrey_deep"
  [topics]="audrey_deep"
)
declare -A CASES=(
  [research]="scripts/eval_prompts_protocol.json"
  [deep]="scripts/eval_prompts_deep.json"
  [fast]="scripts/eval_prompts_fast.json"
  [code]="scripts/eval_prompts_code.json"
  [code-hard]="scripts/eval_prompts_code_hard.json"
  [topics]="scripts/eval_prompts_topics.json"
)

# protocols to run: CLI args if given, else the default trio. code, code-hard,
# and topics are OPT-IN (name them explicitly) so the routine full-suite runtime
# doesn't grow; the deep protocol carries 2 coding anchor cases so every default
# run still exercises deep-mode coding. code = the easy plumbing/regression tier;
# code-hard = the discriminating tier for lineup optimization (edge cases, subtle
# bugs). Per-model sweeps don't run through this script — see
# docs/testing/README.md "Per-model sweeps".
if [[ $# -gt 0 ]]; then
  PROTOCOLS=("$@")
else
  PROTOCOLS=(research deep fast)
fi

# --- preflight: fail fast and loud, before any slow network work ------------
die() { echo "ERROR: $*" >&2; exit 2; }

[[ -x "${PY}" ]] || die ".venv/bin/python not found — create the venv (uv sync) and run from the repo root."
[[ -f "${HARNESS}" ]] || die "scripts/eval_research.py not found — are you in the repo?"
[[ -f "${REPO_ROOT}/.env.test.local" ]] || die ".env.test.local missing at repo root. See docs/testing/README.md for setup (AUDREY_EVAL_BASE_URL + AUDREY_EVAL_API_KEY)."

for p in "${PROTOCOLS[@]}"; do
  [[ -n "${MODEL[$p]:-}" ]] || die "unknown protocol '$p' (valid: research deep fast code code-hard topics)"
  [[ -f "${REPO_ROOT}/${CASES[$p]}" ]] || die "cases file missing for '$p': ${CASES[$p]}"
done

mkdir -p "${TESTING_DIR}"

echo "=================================================================="
echo " Audrey live eval suite"
echo " date stamp : ${DATE}"
echo " protocols  : ${PROTOCOLS[*]}"
echo " answers ->  : ${TESTING_DIR}/${DATE}-<protocol>-answers.md"
[[ -n "${ONLY:-}" ]] && echo " --only     : ${ONLY}"
echo " NOTE: slow (~60–90 min for the full suite). Quality read is yours."
echo "=================================================================="
echo

declare -A RESULT
overall=0

for p in "${PROTOCOLS[@]}"; do
  save_file="${TESTING_DIR}/${DATE}-${p}-answers.md"
  echo "------------------------------------------------------------------"
  echo ">> ${p}  (model=${MODEL[$p]}, cases=${CASES[$p]})"
  echo "------------------------------------------------------------------"
  "${PY}" "${HARNESS}" \
    --model "${MODEL[$p]}" \
    --cases "${CASES[$p]}" \
    --save-file "${save_file}" \
    "${ONLY_FLAG[@]}"
  rc=$?
  if [[ ${rc} -eq 0 ]]; then
    RESULT[$p]="PASS"
  else
    RESULT[$p]="FAIL (exit ${rc})"
    overall=1
  fi
  echo ">> ${p}: ${RESULT[$p]}  ->  ${save_file}"
  echo
done

echo "=================================================================="
echo " Suite summary (${DATE})"
for p in "${PROTOCOLS[@]}"; do
  printf "   %-10s %s\n" "${p}" "${RESULT[$p]}"
done
echo "=================================================================="
echo
echo "Next: write the paired report(s) by hand — the quality read."
echo "  ${TESTING_DIR}/${DATE}-<protocol>-report.md"
echo "See docs/testing/README.md > 'The run/diff workflow'."

exit ${overall}
