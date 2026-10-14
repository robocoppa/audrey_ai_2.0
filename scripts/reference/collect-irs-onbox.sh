#!/usr/bin/env bash
# Short checks run here; long public collection runs use nohup.
set -euo pipefail
IFS=$'\n\t'
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/mnt/user/knowledge/reference-libraries/tax/us/federal/irs}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/testing-out/reference-archives}"
IMAGE="${IMAGE:-audrey:latest}"
COLLECTOR_CONTAINER="${COLLECTOR_CONTAINER:-audrey-irs-archive}"
WATCHDOG_ENV="${WATCHDOG_ENV:-/mnt/user/appdata/fleet-watchdog/.env}"
NOTIFY="${NOTIFY:-1}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)-$$"
REPORT="${OUT_DIR}/${STAMP}-irs-archive-report.json"
LOG="${OUT_DIR}/${STAMP}-irs-archive.log"

notify() (
    local message="$1" document="${2:-}" api reply
    [[ "${NOTIFY}" == "1" ]] || exit 0
    if [[ ! -f "${WATCHDOG_ENV}" ]]; then
        echo "WARN: notification settings unavailable; report remains on Tower." >&2
        exit 0
    fi
    set +u
    set -a
    # shellcheck disable=SC1090
    source "${WATCHDOG_ENV}"
    set +a
    set -u
    if [[ -z "${WATCHDOG_TOKEN:-}" || -z "${WATCHDOG_CHAT_ID:-}" ]]; then
        echo "WARN: notification credentials unavailable; report remains on Tower." >&2
        exit 0
    fi
    api="https://api.telegram.org/bot${WATCHDOG_TOKEN}"
    if ! reply="$(curl --fail --silent --connect-timeout 10 --max-time 30 \
        "${api}/sendMessage" --data-urlencode "chat_id=${WATCHDOG_CHAT_ID}" \
        --data-urlencode "text=${message}" 2>/dev/null)"; then
        echo "WARN: Telegram message delivery failed." >&2
    elif ! grep -Eq '"ok"[[:space:]]*:[[:space:]]*true' <<<"${reply}"; then
        echo "WARN: Telegram rejected the message." >&2
    fi
    if [[ -n "${document}" && -f "${document}" ]]; then
        if ! reply="$(curl --fail --silent --connect-timeout 10 --max-time 60 \
            "${api}/sendDocument" --form-string "chat_id=${WATCHDOG_CHAT_ID}" \
            -F "document=@${document}" 2>/dev/null)"; then
            echo "WARN: Telegram report delivery failed; report remains on Tower." >&2
        elif ! grep -Eq '"ok"[[:space:]]*:[[:space:]]*true' <<<"${reply}"; then
            echo "WARN: Telegram rejected the report; report remains on Tower." >&2
        fi
    fi
    exit 0
)

finish() {
    local code=$? verdict
    trap - EXIT
    if [[ -d "${OUT_DIR}" && ! -s "${REPORT}" ]]; then
        printf '{"schema":1,"status":"runner_failure","exit_code":%d}\n' "${code}" >"${REPORT}" || true
    fi
    case "${code}" in
        0) verdict="✅ collection run finished; see report for remaining coverage" ;;
        1) verdict="⚠️ collection finished with findings" ;;
        2) verdict="❌ SETUP ERROR" ;;
        130|143) verdict="⛔ INTERRUPTED" ;;
        *) verdict="❌ CRASHED" ;;
    esac
    notify "${verdict} Audrey IRS archive finished (exit ${code}) → ${REPORT}" "${REPORT}" || true
    exit "${code}"
}
collector_worker_pid=""

interrupt() {
    local code="$1"
    trap '' INT TERM
    if [[ -n "${collector_worker_pid}" ]]; then
        docker stop --time 5 "${COLLECTOR_CONTAINER}" >/dev/null 2>&1 || true
        kill "${collector_worker_pid}" 2>/dev/null || true
        wait "${collector_worker_pid}" 2>/dev/null || true
    fi
    exit "${code}"
}

trap finish EXIT
trap 'interrupt 130' INT
trap 'interrupt 143' TERM

die() {
    echo "SETUP ERROR: $*" >&2
    exit 2
}

[[ "${NOTIFY}" == "0" || "${NOTIFY}" == "1" ]] || die "NOTIFY must be 0 or 1"
command -v docker >/dev/null || die "Docker is required on Tower"
command -v tee >/dev/null || die "tee is required"
[[ "${ARCHIVE_ROOT}" == /* && "${OUT_DIR}" == /* ]] || die "Archive/report roots must be absolute paths"
[[ ! -L "${ARCHIVE_ROOT}" && ! -L "${OUT_DIR}" ]] || die "Archive/report roots cannot be symlinks"
mkdir -p -- "${ARCHIVE_ROOT}" "${OUT_DIR}" || die "Could not create archive/report directories"
[[ -w "${ARCHIVE_ROOT}" && -w "${OUT_DIR}" ]] || die "Archive/report directories must be writable"
docker image inspect "${IMAGE}" >/dev/null 2>&1 || die "Build the Audrey image first: docker compose build audrey"
if docker container inspect "${COLLECTOR_CONTAINER}" >/dev/null 2>&1; then
    die "An IRS collector container already exists; finish it before starting another"
fi

notify "▶ Audrey IRS archive started. Public documents only; no model calls. Report: ${REPORT}" || true
echo "Archive: ${ARCHIVE_ROOT}"
echo "Report: ${REPORT}"
echo "Log: ${LOG}"
set +e
(
    docker run --rm --name "${COLLECTOR_CONTAINER}" \
        --user "$(id -u):$(id -g)" \
        --read-only --no-healthcheck --cap-drop ALL \
        --security-opt no-new-privileges --pids-limit 64 --memory 768m \
        --tmpfs /tmp:rw,nosuid,nodev,size=128m \
        --mount "type=bind,src=${ARCHIVE_ROOT},dst=/archive" \
        --mount "type=bind,src=${OUT_DIR},dst=/reports" \
        --entrypoint /opt/venv/bin/python \
        "${IMAGE}" -m audrey.references.irs_collect \
        --root /archive --report "/reports/$(basename -- "${REPORT}")" "$@" \
        2>&1 | tee "${LOG}"
    pipeline_codes=("${PIPESTATUS[@]}")
    code="${pipeline_codes[0]}"
    if [[ "${code}" == "0" && "${pipeline_codes[1]}" != "0" ]]; then
        code=3
    fi
    exit "${code}"
) &
collector_worker_pid=$!
wait "${collector_worker_pid}"
code=$?
collector_worker_pid=""
set -e
if [[ "${code}" == "0" && ! -s "${REPORT}" ]]; then
    echo "ERROR: Collector did not write its report." >&2
    code=3
fi
exit "${code}"

