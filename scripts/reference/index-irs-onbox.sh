#!/usr/bin/env bash
# Short checks run here; long public indexing runs use the existing nohup flow.
set -euo pipefail
IFS=$'\n\t'
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
KNOWLEDGE_ROOT="${KNOWLEDGE_ROOT:-/mnt/user/knowledge}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-${KNOWLEDGE_ROOT}/reference-libraries/tax/us/federal/irs}"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/config.yaml}"
INDEX_HOST="${INDEX_HOST:-/mnt/user/appdata/runtime/reference-libraries}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/testing-out/reference-indexes}"
IMAGE="${IMAGE:-audrey:latest}"
NETWORK="${NETWORK:-ollama-net}"
INDEX_CONTAINER="${INDEX_CONTAINER:-audrey-reference-index}"
WATCHDOG_ENV="${WATCHDOG_ENV:-/mnt/user/appdata/fleet-watchdog/.env}"
NOTIFY="${NOTIFY:-1}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)-$$"
REPORT_NAME="${STAMP}-reference-index-report.json"
STATE_REPORT="${INDEX_HOST}/reports/${REPORT_NAME}"
REPORT="${OUT_DIR}/${REPORT_NAME}"
LOG="${OUT_DIR}/${STAMP}-reference-index.log"

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
    if [[ -s "${STATE_REPORT}" && ! -L "${STATE_REPORT}" ]]; then
        if ! cp -- "${STATE_REPORT}" "${REPORT}"; then
            echo "ERROR: Could not copy the index report from public index state." >&2
            [[ "${code}" != "0" ]] || code=3
        fi
    fi
    if [[ -d "${OUT_DIR}" && ! -s "${REPORT}" ]]; then
        printf '{"schema":1,"status":"runner_failure","exit_code":%d}\n' "${code}" >"${REPORT}" || true
    fi
    case "${code}" in
        0) verdict="✅ reference index run finished; see report for publication status" ;;
        1) verdict="⚠️ reference index run finished with findings" ;;
        2) verdict="❌ SETUP ERROR" ;;
        130|143) verdict="⛔ INTERRUPTED" ;;
        *) verdict="❌ CRASHED" ;;
    esac
    notify "${verdict} Audrey reference index finished (exit ${code}) → ${REPORT}" "${REPORT}" || true
    exit "${code}"
}
index_worker_pid=""

interrupt() {
    local code="$1"
    trap '' INT TERM
    if [[ -n "${index_worker_pid}" ]]; then
        docker stop --time 5 "${INDEX_CONTAINER}" >/dev/null 2>&1 || true
        kill "${index_worker_pid}" 2>/dev/null || true
        wait "${index_worker_pid}" 2>/dev/null || true
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
[[ "${OUT_DIR}" == /* && ! -L "${OUT_DIR}" ]] || die "Report root must be an absolute non-symlink path"
mkdir -p -- "${OUT_DIR}" || die "Could not create report directory"
for directory in "${KNOWLEDGE_ROOT}" "${ARCHIVE_ROOT}" "${INDEX_HOST}"; do
    [[ "${directory}" == /* && -d "${directory}" && ! -L "${directory}" ]] ||
        die "Knowledge, archive and index directories must already exist as absolute non-symlink paths"
done
[[ "${CONFIG_FILE}" == /* && -r "${CONFIG_FILE}" && -f "${CONFIG_FILE}" ]] ||
    die "Readable repository configuration is required"
[[ -f "${ARCHIVE_ROOT}/catalogs/archive.sqlite" ]] ||
    die "Collect the IRS archive before indexing"
docker image inspect "${IMAGE}" >/dev/null 2>&1 ||
    die "Build the current Audrey image first: docker compose build audrey"
docker network inspect "${NETWORK}" >/dev/null 2>&1 ||
    die "The Ollama/Qdrant Docker network is unavailable"
if docker container inspect "${INDEX_CONTAINER}" >/dev/null 2>&1; then
    die "A reference index container already exists; finish it before starting another"
fi
# The image's unprivileged user must be able to write public index state.
# A backend deployment creates this directory with the correct owner.
# Embeddings run in Ollama. Local numerical libraries need no parallel pool;
# high-core hosts can otherwise exhaust the index container's thread budget.
docker run --rm --network none --read-only --no-healthcheck --cap-drop ALL \
    --security-opt no-new-privileges \
    -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 \
    --mount "type=bind,src=${INDEX_HOST},dst=/data/reference-libraries" \
    --entrypoint /opt/venv/bin/python "${IMAGE}" -c \
    'import importlib.util, tempfile; from pathlib import Path; spec = importlib.util.find_spec("audrey.references.index_cli"); handle = tempfile.NamedTemporaryFile(prefix=".index-preflight-", dir="/data/reference-libraries"); handle.close(); raise SystemExit(0 if spec else 2)' \
    >/dev/null 2>&1 ||
    die "Current index CLI and writable public index state are required; deploy the backend first"

notify "▶ Audrey reference index started. Public references only; local embeddings. Report: ${REPORT}" || true
echo "Archive: ${ARCHIVE_ROOT}"
echo "Index state: ${INDEX_HOST}"
echo "Report: ${REPORT}"
echo "Log: ${LOG}"
set +e
(
    connection_flags=()
    for setting in OLLAMA_HOST QDRANT_HOST QDRANT_PORT; do
        if [[ -n "${!setting:-}" ]]; then
            connection_flags+=(-e "${setting}")
        fi
    done
    docker run --rm --name "${INDEX_CONTAINER}" --network "${NETWORK}" \
        --read-only --no-healthcheck --cap-drop ALL \
        --security-opt no-new-privileges --pids-limit 64 --memory 2g \
        -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 \
        --tmpfs /tmp:rw,nosuid,nodev,size=128m \
        --mount "type=bind,src=${KNOWLEDGE_ROOT},dst=/datasets,readonly" \
        --mount "type=bind,src=${CONFIG_FILE},dst=/app/config.yaml,readonly" \
        --mount "type=bind,src=${INDEX_HOST},dst=/data/reference-libraries" \
        "${connection_flags[@]}" \
        --entrypoint /opt/venv/bin/python \
        "${IMAGE}" -m audrey.references.index_cli \
        --library irs-federal --catalog current \
        --report "/data/reference-libraries/reports/${REPORT_NAME}" "$@" \
        2>&1 | tee "${LOG}"
    pipeline_codes=("${PIPESTATUS[@]}")
    code="${pipeline_codes[0]}"
    if [[ "${code}" == "0" && "${pipeline_codes[1]}" != "0" ]]; then
        code=3
    fi
    exit "${code}"
) &
index_worker_pid=$!
wait "${index_worker_pid}"
code=$?
index_worker_pid=""
set -e
if [[ "${code}" == "0" && ! -s "${STATE_REPORT}" ]]; then
    echo "ERROR: Indexer did not write its report." >&2
    code=3
fi
exit "${code}"

