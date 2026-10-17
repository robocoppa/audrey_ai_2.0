#!/usr/bin/env bash
# Run after building the images and stopping the four application writers.
set -euo pipefail
IFS=$'\n\t'
umask 077
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_HOST="${DATA_HOST:-/mnt/user/appdata/runtime}"
TOOLS_HOST="${TOOLS_HOST:-/mnt/user/appdata/custom-tools}"
SCOPE="${SCOPE:-all}"
if [[ "$#" != "0" ]]; then
    if [[ "$#" != "2" || "$1" != "--scope" ]]; then
        echo "SETUP ERROR: usage: reembed-onbox.sh [--scope all|history]" >&2
        exit 2
    fi
    SCOPE="$2"
fi
case "${SCOPE}" in
    all) state_name="embedding-migration-qwen4"; run_name="text-reembed"; migration_label="text embedding migration" ;;
    history) state_name="history-reembed-nomic"; run_name="history-reembed"; migration_label="history embedding migration" ;;
    *) echo "SETUP ERROR: scope must be all or history" >&2; exit 2 ;;
esac
STATE_HOST="${STATE_HOST:-${DATA_HOST}/${state_name}}"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/config.yaml}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/testing-out/embedding-migrations}"
IMAGE="${IMAGE:-audrey:latest}"
NETWORK="${NETWORK:-ollama-net}"
MIGRATION_CONTAINER="${MIGRATION_CONTAINER:-audrey-${run_name}}"
WATCHDOG_ENV="${WATCHDOG_ENV:-/mnt/user/appdata/fleet-watchdog/.env}"
NOTIFY="${NOTIFY:-1}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)-$$"
REPORT_NAME="${STAMP}-${run_name}-report.json"
STATE_REPORT="${STATE_HOST}/reports/${REPORT_NAME}"
REPORT="${OUT_DIR}/${REPORT_NAME}"
LOG="${OUT_DIR}/${STAMP}-${run_name}.log"

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
            echo "ERROR: Could not copy the migration report from private state." >&2
            [[ "${code}" != "0" ]] || code=3
        fi
    fi
    if [[ -d "${OUT_DIR}" && ! -s "${REPORT}" ]]; then
        printf '{"schema":1,"status":"runner_failure","exit_code":%d}\n' "${code}" >"${REPORT}" || true
    fi
    case "${code}" in
        0) verdict="✅ ${migration_label} and restart finished" ;;
        1) verdict="⚠️ ${migration_label} finished with findings" ;;
        2) verdict="❌ SETUP ERROR" ;;
        130|143) verdict="⛔ INTERRUPTED" ;;
        *) verdict="❌ CRASHED" ;;
    esac
    notify "${verdict} Audrey ${migration_label} finished (exit ${code}) → ${REPORT}" "${REPORT}" || true
    exit "${code}"
}
migration_worker_pid=""

interrupt() {
    local code="$1"
    trap '' INT TERM
    if [[ -n "${migration_worker_pid}" ]]; then
        docker stop --time 10 "${MIGRATION_CONTAINER}" >/dev/null 2>&1 || true
        kill "${migration_worker_pid}" 2>/dev/null || true
        wait "${migration_worker_pid}" 2>/dev/null || true
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
docker info >/dev/null 2>&1 || die "Docker is unavailable"
[[ "${OUT_DIR}" == /* && ! -L "${OUT_DIR}" ]] || die "Report root must be an absolute non-symlink path"
mkdir -p -- "${OUT_DIR}" || die "Could not create report directory"
for directory in "${DATA_HOST}" "${TOOLS_HOST}"; do
    [[ "${directory}" == /* && -d "${directory}" && ! -L "${directory}" ]] ||
        die "Application data directories must already exist as absolute non-symlink paths"
done
[[ "${STATE_HOST}" == /* && ! -L "${STATE_HOST}" ]] || die "Migration state must be an absolute non-symlink path"
if [[ ! -d "${STATE_HOST}" ]]; then
    install -d -m 700 -o 99 -g 100 -- "${STATE_HOST}" || die "Cannot create private migration state"
fi
[[ -f "${CONFIG_FILE}" && -r "${CONFIG_FILE}" && ! -L "${CONFIG_FILE}" ]] || die "Readable repository configuration is required"
docker image inspect "${IMAGE}" >/dev/null 2>&1 || die "Build the prepared Audrey and custom-tools images first"
docker image inspect audrey-custom-tools >/dev/null 2>&1 || die "Build the prepared custom-tools image first"
docker network inspect "${NETWORK}" >/dev/null 2>&1 || die "Ollama/Qdrant network is unavailable"
if docker container inspect "${MIGRATION_CONTAINER}" >/dev/null 2>&1; then
    die "A migration container already exists; finish that run first"
fi

connection_flags=()
for service in audrey custom-tools media-worker media-fetcher; do
    if running="$(docker container inspect --format '{{.State.Running}}' "${service}" 2>/dev/null)"; then
        [[ "${running}" == "false" ]] || die "Stop audrey, custom-tools, media-worker and media-fetcher before migration"
        if [[ "${service}" == "audrey" || "${service}" == "custom-tools" ]]; then
            # Keep connection/collection overrides; never forward API credentials.
            values="$(docker container inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "${service}")"
            while IFS='=' read -r key value; do
                case "${key}" in
                    OLLAMA_HOST|QDRANT_HOST|QDRANT_PORT|MEMORY_COLLECTION|CHAT_ARCHIVE_COLLECTION|CHAT_ARCHIVE_RETENTION_DAYS)
                        connection_flags+=(-e "${key}=${value}") ;;
                    OLLAMA_URL)
                        [[ "${service}" != "custom-tools" ]] || connection_flags+=(-e "OLLAMA_HOST=${value}") ;;
                esac
            done <<<"${values}"
            unset values value
        fi
    fi
done
for setting in OLLAMA_HOST QDRANT_HOST QDRANT_PORT MEMORY_COLLECTION CHAT_ARCHIVE_COLLECTION CHAT_ARCHIVE_RETENTION_DAYS; do
    if [[ -n "${!setting:-}" ]]; then
        connection_flags+=(-e "${setting}=${!setting}")
    fi
done

# Embeddings run in Ollama. Local NumPy/OpenBLAS needs no parallel pool;
# on high-core hosts it can consume the container's thread budget before DNS.
docker run --rm --network none --read-only --no-healthcheck --cap-drop ALL \
    --security-opt no-new-privileges \
    -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 \
    --mount "type=bind,src=${STATE_HOST},dst=/migration" \
    --entrypoint /opt/venv/bin/python "${IMAGE}" -c \
    'import tempfile; import audrey.kb.reembed; handle=tempfile.NamedTemporaryFile(dir="/migration"); handle.close()' \
    >/dev/null 2>&1 || die "Current migration CLI and writable private migration state are required"

notify "▶ Audrey ${migration_label} started. Private backups stay on Tower; only counts and digests are reported. Services restart after verified success. Report: ${REPORT}" || true
echo "Private migration state: ${STATE_HOST}"
echo "Safe report: ${REPORT}"
echo "Log: ${LOG}"
set +e
(
    docker run --rm --name "${MIGRATION_CONTAINER}" --network "${NETWORK}" \
        --read-only --no-healthcheck --cap-drop ALL \
        --security-opt no-new-privileges --pids-limit 64 --memory 2g \
        -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 \
        --tmpfs /tmp:rw,nosuid,nodev,size=128m \
        --mount "type=bind,src=${CONFIG_FILE},dst=/app/config.yaml,readonly" \
        --mount "type=bind,src=${DATA_HOST},dst=/data,readonly" \
        --mount "type=bind,src=${TOOLS_HOST},dst=/tools-data,readonly" \
        --mount "type=bind,src=${STATE_HOST},dst=/migration" \
        "${connection_flags[@]}" \
        --entrypoint /opt/venv/bin/python "${IMAGE}" -m audrey.kb.reembed \
        --config /app/config.yaml --uploads-db /data/uploads.sqlite \
        --archive-db /tools-data/chat_archive.db --app-db /data/audrey_app.sqlite \
        --state /migration --report "/migration/reports/${REPORT_NAME}" \
        --scope "${SCOPE}" --maintenance-confirmed \
        2>&1 | tee "${LOG}"
    pipeline_codes=("${PIPESTATUS[@]}")
    code="${pipeline_codes[0]}"
    if [[ "${code}" == "0" && "${pipeline_codes[1]}" != "0" ]]; then
        code=3
    fi
    exit "${code}"
) &
migration_worker_pid=$!
wait "${migration_worker_pid}"
code=$?
migration_worker_pid=""
set -e
if [[ "${code}" != "0" ]]; then
    echo "Migration incomplete. Application writers remain stopped; rerun this command after resolving the reported failure." >&2
    exit "${code}"
fi

# Only verified success can reopen writers on the new vector space.
cd -- "${REPO_ROOT}"
if ! docker compose up -d --force-recreate --wait custom-tools audrey; then
    echo "Migration succeeded, but application startup failed. Inspect Compose logs before reopening traffic." >&2
    exit 3
fi
if ! docker compose up -d media-worker media-fetcher; then
    echo "Migration succeeded, but media worker startup failed. Inspect Compose logs." >&2
    exit 3
fi
echo "Migration verified; backend, tools and media services restarted. Private recovery data remains on Tower."
