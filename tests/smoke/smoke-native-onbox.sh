#!/usr/bin/env bash
# Run a native smoke through the standalone UI proxy from Docker-only Tower.

set -euo pipefail
IFS=$'\n\t'

APPDATA="${APPDATA:-/mnt/user/appdata/audrey_ai_2.0}"
ENV_FILE="${ENV_FILE:-${APPDATA}/.env.smoke.local}"
IMAGE="${IMAGE:-audrey:latest}"
NETWORK="${NETWORK:-ollama-net}"
BASE_URL="${AUDREY_SMOKE_BASE_URL:-http://audrey-ui:8080}"
SMOKE_SCRIPT="${1:-smoke_native_ui.py}"
SMOKE_ARGS=("${@:2}")

die() {
  echo "ERROR: $*" >&2
  exit 2
}

case "${SMOKE_SCRIPT}" in
  smoke_native_auth_cutover.py|smoke_native_ui.py|smoke_native_files.py|\
  smoke_native_chat_projection.py|smoke_native_access_models.py|\
  smoke_native_preferences.py|smoke_native_tool_events.py|smoke_native_modes.py|\
  smoke_native_restart_persistence.py|smoke_native_document_approvals.py)
    ;;
  *)
    die "unsupported native smoke script: ${SMOKE_SCRIPT}"
    ;;
esac

[[ -f "${ENV_FILE}" ]] || die "smoke env file missing: ${ENV_FILE}"
[[ -f "${APPDATA}/tests/smoke/${SMOKE_SCRIPT}" ]] \
  || die "smoke script missing: ${APPDATA}/tests/smoke/${SMOKE_SCRIPT}"

have_user=0
have_admin=0
line_number=0
while IFS= read -r line || [[ -n "${line}" ]]; do
  line_number=$((line_number + 1))
  [[ -z "${line}" || "${line}" =~ ^[[:space:]]*# ]] && continue
  [[ "${line}" != export[[:space:]]* ]] \
    || die "${ENV_FILE}:${line_number} starts with 'export'; Docker --env-file requires KEY=value"
  [[ "${line}" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] \
    || die "${ENV_FILE}:${line_number} must use KEY=value with no spaces around '='"
  key="${line%%=*}"
  value="${line#*=}"
  [[ -n "${value}" ]] || die "${ENV_FILE}:${line_number} has an empty ${key} value"
  case "${key}" in
    AUDREY_USER_JWT) have_user=1 ;;
    AUDREY_ADMIN_JWT) have_admin=1 ;;
    *)
      die "${ENV_FILE}:${line_number} contains unsupported key ${key}; keep only native AUDREY_USER_JWT and AUDREY_ADMIN_JWT entries"
      ;;
  esac
done < "${ENV_FILE}"

(( have_user == 1 )) \
  || die "${ENV_FILE} is missing AUDREY_USER_JWT"
if [[ "${SMOKE_SCRIPT}" != "smoke_native_auth_cutover.py" ]]; then
  (( have_admin == 1 )) \
    || die "${ENV_FILE} is missing AUDREY_ADMIN_JWT"
fi

OPTIONAL_ENV_NAMES=(
  AUDREY_DIRECT_SMOKE_MODEL_ID
  AUDREY_DIRECT_SMOKE_TIMEOUT_SECONDS
  AUDREY_FILE_SMOKE_TIMEOUT_SECONDS
  AUDREY_MODE_SMOKE_TIMEOUT_SECONDS
  AUDREY_PREFERENCES_SMOKE_TIMEOUT_SECONDS
  AUDREY_RESTART_SMOKE_TIMEOUT_SECONDS
  AUDREY_RESTART_HEALTH_URL
)
DOCKER_ENV_ARGS=()
DOCKER_USER_ARGS=()
DOCKER_VOLUME_ARGS=()
if [[ "${SMOKE_SCRIPT}" == "smoke_native_restart_persistence.py" ]]; then
  PERSISTENCE_STATE_DIR="${AUDREY_PERSISTENCE_STATE_DIR:-${APPDATA}/testing-out/smokes}"
  mkdir -p -- "${PERSISTENCE_STATE_DIR}"
  # Tower normally runs this wrapper as root, while the Audrey image uses
  # UID 99. Root is confined to this disposable probe and its one state mount.
  DOCKER_USER_ARGS+=(--user 0:0)
  DOCKER_ENV_ARGS+=(--env "AUDREY_PERSISTENCE_SNAPSHOT_PATH=/state/c3-restart-persistence.json")
  DOCKER_VOLUME_ARGS+=(--volume "${PERSISTENCE_STATE_DIR}:/state")
fi
if [[ "${SMOKE_SCRIPT}" == "smoke_native_document_approvals.py" ]]; then
  AUDREY_RUNTIME_DIR="${AUDREY_RUNTIME_DIR:-/mnt/user/appdata/runtime}"
  [[ -d "${AUDREY_RUNTIME_DIR}" ]] \
    || die "Audrey runtime directory missing: ${AUDREY_RUNTIME_DIR}"
  DOCKER_VOLUME_ARGS+=(--volume "${AUDREY_RUNTIME_DIR}:/data")
fi
for env_name in "${OPTIONAL_ENV_NAMES[@]}"; do
  env_value="${!env_name-}"
  [[ -z "${env_value}" ]] || DOCKER_ENV_ARGS+=(--env "${env_name}=${env_value}")
done

command -v docker >/dev/null || die "docker not found; run this on Tower"
docker image inspect "${IMAGE}" >/dev/null 2>&1 \
  || die "image ${IMAGE} is not available"
docker network inspect "${NETWORK}" >/dev/null 2>&1 \
  || die "Docker network ${NETWORK} is not available"

exec docker run --rm \
  --network "${NETWORK}" \
  --env-file "${ENV_FILE}" \
  --env "AUDREY_SMOKE_BASE_URL=${BASE_URL}" \
  "${DOCKER_ENV_ARGS[@]}" \
  "${DOCKER_USER_ARGS[@]}" \
  "${DOCKER_VOLUME_ARGS[@]}" \
  --volume "${APPDATA}/tests/smoke:/smoke:ro" \
  "${IMAGE}" \
  /opt/venv/bin/python "/smoke/${SMOKE_SCRIPT}" "${SMOKE_ARGS[@]}"
