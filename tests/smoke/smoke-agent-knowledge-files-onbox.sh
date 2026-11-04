#!/usr/bin/env bash
# Exercise installed file-backed approved Project Knowledge using disposable canonical data.
set -euo pipefail
IFS=$'\n\t'

die() {
  echo "SETUP ERROR: $*" >&2
  exit 2
}

main() {
  local appdata smoke_script container running
  appdata="${AUDREY_SMOKE_APPDATA:-${APPDATA:-/mnt/user/appdata/audrey_ai_2.0}}"
  smoke_script="${appdata}/tests/smoke/smoke_agent_knowledge_files.py"
  container="${AUDREY_SMOKE_CONTAINER:-audrey}"
  [[ -r "${smoke_script}" && -f "${smoke_script}" ]] || die "smoke script missing or unreadable: ${smoke_script}"
  command -v docker >/dev/null || die "docker not found; run this on Tower"
  running="$(docker inspect --type container --format '{{.State.Running}}' -- "${container}" 2>/dev/null)" \
    || die "cannot inspect container ${container}"
  [[ "${running}" == true ]] || die "container ${container} is not running"
  exec docker exec -i -e PYTHONDONTWRITEBYTECODE=1 -- "${container}" \
    /opt/venv/bin/python - "$@" < "${smoke_script}"
}

main "$@"
