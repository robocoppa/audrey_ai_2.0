#!/usr/bin/env bash
# Pulls every Ollama model Audrey depends on.
# Run on Unraid after Phase 1 Ollama recreation.
# Cloud models don't download weights — `ollama pull` just registers them.

set -euo pipefail

OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"

LOCAL_MODELS=(
  "qwen3:4b"
  "qwen3.6:35b"
  "qwen3.5:35b-a3b"
  "qwen3-coder-next:latest"
  "qwen2.5-coder:32b"
  "devstral-small-2:latest"
  "llama4:latest"
  "nemotron-cascade-2:latest"
  "olmo-3.1:32b"
  "glm-4.7-flash:q8_0"
  "deepseek-r1:32b"
  "qwen3-vl:32b"
  "llava:34b"
  "gemma4:31b"
  "nomic-embed-text:latest"
)

CLOUD_MODELS=(
  "kimi-k2.6:cloud"
  "qwen3.5:397b-cloud"
  "deepseek-v3.2:cloud"
  "deepseek-v4-flash:cloud"
  "cogito-2.1:671b-cloud"
  "nemotron-3-super:cloud"
  "minimax-m2.7:cloud"
  "glm-5.1:cloud"
)

pull_model() {
  local m="$1"
  echo "  → $m"
  # Stream progress; print final status or any error line.
  curl -s -N -X POST "$OLLAMA_HOST/api/pull" -d "{\"model\":\"$m\"}" \
    | awk '/"status":"success"/{ok=1} /"error"/{print; err=1} END{ exit err?1:(ok?0:2) }' \
    || { echo "  ✗ pull failed for $m"; exit 1; }
}

echo "Pulling local models (large ones take a while; be patient on llama4)…"
for m in "${LOCAL_MODELS[@]}"; do pull_model "$m"; done

echo "Registering cloud models…"
for m in "${CLOUD_MODELS[@]}"; do pull_model "$m"; done

echo
echo "Done. Verify with:"
echo "  curl -s $OLLAMA_HOST/api/tags | jq -r '.models[].name' | sort"
