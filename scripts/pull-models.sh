#!/usr/bin/env bash
# Pulls every Ollama model Audrey depends on.
# Run on Unraid after Phase 1 Ollama recreation.
# Cloud models don't download weights — `ollama pull` just registers them.

set -euo pipefail

OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"

LOCAL_MODELS=(
  # THE ROUTER — deliberately tiny, on the hot path of every non-skipped
  # request, and NOT GPU-gated, so a big model here evicts the deep worker
  # rather than queueing behind it. Replaced qwen3:4b on 2026-08-16 after
  # probing; see the notes at `router:` in config.yaml.
  "qwen3.5:4b"
  # 2026-08-15: replaced qwen3.6:35b, qwen3-coder-next:latest and
  # qwen2.5-coder:32b across every text role (code, reasoning, general).
  "qwen3.8:latest"
  # Second local workers for the audrey_local panels (2026-08-16). devstral is
  # the code pool's, qwen3.5:35b-a3b serves reasoning + general. Both replaced
  # names that sat in config for weeks without ever being pulled.
  "qwen3.5:35b-a3b"
  "devstral-small-2:latest"
  "llama4:latest"
  "nemotron-cascade-2:latest"
  "glm-4.7-flash:q8_0"
  "qwen3-vl:32b"
  "llava:34b"
  "gemma4:31b"
  "nomic-embed-text:latest"
)

CLOUD_MODELS=(
  "deepseek-v4-pro:cloud"
  "kimi-k2.6:cloud"
  "qwen3.5:397b-cloud"
  "deepseek-v3.2:cloud"
  "deepseek-v4-flash:cloud"
  "nemotron-3-super:cloud"
  # ⚠️ config names `minimax-m3:cloud`; this list said `minimax-m2.7:cloud`
  # until 2026-08-16, so the model config actually dispatches was never
  # registered by this script. A cloud pull only registers a name — cheap.
  "minimax-m3:cloud"
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
