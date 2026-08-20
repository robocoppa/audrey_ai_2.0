#!/usr/bin/env bash
# Pulls every Ollama model Audrey depends on.
# Run on Unraid after Phase 1 Ollama recreation.
# Cloud models don't download weights — `ollama pull` just registers them.

set -euo pipefail

OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"

# ⚠️ RECONCILED AGAINST A REAL `ollama list` ON 2026-08-16. Before that this
# script and `config.yaml` were two independent authorities and had silently
# diverged in BOTH directions: names here that were never on the box
# (`nemotron-cascade-2`, `gemma4:31b`, `glm-5.1:cloud`), a name here that config
# never used while config dispatched a different one (`minimax-m2.7` vs
# `minimax-m3`), and a model on the box that this script never listed
# (`nemotron-3.5-lightning`, which reached config only through
# `passthrough.allowed_models`, so a rebuilt box would have lacked it).
# ▶ `scripts/check_model_inventory.py` compares CONFIG to the box; it cannot see
#   this file. Reconcile the two by hand whenever either changes.
LOCAL_MODELS=(
  # THE ROUTER — deliberately tiny, on the hot path of every non-skipped
  # request, and NOT GPU-gated, so a big model here evicts the deep worker
  # rather than queueing behind it. Replaced qwen3:4b on 2026-08-16 after
  # probing; see the notes at `router:` in config.yaml.
  "qwen3.5:4b"
  # 2026-08-15: replaced qwen3.6:35b, qwen3-coder-next:latest and
  # qwen2.5-coder:32b across every text role (code, reasoning, general).
  "qwen3.8:latest"
  # The two local panel workers. nemotron is the sole local draft on
  # deep_panel.code and the second worker on deep_panel_local.code; muse-glimmer
  # is the second worker on deep_panel_local.reasoning + .general. Both have
  # been on the box since the local bake-off and neither was listed here.
  "nemotron-3.5-lightning:latest"
  "muse-glimmer:latest"
  "llama4:latest"
  "glm-4.7-flash:q8_0"
  "qwen3-vl:32b"
  "llava:34b"
  "nomic-embed-text:latest"
  # ── 2026-08-18 bake-off candidates, NOT production roles ──
  # These hold no `model_registry` slot and sit in no pool; they reach config
  # through `passthrough.allowed_models` alone, so they are targetable by
  # `eval_research.py --models` and nothing else. They are listed here because
  # `test_every_model_the_config_names_is_pulled_by_the_script` is right that a
  # rebuilt box must not come up missing a name config mentions — that is the
  # exact trap this file's header records. ▶ If a candidate loses its bake-off,
  # delete it from BOTH files rather than leaving a 96 GB download in a rebuild.
  # ⚠️ `laguna-s-2.1` is 96 GB against 48 GB of VRAM — it cannot be resident,
  # and pulling it costs an hour and a fifth of the array.
  "laguna-s-2.1:latest"
  "laguna-xs-2.1:latest"
  "ornith-1.5:35b"
)

CLOUD_MODELS=(
  "deepseek-v4-pro:cloud"
  "kimi-k2.6:cloud"
  "kimi-k2.7-code:cloud"
  "qwen3.5:397b-cloud"
  "deepseek-v3.2:cloud"
  "deepseek-v4-flash:cloud"
  "nemotron-3-super:cloud"
  "glm-5.2:cloud"
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
