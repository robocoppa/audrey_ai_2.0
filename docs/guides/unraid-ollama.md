# Unraid — Ollama container (canonical reference)

This is the authoritative "how is Ollama configured" reference. Whenever you
recreate or edit the container, diff against this.

For the step-by-step first-time setup, see `phase-1-deploy.md`.

## Container

| | |
|---|---|
| Name | `ollama` |
| Image | `ollama/ollama:latest` |
| Network | `Custom: ollama-net` (resolvable as hostname `ollama`) |
| Port | `11434:11434/tcp` |
| Extra Parameters | `--gpus all --runtime=nvidia` |
| Device entries | **none** (adding any — even with a blank value — breaks startup) |

## Volumes

| Container path | Host path | Purpose |
|---|---|---|
| `/root/.ollama` | `/mnt/user/appdata/ollama` | All Ollama data: model weights (in `./models/`), cloud-bridge keypair (`id_ed25519*`), runtime state. **Weights are ~400 GB — never delete without backup.** |

Single bind mount. No separate `models` share.

## Environment

| Var | Value | Why |
|---|---|---|
| `OLLAMA_HOST` | `0.0.0.0:11434` | Listen on all interfaces inside the container network. |
| `OLLAMA_NUM_PARALLEL` | `1` | One request at a time per model. |
| `OLLAMA_MAX_LOADED_MODELS` | `1` | Prevents Ollama from keeping two large models in VRAM at once. Audrey's semaphore enforces the same upstream. |
| `NVIDIA_VISIBLE_DEVICES` | `all` | Both 3090 Ti GPUs visible inside the container. |
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility` | Enough for inference + nvidia-smi. |

> `OLLAMA_MODELS` is **not** set — Ollama uses the default
> `/root/.ollama/models`, which resolves to `/mnt/user/appdata/ollama/models`
> via the bind mount. Setting it to anything else would orphan your existing
> weights.

## How Audrey reaches it

From inside `ollama-net` (Audrey, custom-tools, open-webui): `http://ollama:11434`.
From the Unraid host or LAN: `http://<unraid-ip>:11434`.

## Common ops

```bash
# Start / stop
docker start ollama
docker stop ollama

# List models
curl -s http://localhost:11434/api/tags | jq '.models[].name' | sort

# Pull a model manually
docker exec ollama ollama pull qwen3.6:35b

# See which model is currently resident in VRAM
docker exec ollama ollama ps

# Watch VRAM / GPU load in real time
watch -n 1 docker exec ollama nvidia-smi

# Tail logs
docker logs -f ollama
```

## Known quirks

- **Ollama Pro cloud models** count against your subscription's concurrent-run
  limit (3). Audrey's `MAX_DEEP_WORKERS_CLOUD=3` respects this.
- **First load of a large local model** (e.g. `llama4:latest` at 67 GB) takes
  longer than subsequent loads because Ollama is mapping it into GPU memory.
  Subsequent calls inside the same session are fast.
- **Switching between two large local models** forces an eviction. With
  `OLLAMA_MAX_LOADED_MODELS=1` Ollama guarantees only one model at a time, so
  a deep-panel that uses two different local workers will be sequential.
  That's intentional given 48 GB VRAM and PSU headroom.
