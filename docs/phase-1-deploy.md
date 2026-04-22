# Phase 1 — Ollama clean recreation (Unraid)

**Goal:** stop the current Ollama container, recreate it cleanly on
`ollama-net` with the right env vars and both GPUs, **without touching model
weights or Ollama Pro cloud auth**. Verify via `/api/tags`.

**Prereqs confirmed:**
- `ollama-net` Docker network already exists.
- Both RTX 3090 Ti GPUs visible under **Tools → System Devices** on Unraid.
- Nvidia driver + runtime installed (Nvidia-Driver plugin).
- Existing data layout: everything under `/mnt/user/appdata/ollama/`, including
  model weights at `/mnt/user/appdata/ollama/models/`.

---

## Step 1 — Note what you have

Open the Unraid web UI → **Docker** tab. Find the current `ollama` container.

Click its icon → **Edit**. In the form you'll see:
- The current `Repository` (image).
- The current `Path`, `Port`, `Variable`, `Device` mappings.

Scroll to the bottom, click **Show more settings…**, and take a screenshot (or
copy to a notes file) in case you want to diff later. Nothing destructive yet.

---

## Step 2 — Stop and remove the current container

Docker tab → `ollama` row → click the icon → **Stop**. Wait until the status
turns grey.

Click the icon again → **Remove**. Confirm.

> This deletes the *container*, not the *image* and not the bind-mounted data
> on disk. It's a clean slate for the container config only.

---

## Step 3 — Verify your existing data (do not delete anything)

Your model weights and Ollama Pro cloud-bridge keypair both live under
`/mnt/user/appdata/ollama`. The container recreation **reuses** that directory.
Nothing gets deleted.

From the Unraid web terminal (top-right icon → **>_ Terminal**), confirm:

```bash
ls /mnt/user/appdata/ollama/models | head   # should show model dirs (manifests, blobs)
ls /mnt/user/appdata/ollama/id_ed25519*     # cloud-bridge auth keypair should be present
```

If the first command is empty, **stop and tell me** — something's off and we
shouldn't proceed without understanding why. If the second is empty, that's
fine (you may not have used cloud models yet, or the file lives inside `models/`
on older Ollama versions).

---

## Step 4 — Recreate the Ollama container

Docker tab → **Add Container** (bottom-right button).

Fill in **exactly** these fields. Leave everything else at default unless
specified.

| Field | Value |
|---|---|
| **Name** | `ollama` |
| **Repository** | `ollama/ollama:latest` |
| **Network Type** | `Custom: ollama-net` |
| **Console shell command** | `bash` |
| **Privileged** | off |
| **Icon URL** | (optional, leave blank) |

### Port mapping
Add one port (**Add another Path, Port, Variable, Label or Device** → **Port**):
- Name: `API`
- Container Port: `11434`
- Host Port: `11434`
- Connection Type: `TCP`

### Path mapping (one only)
- Name: `ollama-data`
- Container Path: `/root/.ollama`
- Host Path: `/mnt/user/appdata/ollama`
- Access Mode: `Read/Write`

This single mount covers both model weights (in
`/mnt/user/appdata/ollama/models/`, Ollama's default location) and the
cloud-bridge keypair + runtime state.

### Environment variables (add three)
| Name | Value |
|---|---|
| `OLLAMA_HOST` | `0.0.0.0:11434` |
| `OLLAMA_NUM_PARALLEL` | `1` |
| `OLLAMA_MAX_LOADED_MODELS` | `1` |

> No `OLLAMA_MODELS` variable needed — Ollama defaults to
> `/root/.ollama/models` which resolves to your existing weights via the bind
> mount above.
>
> Why `NUM_PARALLEL=1` and `MAX_LOADED_MODELS=1`: Audrey's semaphore already
> serializes local model calls (`GPU_CONCURRENCY=1`). Letting Ollama try to
> parallel-load a second model on top would thrash VRAM against the 48 GB cap.

### GPU / device passthrough
Click **Add another Path, Port, Variable, Label or Device** → **Device**:
- Name: `nvidia-runtime`
- Value: leave blank (handled via extra args below)

Scroll to **Extra Parameters** (in "Show more settings…") and set:
```
--gpus all --runtime=nvidia
```

Scroll to **Nvidia driver** / **NVIDIA_VISIBLE_DEVICES** fields if the
Unraid Nvidia plugin added them:
- `NVIDIA_VISIBLE_DEVICES` = `all`
- `NVIDIA_DRIVER_CAPABILITIES` = `compute,utility`

(If those fields don't appear, the `--gpus all` in Extra Parameters covers it.)

### Apply
Click **Apply** at the bottom. Unraid will pull the image if needed, then start
the container. Watch the Docker tab until `ollama` shows **started** (green).

---

## Step 5 — Verify the container is healthy

From the Unraid web terminal:

```bash
# Network: ollama must be attached to ollama-net
docker inspect ollama --format '{{json .NetworkSettings.Networks}}' | jq

# API responds
curl -s http://localhost:11434/api/tags | jq '.models | length'

# GPUs visible inside the container
docker exec ollama nvidia-smi | head -20
```

Expected:
- The first command prints a JSON object whose key is `ollama-net`.
- The second prints a number ≥ 1 (your existing models are discovered from
  `/mnt/user/models`).
- The third prints the nvidia-smi table showing **both** 3090 Ti cards.

If any of those fail, **stop and paste the output** — don't proceed.

---

## Step 6 — Verify models (your pulls already done)

You've already pulled everything via the Ollama CLI. Just confirm the new
container's `/api/tags` sees the full list:

```bash
curl -s http://localhost:11434/api/tags | jq -r '.models[].name' | sort
```

Expected names (see `CONTINUITY.md` for the full reference): `qwen3.6:35b`,
`kimi-k2.6:cloud`, `qwen3-coder-next:latest`, `qwen2.5-coder:32b`,
`devstral-small-2:latest`, `llama4:latest`, `nemotron-cascade-2:latest`,
`olmo-3.1:32b`, `glm-4.7-flash:q8_0`, `deepseek-r1:32b`, `qwen3-vl:32b`,
`llava:34b`, `gemma4:31b`, `qwen3:4b`, `qwen2.5:1.5b`,
`nomic-embed-text:latest`, `qwen3.5:35b-a3b` (+ `qwen3.5:35b`),
`qwen3.5:397b-cloud`, `deepseek-v3.2:cloud`, `cogito-2.1:671b-cloud`,
`nemotron-3-super:cloud`, `minimax-m2.7:cloud`, `kimi-k2.5:cloud`,
`glm-5.1:cloud`, `glm-4.7-flash:latest`.

If anything is missing after the container recreation, re-register with the
bundled script — it's a no-op on already-present models:

```bash
cd /mnt/user/projects/audrey_ai_2.0
./scripts/pull-models.sh
```

---

## Step 7 — Smoke test

```bash
# Router model responds with a real generation
curl -s http://localhost:11434/api/generate -d '{
  "model": "qwen3:4b",
  "prompt": "say hi in 3 words",
  "stream": false
}' | jq -r '.response'

# A cloud model responds too (uses your Ollama Pro bridge)
curl -s http://localhost:11434/api/generate -d '{
  "model": "kimi-k2.6:cloud",
  "prompt": "say hi in 3 words",
  "stream": false
}' | jq -r '.response'
```

Both should return a short text response within a few seconds. Cloud will be
faster than local because no VRAM load is involved.

---

## Phase 1 success criteria

- [ ] `ollama` container restarted cleanly with the config from Step 4
- [ ] Attached to `ollama-net`
- [ ] Both GPUs visible to the container
- [ ] `/api/tags` lists all the models from `CONTINUITY.md`
- [ ] One local + one cloud generation smoke test pass

Paste the outputs from Step 5 and Step 7 back and I'll update
`CONTINUITY.md` to mark Phase 1 verified, then we move to Phase 2 (custom-tools
v1).

---

## Rollback (if anything goes wrong)

Everything is reversible — no filesystem changes at any step:

1. Stop + remove the new `ollama` container.
2. Create a container with whatever your old config was (the screenshot from
   Step 1).

No model weights and no cloud-auth keys are ever touched in this procedure.
