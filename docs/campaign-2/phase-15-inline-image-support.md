# Campaign 2 Phase 15 — Inline image support on the chat path

Lets a user attach an image in Open WebUI to `audrey_auto` (or any pipeline
virtual model) and get a real answer about it. Before this phase, an attached
image **failed with HTTP 422** before any pipeline code ran.

## What it does

Open WebUI sends an attached image as the OpenAI multimodal `content` shape — a
list of typed parts instead of a plain string:

```json
"content": [
  {"type": "text", "text": "describe this image"},
  {"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}
]
```

The request schema only accepted `content: str`, so Pydantic rejected the list
at the door (`422 Unprocessable Entity`, `loc: messages[N].content, "Input
should be a valid string"`). After this phase:

  - The schema accepts `str | list[dict]`.
  - A turn carrying an `image_url` part is detected and **forced onto the fast
    path on the `vl:` (vision) pool** — `qwen3-vl:32b` (local primary), `llava:34b`
    (local fallback). The vision model receives the image verbatim and answers.

## Why this exists

The vision machinery was already mostly wired: a `vl:` pool, a `vl` task type,
and image-keyword classification all existed — but the request never survived
schema validation to reach them. And even past the schema, two gaps had to be
closed:

  - **Classification misses neutral wording.** "describe this" with an attached
    image won't trip the text-keyword classifier. So an actual image *part* now
    forces `task="vl"` regardless of the words.
  - **The deep path drops images.** Planner/panel/synthesize rebuild prompts
    text-only, so a deep request would silently lose the picture. Image turns
    therefore skip deep entirely and run the fast path, where the original
    messages (image parts intact) are forwarded straight to the vision model.

Multi-draft deep synthesis isn't suited to vision anyway, so "image ⇒ fast+vl"
is both the correct and the simplest behavior.

## What's in scope

  - **[`src/audrey/routes/openai.py`](../../src/audrey/routes/openai.py)** —
    `ChatMessage.content` loosened to `str | list[dict[str, Any]]`; the
    streaming complexity gate gains a highest-priority `image_turn` branch
    (`task="vl"`, force fast) above the existing `owui_task`/forced-model
    ladder; log line surfaces `image=1`.
  - **[`src/audrey/pipeline/graph.py`](../../src/audrey/pipeline/graph.py)** —
    `node_complexity` (the non-streaming twin) gains the same image branch,
    overriding `mode="fast"` and `task_type="vl"`.
  - **[`src/audrey/pipeline/messages.py`](../../src/audrey/pipeline/messages.py)** —
    new `has_image_part(messages)` helper (reverse-walk to the latest user
    turn; True if its `content` list contains an `image_url` part).
  - **[`config.yaml`](../../config.yaml)** — `vl:` pool reordered to
    **local-only**: `qwen3-vl:32b` (priority 100) primary, `llava:34b` (90)
    fallback. `nemotron3:33b` (a text/reasoning model) removed from `vl:` — it
    keeps its reasoning + deep-panel-worker roles. The old pool led with a
    non-vision model and listed unverified cloud entries; an image routed there
    would have been answered blind.

## What's not in scope

  - **Deep-path vision.** Image turns never go deep; the deep path is untouched.
  - **Audrey's own image routes.** KB image-search (`routes/kb.py`,
    `image_url`/`image_b64`, CLIP) and the `/upload` UI are unchanged — those
    are a different entry path and already worked.
  - **Cloud vision models.** The `vl:` pool is local-only on purpose. To add a
    cloud vl model later, first confirm it actually accepts images.

## Prerequisite — a local vision model must be pulled

The `vl:` pool names `qwen3-vl:32b` and `llava:34b`. Listing in `config.yaml`
is not the same as being pulled on the box. Check, and pull if missing
(`qwen3-vl:32b` is the primary — having just that one is enough):

```
docker exec ollama ollama list
docker exec ollama ollama pull qwen3-vl:32b
```

If no `vl:` model is healthy, `first_healthy("vl", …)` raises and image turns
fail with a clear error rather than 422. (As of this phase both `qwen3-vl:32b`
and `llava:34b` are already pulled — verify before deploying.)

## Deploy on Unraid

From `/mnt/user/appdata/audrey_ai_2.0` (no custom-tools change this phase):

```
docker compose up -d --build audrey-ai
docker compose logs -f audrey-ai
```

## Verification

Hermetic (laptop, already green): `460 pytests pass`, ruff clean on touched
files. New coverage: schema accepts list content, `has_image_part` across all
shapes, `vl:` pool resolves to `qwen3-vl:32b` (falls back to `llava:34b`).

Live, on the box:

  1. In OWUI, attach an image to **audrey_auto** and ask "describe this image."
     Expect a real description back — no 422, no "I can't see images."
  2. Confirm the route picked vision + fast:

     ```
     docker logs audrey-ai 2>&1 | grep -E "task=vl .* mode=fast image=1"
     ```

  3. Regression: a **text-only** prompt to audrey_auto still classifies and
     gates exactly as before (a long paste still goes deep; a short question
     still goes fast). The `image=1` marker must be absent for text turns.

## What this unblocks

Inline vision on the public chat surface. Closes the "Inline vision on the chat
path" followup. Audrey can now answer about an image a user pastes into the
chat box, using a local vision model under the same fair-scheduling layers as
all other pipeline traffic.
