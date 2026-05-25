# Lesson 8 — Deep mode: planner, panel, synthesizer, reflect

_(Placeholder — full lesson pending draft.)_

This lesson opens the four-stage pipeline that runs when the complexity
gate (Lesson 7) routes a request down the deep path instead of the fast
one. Each stage gets the same treatment Lessons 11 and 12 will give the
KB:

- **Planner** ([`pipeline/planner.py`](../../src/audrey/pipeline/planner.py))
  — optional sub-question decomposition before dispatch.
- **Deep panel** ([`pipeline/deep_panel.py`](../../src/audrey/pipeline/deep_panel.py))
  — parallel worker dispatch across the configured pool, with the GPU
  gate serializing local workers and Ollama Pro concurrency capping
  cloud ones. Workers can be tool-capable (running ReAct internally,
  which Lesson 9 opens).
- **Synthesizer** ([`pipeline/synthesize.py`](../../src/audrey/pipeline/synthesize.py))
  — merges worker drafts into the final answer, with a configured
  fallback if the primary synth fails and a degrade-to-longest-draft
  path if both fail.
- **Reflect** ([`pipeline/reflect.py`](../../src/audrey/pipeline/reflect.py))
  — deterministic quality gate that can trigger one retry of the
  panel+synth before shipping what we have.
