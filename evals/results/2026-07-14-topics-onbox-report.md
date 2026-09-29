# eval report — 2026-07-14-topics (audrey_deep, on-box)

Paired with `2026-07-14-topics-onbox-answers.md`. Protocol: `topics` on `audrey_deep`
(deep panel + synth). Harness: 13/13 cases PASS all applicable structural checks.
This report is the **human read** the harness cannot do — factual correctness of
the reasoning/GK answers and constraint-compliance of the writing cases.

## Verdict

**13/13 sound.** Every reasoning and general-knowledge answer is not merely
anchor-present but factually correct; the three science explainers are accurate;
both constrained-writing cases meet their stated constraints. No regressions, no
hallucinations, no hedging failures. Nothing here argues for a lineup change.

## Objective cases — verified beyond the anchor

The harness only checks that the `answer_contains` string appears. I verified the
surrounding reasoning is actually correct (an anchor can be present in a wrong
derivation):

| case | anchor | correct? | note |
|---|---|:-:|---|
| reasoning-pen-notebook | 1.25, 7.25 | ✅ | pen $1.25 / notebook $7.25; sum 8.50, diff 6.00 both verified in-answer |
| reasoning-markup-discount | 82.8 | ✅ | 80→92→82.80; also flagged the trap (net +3.5%, not back to $80) |
| reasoning-calendar | sunday | ✅ | 30 mod 7 = 2, Fri→Sun; two independent derivations |
| reasoning-race-order | (none) | ✅ | Cal, Ada, Ben; all three clues satisfied and checked |
| gk-second-highest-mountain | k2 | ✅ | K2, 8,611 m; named Everest as #1 for contrast (used web_search) |
| gk-element-w | tungsten, wolfram | ✅ | W from Wolfram/wolframite; etymology correct |
| gk-berlin-wall | 1989 | ✅ | Nov 9 1989; Schabowski's "sofort, unverzüglich" trigger correct |

The GK cases were framed to invite hedging ("if you're not certain, say so").
None hedged — because all three are things the panel *is* certain of and got
right. That's the correct behavior (hedge-when-unsure, not hedge-always).

## Science explainers — human read (no anchors)

- **science-attention** — Correct throughout: Q/K/V projections, scaled
  dot-product `softmax(QKᵀ/√dₖ)V`, row-wise softmax, √dₖ flagged as a numerical
  (not conceptual) detail, multi-head, decoder −∞ masking. Library/catalog
  analogy is apt; the "it → cat/mat" worked example is concrete and correct.
- **science-plate-tectonics** — Correct mechanism (slab pull dominant, ridge
  push, mantle drag as often net-resistance) and a complete evidence stack
  (magnetic striping, continental fit + Glossopteris/Mesosaurus, Wadati-Benioff,
  seafloor age ≤200 Ma, Hawaiian-Emperor ~47 Ma bend, GPS/VLBI). Ran a real
  kb_search+web_search round.
- **science-mrna-vaccines** — Correct: LNP delivery, cytoplasmic translation,
  never enters nucleus / no DNA interaction, B/CD4/CD8 response, memory cells;
  clean comparison table vs live-attenuated and inactivated platforms.

## Constrained-writing cases — the constraints the harness can't see

These are the only cases where "PASS" from the harness could hide a violation,
so they got explicit checks:

- **writing-cold-email** — "120 words maximum, no 'passionate'/'synergy', one
  specific ask." Synth body = 76 words (88 with subject/greeting/sign-off), well
  under cap. No banned clichés. Ends with one concrete ask (10-min call Tue/Wed).
  **Compliant.**
- **writing-haiku-server-room** — "5-7-5 strictly." Five of six non-trivial lines
  land 5-7-5 cleanly. The one soft spot is Haiku 2 line 1, "Blue LEDs glow": it
  hits 5 only if "LEDs" is read as an initialism (el-ee-dee-z = 4) + "Blue" = 5.
  Defensible, but it leans on acronym pronunciation to make count — a strict-form
  purist could dock it. Everything else is on the nose.
- **writing-eli5-rewrite** — Preserves all four load-bearing ideas (sunlight as
  energy, water split → O₂, CO₂ → sugar, Calvin cycle) at a genuine 10-year-old
  reading level. Good.

## Panel behavior notes (from the debug drafts)

- **qwen3.6:35b latency** remains the standout cost: it is consistently the
  slowest worker in every case (20–36s vs 3–13s for the cloud pair), and on
  science-attention it was 35.7s against deepseek 12.8s / kimi 83.7s. It rarely
  changes the synth's substance — the synthesizer leans on the cloud drafts. This
  reinforces the standing flag that qwen3.6:35b is a latency liability as a
  general/reasoning worker; worth a dedicated sweep to decide whether to swap or
  drop it. (Not actioned here — this run was a quality check, not a lineup test.)
- No worker produced a wrong answer that the synth had to correct; drafts were
  mutually consistent, so the panel added redundancy more than error-correction
  on this (relatively easy) topic tier.

## Bottom line

The topics suite is behaving as a clean quality gate: structural checks green,
and the human read confirms substance. The only actionable thread is the
qwen3.6:35b latency question, which is a separate lineup investigation, not a
correctness problem with this run.
