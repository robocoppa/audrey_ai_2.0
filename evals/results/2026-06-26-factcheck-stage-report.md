# audrey_research evaluation — fact-check stage (Phase 25)

Second protocol run, against the deployed Phase 25 build with the fact-check
stage **confirmed live**. Paired answers: `2026-06-26-factcheck-stage-answers.md`.
Compared against the pre-fact-check baseline (`2026-06-26-accuracy-stress-*`).

**Banner confirmed:** all 10 cases now stream the 5-stage sequence
`Planning → Researching → Verifying → Fact-checking → Writing`. Structural
checks 10/10 PASS. (Note: an earlier run mis-reported "no banner" — that was a
gap in the eval script's banner list, since fixed; the stage was streaming all
along.)

**Reading this:** I remain a fallible accuracy judge. Verdicts below are
directional, one sample each.

## Headline: the fact-check stage works — but its effect is "caution," not "correction"

The stage measurably changes behavior on the current-facts cases, in the
direction of **more hedging and more omission** rather than fixing-the-date. On
one case that's a clean win; on another it overcorrects into incompleteness. The
biographies and controls are unchanged (no regression, no bloat).

## The two cases that motivated this (current facts)

### current-rust-async — CLEAN WIN ✅
The exact claim flagged in the baseline:
- **Before:** "async-std was officially deprecated in 2024." (overconfident, the
  visible push was 2025)
- **After:** "async-std's status is currently disputed. While some sources point
  to a 2025 discontinuation, no official announcement has been confirmed, and
  other reports indicate it has been largely unmaintained since around 2021."

This is exactly the intended behavior: an overconfident dated claim got checked,
softened, and multi-sourced. Best-case outcome for the stage.

### current-2025-recent — OVERCORRECTION ⚠️
- **Before:** listed ~6 releases (DeepSeek-R1, Gemma 3, Qwen3, GLM-4.5, Llama 4,
  OLMo) with specific dates — some wrong (DeepSeek "Jan 26" vs official Jan 20;
  Qwen3 "Apr 28" vs Apr 29).
- **After:** commits to only **2** releases (Llama 4 April, Mistral 3 Dec 2) and
  explicitly drops everything else as "unverified due to gaps in the captured
  evidence" — including DeepSeek, Gemma, Qwen, which are real.

The stage did **not correct** the DeepSeek date — it **omitted DeepSeek entirely**
rather than verify it. So the answer is now *more accurate per-claim* (nothing
confidently wrong) but *substantially less complete* (dropped real, well-known
releases). Whether this is "better" is a genuine judgment call: it traded
coverage for safety, harder than ideal. Notably it also introduced a new
specific claim — "Mistral Large 3, 675-billion-parameter MoE, Dec 2" — which is
itself a precise dated claim that would need checking (the fact-checker
apparently trusted it).

## No regressions elsewhere

- **Biographies (Euclid/Pythagoras/Archimedes):** unchanged quality; +30 verifier
  discipline intact ("attributed to", "disputed or spurious authorship", "now
  attributed to Cleonides"). The fact-check stage didn't disturb the
  already-good ancient-history behavior (few current/checkable claims to act on).
- **Controls (toast, recursion):** clean — the stage ran through them (banner
  present) but added no spurious verification caveats or sources. No bloat in the
  prose; latency cost only.

## Cost

The stage adds a full web-using ReAct loop per request. This run's window was
~35 min for 10 cases vs. ~30 min for the baseline — a **~15–20% wall-clock
increase**, consistent with one extra cloud tool-stage. Acceptable for an opt-in
"thorough" mode, but real and worth noting if latency ever matters.

## Verdict

**Keep the stage — it does what it was built to do** (catch and soften
overconfident current claims, per the Rust async win), with no regression to the
biographies/controls. **But tune its disposition:** on `current-2025-recent` it
is too eager to *drop* claims it can't instantly verify rather than *hedge* them.
The ideal behavior is "soften to a hedge," not "omit." Candidate adjustments
(prompt-only, eval each against this run):

1. `FACTCHECK_SYSTEM` / `WRITER_SYSTEM`: prefer **hedging** an unverified-but-
   plausible claim ("reportedly released in late January") over **dropping** it —
   only drop when actively contradicted.
2. Watch that the fact-checker doesn't *introduce* new precise claims it didn't
   verify (the Mistral-3 specifics).

This is a tuning question, not a "does it work" question — the stage works. Next
change should be measured by re-running this protocol and diffing.

## Caveats
- One run, one sample per case; non-deterministic system. The
  caution-vs-completeness shift is directionally clear but not quantified.
- I can't independently verify the 2025 dates myself — the per-claim accuracy
  read leans on internal consistency + hedging quality. User is the authority,
  especially on whether dropping DeepSeek/Qwen is acceptable.
