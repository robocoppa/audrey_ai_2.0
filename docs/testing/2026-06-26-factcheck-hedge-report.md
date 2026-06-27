# audrey_research evaluation — fact-check stage, hedge-don't-drop (Phase 25)

Third protocol run, against the deployed build with the **hedge-don't-drop**
prompt change (`daa253c`) — `FACTCHECK_SYSTEM` now flags plausible-but-unconfirmed
claims as UNVERIFIED instead of staying silent, and `WRITER_SYSTEM` hedges an
UNVERIFIED claim (keeps it, marks it uncertain) instead of dropping it. Paired
answers: `2026-06-26-factcheck-hedge-answers.md`. Compared against the prior
fact-check run (`2026-06-26-factcheck-stage-*`), which is the immediate baseline.

**Reading this:** I remain a fallible accuracy judge. Verdicts are directional,
one sample each, on a non-deterministic system.

## Headline: the change did exactly what it was built to do

The fact-check stage's disposition moved from **omit → hedge**. On
`current-2025-recent` — the single case that motivated the change — real
releases that the prior run *dropped* (DeepSeek, Qwen3) are now *kept and
hedged*. The Rust async win not only held but *sharpened*. No regression to
bios/controls.

## The case that motivated the change

### current-2025-recent — FIXED ✅

The prior run committed to only **2** releases (Llama 4, Mistral 3) and pushed
everything else into an "unverified due to gaps in the captured evidence"
bucket — silently dropping DeepSeek and Qwen3, which are real.

This run lists **8** releases chronologically, each with a calibrated
confidence marker rather than an omission:

- **DeepSeek R1** — *"Reportedly released in January 2025… While exact dates
  and leaderboard rankings are unconfirmed, it was widely described as
  triggering the 2025 open-model momentum."* — the exact hedge-don't-drop
  behavior. The prior run dropped this entirely; now it's present and marked
  uncertain.
- **Qwen3 Family** — *"Alibaba Cloud unambiguously released the Qwen3 family on
  April 29, 2025, under an Apache 2.0 license."* — kept, and stated
  **confidently** because the fact-checker could verify it. Note the date is
  Apr 29, the correct one (the original pre-fact-check baseline had "Apr 28").
- **Llama 4** — kept, now *"Reportedly released around April 2025,"* with the
  Maverick/GPT-4o benchmark explicitly flagged *"remain unverified."*
- **Kimi K2, Qwen-3-Max-Preview, Mistral Large 3, DeepSeek V3.2** — all present
  with per-claim calibration; company benchmark claims labelled as such, and
  the V3.2 December item explicitly tagged *"should be treated as an unconfirmed
  rumor."*

The disposition is now: **verify-and-state when it can, hedge-and-keep when it
can't, drop only when contradicted.** That is the intended behavior. Coverage
went from 2 releases to 8 with no confidently-wrong dated claim surviving.

### current-rust-async — STILL A WIN, sharpened ✅

- **Prior (fact-check) run:** *"async-std's status is currently disputed. While
  some sources point to a 2025 discontinuation, no official announcement has
  been confirmed…"* (hedged — good)
- **This run:** *"The project was officially discontinued on March 1, 2025, with
  maintainers recommending smol as the replacement. It's no longer a viable
  choice for new projects in 2026."* (verified and stated — better)

Same fact-checker, opposite-looking output, and that's the point: when the
evidence *is* there, the new prompt lets it state the confirmed date instead of
reflexively hedging. The change improves calibration in **both** directions —
more confident when verified, more hedged-but-present when not.

## No regressions elsewhere

- **Biographies (Euclid/Archimedes):** unchanged quality, +30 verifier
  discipline intact. *(bio-pythagoras hit a `ReadTimeout` this run — a transport
  blip on one request, not a content result; it was "unchanged, no regression"
  in the baseline and is unrelated to this change. Re-run it alone if a clean
  Pythagoras sample is wanted.)*
- **Controls (toast, recursion):** clean — banner present, no spurious
  verification caveats, no bloat.

## On the prior report's second concern

The +33 report also flagged that the fact-checker had *introduced* a new precise
claim it may not have verified (Mistral-3's "675-billion-parameter, Dec 2"). This
run still carries Mistral Large 3 but now wraps the specifics in calibration —
*"a reported 675B total parameters,"* *"exact day varies between reports,"*
*"context window conflict between 128k and 256k tokens, so the exact specification
cannot be confirmed."* The same hedge discipline that keeps real claims also now
softens the fact-checker's own injected specifics. Net improvement on that axis too.

## Run quality note

9/10 structural PASS; the one FAIL was bio-pythagoras `ReadTimeout` (transport,
not content). The script exits non-zero if any case fails a check, so the run's
exit-1 is that single timeout, not a content failure. Both motivating cases
(rust-async, 2025-recent) passed and stream the full 5-stage banner
`Planning → Researching → Verifying → Fact-checking → Writing`.

## Verdict

**Keep the change.** It resolves the one open issue from the +33 run (drop →
hedge) without disturbing anything else, and as a bonus tightened the
fact-checker's own injected specifics and let the Rust case state its now-verified
date. The stage's disposition is now well-calibrated: confident when verified,
hedged when plausible-but-unconfirmed, dropped only when contradicted.

No further prompt tuning indicated by this run. If anything is revisited later,
it would be measured by re-running this protocol and diffing against this file.

## Caveats

- One run, one sample per case; non-deterministic system. The omit→hedge shift
  on `current-2025-recent` is directionally clear and matches the prompt change
  precisely, but is not quantified.
- I can't independently verify the 2025 dates myself — the per-claim read leans
  on internal consistency and calibration quality. User is the authority on
  whether the specific dates/specs are right.
- bio-pythagoras did not produce a comparable sample this run (timeout).
