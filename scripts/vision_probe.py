"""Experiment harness for the keyframe describe call (Phase 38).

Phase 38 twice reasoned about a characters-per-token ratio and twice got it
wrong, at the cost of a deploy each time and three silently dropped keyframes.
This runs the variants side by side and prints what actually came back.

## What is already settled (qwen3-vl:32b, 2026-08-04)

**Thinking cannot be turned off.** Neither `think: false` nor Qwen3's
`/no_think` prompt switch reduces it — 93-101% of baseline either way. Treat
reasoning as a cost of this model, not a setting. `MODE=think` re-runs that
check against a different model.

**Content + thinking accounts for every generated token** (3.4-4.4 chars/tok in
every variant). There is no third thing to find.

**Thinking scales with visual clutter, not with output length.** A static
two-shot produced ~850 characters of it; an office desk produced ~5,700, and
the reasoning text shows why — it was hunting for legible text on a coffee cup
and a stack of paper because the prompt told it to transcribe text exactly.

**The cap is spent on thinking before any description is written.** Three of
six runs on the cluttered frame hit `num_predict: 2048` and emitted ZERO
characters, which reaches the worker as a 502 and drops the keyframe. That is
the failure this harness exists to prevent shipping again.

## Modes

    MODE=prompts   (default) every prompt variant, fixed sampling. The main
                   experiment: which instruction produces a usable description
                   for the fewest tokens.
    MODE=sampling  one prompt, num_predict x temperature. Answers "does 4096
                   rescue the frames that came back empty".
    MODE=think     think/no_think variants. Settled for qwen3-vl; re-run when
                   changing model.

## Judging quality, not just cost

A cheap prompt that loses the on-screen text is a regression, and no timing
column can say so. `EXPECT` names the strings that SHOULD appear, per image:

    EXPECT='a.jpg:ACOM TECHNOLOGIES;b.jpg:enertec,AM'

Matching is case-insensitive substring. The summary then reports `found 2/2`
beside the cost, so the trade is visible in one table — and the full text of
every description is printed at the end, because a hit count is a proxy and
reading them is not.

## Running it

The worker cannot reach Ollama (its compose network is `internal: true`, by
design), so this runs inside `audrey-ai`, the only container on both networks.
Fed over stdin so no rebuild is needed:

    # Unraid box
    docker exec -i -e IMAGES=/tmp/a.jpg,/tmp/b.jpg \
      -e EXPECT='a.jpg:ACOM TECHNOLOGIES;b.jpg:enertec' \
      audrey-ai python3 - < scripts/vision_probe.py

Extracting frames is two steps, because `audrey-ai` has no ffmpeg (the whole
reason the sidecar exists) and the worker deletes its frames when a job ends:

    # Unraid box
    docker exec media-worker sh -c \
      'ffmpeg -loglevel error -ss 30 -i /data/uploads/<user>/<file_id>.mp4 \
         -frames:v 1 -vf scale=1280:-2 -q:v 3 /tmp/a.jpg'
    docker cp media-worker:/tmp/a.jpg /tmp/a.jpg
    docker cp /tmp/a.jpg audrey-ai:/tmp/a.jpg

**Use more than two images, and make them different in kind.** Thinking varied
7x between a talking head and a cluttered desk, so a conclusion drawn from one
kind of frame will not hold. A slide, a whiteboard, a screen recording and a
dark or motion-blurred frame are each worth one.

Environment:

    IMAGES      comma-separated paths inside the container (default /tmp/probe.jpg)
    EXPECT      'img.jpg:text,text;other.jpg:text' — strings that should appear
    MODE        prompts | sampling | think        (default prompts)
    ONLY        comma-separated variant names, to re-run one
    SAMPLES     runs per variant (default 2) — thinking length is noisy
    MODEL       default qwen3-vl:32b
    OLLAMA_HOST default http://ollama:11434
    HINT        transcript context, as the worker now sends
    NUM_PREDICT default 4096, matching config.yaml
    TEMPERATURE default 0.3, matching config.yaml
    TIMEOUT_S   default 240
    EXCERPT     chars of each description to print (default 700, 0 = off)
    THINKING    1 to also print the reasoning text (long)
"""

from __future__ import annotations

import base64
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from audrey.pipeline.vision import DESCRIBE_SYSTEM, KEYFRAME_SYSTEM, TRANSCRIPT_HINT

MODEL = os.environ.get("MODEL", "qwen3-vl:32b")
HOST = os.environ.get("OLLAMA_HOST", "http://ollama:11434").rstrip("/")
HINT = os.environ.get("HINT", "")
TIMEOUT_S = float(os.environ.get("TIMEOUT_S", "240"))
MODE = os.environ.get("MODE", "prompts")
NS = 1_000_000_000.0

_DEFAULT_IMAGE = "/tmp/probe.jpg"  # noqa: S108
_NUM_PREDICT = int(os.environ.get("NUM_PREDICT", "4096"))
_TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.3"))
_SAMPLES = int(os.environ.get("SAMPLES", "2"))
_EXCERPT = int(os.environ.get("EXCERPT", "700"))
_SHOW_THINKING = os.environ.get("THINKING", "") == "1"

#: A variant leaving thinking under this fraction of baseline counts as
#: suppressing it. Loose on purpose — the useful answer is "does this mechanism
#: work at all", and two samples cannot distinguish 3% from 8%.
_SUPPRESSED_AT = 0.25


# ─── Prompt variants ───────────────────────────────────────────────────
#
# Each isolates one hypothesis about where the reasoning comes from. The
# reasoning text on the box showed the model hunting for legible characters on
# a coffee cup and a stack of paper — so the OCR instruction is the prime
# suspect, and `no-ocr` is the control that measures its cost directly.

#: What shipped before the "do not strain" clause was added. The A-B for that
#: change: it demanded every visible character be transcribed exactly, which is
#: what sent the model looking at cup logos.
_STRICT = KEYFRAME_SYSTEM.replace(
    "- Lead with any text that is CLEARLY LEGIBLE: slides, documents, "
    "whiteboards, captions, name plates, titles, signs, logos, code, error "
    "messages. Transcribe that exactly — it is usually the only thing anyone "
    "will search for.\n"
    "- Do NOT strain to decipher text. If something is small, blurred, angled "
    "or partly hidden, leave it out — do not guess at it and do not work at "
    "it. Deciphering unclear text is the most expensive thing you can do here "
    "and the least reliable.\n",
    "- Lead with any text visible in the frame, transcribed EXACTLY: slides, "
    "documents, whiteboards, captions, name plates, titles, signs, code, "
    "error messages. This is the most valuable thing you can record and often "
    "the only thing anyone will search for.\n",
)

#: Minimal instruction. Tests whether the elaborate prompt is earning its cost
#: at all, or whether a short one gets the same description for fewer tokens.
_TERSE = (
    "Describe this still frame from a video in two or three plain sentences, "
    "so it can be found later by search. Transcribe any text you can read "
    "easily. Plain prose only — no markdown, no bullet points, no headings."
)

#: No text instruction whatsoever. If thinking collapses here, the OCR demand
#: is the cost and the question becomes how to ask for text cheaply. If it does
#: not, the model reasons about images regardless and the prompt is not the
#: lever.
_NO_OCR = (
    "Describe this still frame from a video in two or three plain sentences, "
    "so it can be found later by search. Say what is happening and who or what "
    "is present. Plain prose only — no markdown, no bullet points, no "
    "headings."
)

#: Asks for the text and nothing else. The opposite extreme from `no-ocr`, and
#: a real candidate: if a transcript already covers what was said and a summary
#: covers the gist, on-screen text may be all the visual pass owes.
_TEXT_ONLY = (
    "Look at this still frame from a video and report ONLY the text that is "
    "clearly legible in it — slides, documents, whiteboards, captions, name "
    "plates, titles, signs, logos. Transcribe it exactly, in plain prose with "
    "no markdown. Do not describe people, furniture, rooms or actions. If "
    "there is no clearly legible text, reply with the single word NONE."
)

#: TESTED AND REJECTED 2026-08-04. Do not re-propose without new evidence.
#:
#: Proposed after seeing `current` open a description with
#: "92-7 106L-388342 F491587" — apparently guessing at unreadable reference
#: codes and writing the guesses into a retrieval chunk.
#:
#: Two things were wrong with that. **The problem was an artifact**: those
#: serials came from a frame extracted with the wrong seek, which the pipeline
#: had never described. Against the pipeline's real keyframes `current`
#: produces clean output ("enertec A man in a blue shirt...") and there are no
#: serials to skip.
#:
#: **And the cure was worse.** Over four real keyframes this cost 3,514 tokens
#: against `current`'s 2,268 — 55% more for identical coverage — and on the
#: calendar frame it misread "July 2024" as "May 2024" where `current`,
#: `strict` and `text-only` all read it correctly. One more instruction bought
#: more reasoning and a wrong answer.
#:
#: Kept as a variant because the negative result is worth being able to
#: reproduce, and because "add a clause to fix a description" is a tempting
#: move that this measured as a regression.
_NO_SERIALS = KEYFRAME_SYSTEM + (
    "\n- Skip reference numbers, serial numbers, order codes, phone numbers "
    "and part numbers. They are never what someone is searching for, and they "
    "are the text you are most likely to misread. Report names, titles, "
    "headings, company names, slide text and error messages — text that means "
    "something to a reader."
)

PROMPTS: dict[str, str] = {
    "current": KEYFRAME_SYSTEM,
    "no-serials": _NO_SERIALS,
    "strict": _STRICT,
    "terse": _TERSE,
    "no-ocr": _NO_OCR,
    "text-only": _TEXT_ONLY,
    "screenshot": DESCRIBE_SYSTEM,
}


@dataclass
class Trial:
    name: str
    system: str
    think: bool | None = False
    suffix: str = ""
    num_predict: int = _NUM_PREDICT
    temperature: float = _TEMPERATURE


def build_trials() -> list[Trial]:
    if MODE == "think":
        return [
            Trial("think-false", KEYFRAME_SYSTEM, think=False),
            Trial("think-true", KEYFRAME_SYSTEM, think=True),
            Trial("no_think-tag", KEYFRAME_SYSTEM, think=None, suffix=" /no_think"),
            Trial("no_think+false", KEYFRAME_SYSTEM, think=False, suffix=" /no_think"),
        ]
    if MODE == "sampling":
        return [
            Trial(f"np{np}-t{t}", KEYFRAME_SYSTEM, num_predict=np, temperature=t)
            for np in (2048, 4096)
            for t in (0.1, 0.3, 0.7)
        ]
    return [Trial(name, system) for name, system in PROMPTS.items()]


@dataclass
class Result:
    image: str
    variant: str
    ok: bool = True
    error: str = ""
    tokens: int = 0
    content: str = ""
    thinking: str = ""
    eval_s: float = 0.0
    done_reason: str = ""
    extra_keys: tuple[str, ...] = field(default_factory=tuple)

    @property
    def both_per_tok(self) -> float:
        """Content+thinking characters per generated token.

        Near 4 means every token is explained by text we can see. Well under
        means something is being generated that neither field carries.
        """
        return (len(self.content) + len(self.thinking)) / self.tokens if self.tokens else 0.0

    @property
    def lost(self) -> bool:
        """Truncated before writing anything — a dropped keyframe in production."""
        return self.done_reason == "length" and not self.content.strip()


def parse_expect() -> dict[str, list[str]]:
    """`'a.jpg:ACOM,fireplace;b.jpg:enertec'` → {'a.jpg': ['ACOM', 'fireplace']}."""
    out: dict[str, list[str]] = {}
    for group in os.environ.get("EXPECT", "").split(";"):
        if ":" not in group:
            continue
        name, _, terms = group.partition(":")
        wanted = [t.strip().lower() for t in terms.split(",") if t.strip()]
        if wanted:
            out[name.strip()] = wanted
    return out


def found(result: Result, expect: dict[str, list[str]]) -> tuple[int, int]:
    wanted = expect.get(result.image, [])
    if not wanted:
        return 0, 0
    body = result.content.lower()
    return sum(1 for w in wanted if w in body), len(wanted)


def load(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def call(b64: str, trial: Trial, image: str) -> Result:
    """One /api/chat call, built exactly as `_transcribe_one` builds it."""
    hint = TRANSCRIPT_HINT.format(hint=HINT) if HINT.strip() else ""
    payload: dict = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": trial.system},
            {
                "role": "user",
                "content": f"Describe this image.{hint}{trial.suffix}",
                "images": [b64],
            },
        ],
        "stream": False,
        "options": {
            "num_predict": trial.num_predict,
            "temperature": trial.temperature,
        },
    }
    if trial.think is not None:
        payload["think"] = trial.think

    try:
        r = httpx.post(f"{HOST}/api/chat", json=payload, timeout=TIMEOUT_S)
    except httpx.HTTPError as e:
        # One variant erroring must not lose every other call's data.
        return Result(image, trial.name, ok=False, error=f"{type(e).__name__}: {e}")
    if r.status_code != 200:
        return Result(image, trial.name, ok=False,
                      error=f"HTTP {r.status_code}: {r.text[:300]}")

    body = r.json()
    message = body.get("message") or {}
    thinking = message.get("thinking")
    return Result(
        image=image,
        variant=trial.name,
        tokens=int(body.get("eval_count") or 0),
        content=str(message.get("content") or ""),
        thinking=thinking if isinstance(thinking, str) else "",
        eval_s=float(body.get("eval_duration") or 0) / NS,
        done_reason=str(body.get("done_reason") or ""),
        extra_keys=tuple(sorted(set(message) - {"role", "content", "thinking"})),
    )


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def summarise(
    results: list[Result], images: list[str], trials: list[Trial],
    expect: dict[str, list[str]],
) -> None:
    print("\n" + "=" * 86)
    print("PER-VARIANT MEANS   (cost on the left, quality on the right)")
    print("=" * 86)
    for image in images:
        print(f"\n{image}")
        print(f"  {'variant':<16} {'gen':>7} {'tokens':>7} {'content':>9} "
              f"{'thinking':>10} {'lost':>5}  {'found':>7}")
        for trial in trials:
            rows = [r for r in results
                    if r.image == image and r.variant == trial.name and r.ok]
            if not rows:
                failed = [r for r in results
                          if r.image == image and r.variant == trial.name]
                if failed:
                    print(f"  {trial.name:<16} ERROR  {failed[0].error[:52]}")
                continue
            hits = [found(r, expect) for r in rows]
            total_wanted = hits[0][1] if hits else 0
            hit_txt = (
                f"{_mean([float(h[0]) for h in hits]):.1f}/{total_wanted}"
                if total_wanted else "—"
            )
            print(
                f"  {trial.name:<16} "
                f"{_mean([r.eval_s for r in rows]):6.1f}s "
                f"{_mean([float(r.tokens) for r in rows]):7.0f} "
                f"{_mean([float(len(r.content)) for r in rows]):8.0f}ch "
                f"{_mean([float(len(r.thinking)) for r in rows]):9.0f}ch "
                f"{sum(1 for r in rows if r.lost):3d}/{len(rows):<2} "
                f"{hit_txt:>8}"
            )
            extras = {k for r in rows for k in r.extra_keys}
            if extras:
                print(f"  {'':<16} ^ message also carries: {sorted(extras)}")


def verdict(
    results: list[Result], images: list[str], trials: list[Trial],
    expect: dict[str, list[str]],
) -> None:
    print("\n" + "=" * 86)
    print("VERDICT")
    print("=" * 86)

    ok = [r for r in results if r.ok]
    if not ok:
        print("Every call failed — nothing to conclude.")
        return

    lost = [r for r in ok if r.lost]
    if lost:
        print(
            f"\n{len(lost)}/{len(ok)} calls hit num_predict while reasoning and "
            "returned NOTHING.\nIn production each of those is a 502 and a "
            "dropped keyframe. Raise vision.num_predict\nor pick a prompt that "
            "reasons less — this is the failure mode, not a slow frame."
        )
        for r in lost:
            print(f"    {r.image} {r.variant}: {r.tokens}tok, "
                  f"{len(r.thinking)}ch thinking, 0ch content")

    unexplained = [r for r in ok if r.tokens >= 100 and r.both_per_tok < 2.0]
    if unexplained:
        print(
            f"\n{len(unexplained)}/{len(ok)} calls generated tokens that neither "
            "content nor thinking\naccounts for. Check the extra message keys "
            "above before tuning anything."
        )

    if MODE == "think":
        _think_verdict(ok, images, trials)
        return

    # Cost-per-useful-description, which is the number that should pick a
    # prompt. A variant that is cheap because it lost the text is not cheap.
    print("\nRanked by tokens, with what each one cost you in coverage:")
    for image in images:
        wanted = len(expect.get(image, []))
        print(f"\n{image}" + (f"  (expecting {wanted} string(s))" if wanted else ""))
        ranked = []
        for trial in trials:
            rows = [r for r in ok if r.image == image and r.variant == trial.name]
            if not rows:
                continue
            hits = [found(r, expect) for r in rows]
            ranked.append((
                _mean([float(r.tokens) for r in rows]),
                trial.name,
                _mean([float(h[0]) for h in hits]),
                sum(1 for r in rows if r.lost),
            ))
        for tokens, name, hit, n_lost in sorted(ranked):
            flags = []
            if wanted and hit < wanted:
                flags.append(f"MISSED {wanted - hit:.1f} of {wanted}")
            if n_lost:
                flags.append(f"{n_lost} LOST")
            suffix = ("   <- " + ", ".join(flags)) if flags else ""
            print(f"  {tokens:7.0f}tok  {name:<16}"
                  + (f"  found {hit:.1f}/{wanted}" if wanted else "")
                  + suffix)

    print(
        "\nPick the cheapest variant that misses nothing and loses nothing — "
        "then READ its\ndescriptions below. A hit count cannot tell you whether "
        "the prose is any good."
    )


def _think_verdict(ok: list[Result], images: list[str], trials: list[Trial]) -> None:
    for image in images:
        base = [r for r in ok if r.image == image and r.variant == "think-false"]
        if not base:
            continue
        baseline = _mean([float(len(r.thinking)) for r in base])
        print(f"\n{image} — think-false thinking: {baseline:.0f}ch")
        if baseline == 0:
            print("  No thinking at all. `think: false` is honoured here.")
            continue
        for trial in trials:
            if trial.name == "think-false":
                continue
            rows = [r for r in ok if r.image == image and r.variant == trial.name]
            if not rows:
                continue
            mean_think = _mean([float(len(r.thinking)) for r in rows])
            ratio = mean_think / baseline
            mark = "  <<< SUPPRESSES IT" if ratio <= _SUPPRESSED_AT else ""
            print(f"  {trial.name:<16} {mean_think:7.0f}ch  {ratio * 100:5.1f}% "
                  f"of baseline{mark}")
    print(
        "\nNothing marked means thinking cannot be turned off for this model. "
        "Accept the\ncost, or move to one whose `ollama show` omits `thinking`."
    )


def transcripts(results: list[Result], images: list[str], trials: list[Trial]) -> None:
    """Print the descriptions. The hit count is a proxy; this is the evidence."""
    if _EXCERPT <= 0:
        return
    print("\n" + "=" * 86)
    print(f"THE DESCRIPTIONS (first {_EXCERPT} chars of one run each)")
    print("=" * 86)
    for image in images:
        for trial in trials:
            row = next((r for r in results
                        if r.image == image and r.variant == trial.name and r.ok), None)
            if row is None:
                continue
            note = "  [TRUNCATED, NOTHING WRITTEN]" if row.lost else ""
            print(f"\n── {image} · {trial.name} · {len(row.content)}ch{note} "
                  + "─" * 18)
            print(row.content[:_EXCERPT].strip() or "(empty)")
            if _SHOW_THINKING and row.thinking:
                print(f"   ·· thinking ({len(row.thinking)}ch) ··")
                print(row.thinking[:_EXCERPT].strip())


def main() -> int:
    paths = [
        Path(p.strip())
        for p in os.environ.get("IMAGES", _DEFAULT_IMAGE).split(",")
        if p.strip()
    ]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"No such image(s): {[str(p) for p in missing]}")
        print("See this file's docstring for how to extract frames into the container.")
        return 2

    trials = build_trials()
    only = {v.strip() for v in os.environ.get("ONLY", "").split(",") if v.strip()}
    if only:
        trials = [t for t in trials if t.name in only]
    if not trials:
        print(f"ONLY={sorted(only)} matched nothing in MODE={MODE}.")
        return 2

    expect = parse_expect()
    unknown = set(expect) - {p.name for p in paths}
    if unknown:
        print(f"WARNING: EXPECT names images not being probed: {sorted(unknown)}\n")

    # `_STRICT` is built by removing a clause from the DEPLOYED
    # `KEYFRAME_SYSTEM`. If the image predates that clause the replace is a
    # no-op, the two variants are the same prompt, and the comparison reports
    # noise as a result — which is exactly what happened on 2026-08-04, and was
    # only caught by someone asking whether a rebuild was needed.
    degenerate = [
        name for name, system in PROMPTS.items()
        if name not in ("current",) and system == KEYFRAME_SYSTEM
    ]
    if degenerate and any(t.name in degenerate for t in trials):
        print(
            f"WARNING: {degenerate} are IDENTICAL to `current` in this image.\n"
            "         The deployed KEYFRAME_SYSTEM is not the one these "
            "variants were derived\n         from — rebuild audrey-ai, or "
            "these comparisons measure nothing.\n"
        )

    total = len(paths) * len(trials) * _SAMPLES
    print(f"model={MODEL} host={HOST} mode={MODE} hint={'yes' if HINT.strip() else 'no'}")
    print(f"num_predict={_NUM_PREDICT} temperature={_TEMPERATURE} samples={_SAMPLES}")
    print(f"{len(paths)} image(s) x {len(trials)} variant(s) x {_SAMPLES} = {total} calls")
    print(f"Roughly {total * 8 // 60}-{total * 60 // 60} minutes.\n")

    results: list[Result] = []
    started = time.monotonic()
    for path in paths:
        b64 = load(path)
        for trial in trials:
            for run in range(_SAMPLES):
                res = call(b64, trial, path.name)
                results.append(res)
                done = len(results)
                if not res.ok:
                    print(f"[{done}/{total}] {path.name:<10} {trial.name:<16} "
                          f"ERROR {res.error[:56]}")
                    continue
                hit, want = found(res, expect)
                print(
                    f"[{done}/{total}] {path.name:<10} {trial.name:<16} "
                    f"run{run + 1} {res.eval_s:6.1f}s {res.tokens:5d}tok "
                    f"content={len(res.content):5d}ch think={len(res.thinking):6d}ch"
                    + (f" found={hit}/{want}" if want else "")
                    + ("  <-- LOST" if res.lost else "")
                )

    names = [p.name for p in paths]
    summarise(results, names, trials, expect)
    verdict(results, names, trials, expect)
    transcripts(results, names, trials)
    print(f"\nTotal probe time: {(time.monotonic() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
