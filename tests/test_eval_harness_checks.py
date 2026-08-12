"""Hermetic tests for the eval harness's pure pieces (scripts/eval_research.py).

The harness itself is a LIVE tool — it needs the box, OWUI, and a network. But
its newest checks (code extraction + execution, answer_contains), the sweep
expansion, the JSON results writer, and eval_compare's table builder are pure
functions we can pin offline. The subprocess in _run_code_check runs
`sys.executable` on a temp file — no network, no stack, still hermetic.

Same import pattern as test_web_search_routing.py: scripts/ isn't a packaged
module, so we add it to sys.path and import the harness directly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import eval_compare  # noqa: E402
import eval_research as er  # noqa: E402

# ── code-block extraction ───────────────────────────────────────────────────

def test_extract_largest_python_block():
    answer = (
        "Here's a usage example:\n\n```python\nf(1)\n```\n\n"
        "And the implementation:\n\n```python\ndef f(x):\n    return x + 1\n```\n"
    )
    code = er._extract_code_block(answer, "python")
    assert code is not None and "def f(x):" in code


def test_extract_accepts_py_alias():
    answer = "```py\ndef g():\n    return 7\n```\n"
    code = er._extract_code_block(answer, "python")
    assert code is not None and "def g():" in code


def test_extract_none_when_no_python_block():
    assert er._extract_code_block("no code here", "python") is None
    # A bash block is not a python block.
    assert er._extract_code_block("```bash\nls -la\n```", "python") is None


def test_extract_ignores_untagged_fence():
    assert er._extract_code_block("```\nprint('hi')\n```", "python") is None


def test_has_tagged_code_block():
    assert er._has_tagged_code_block("```bash\nls\n```") is True
    assert er._has_tagged_code_block("```\nls\n```") is False
    assert er._has_tagged_code_block("plain prose") is False


def test_extraction_stops_at_debug_blocks():
    # With debug_panel_drafts on, worker drafts stream AFTER the answer and can
    # carry their own (bigger) code fences — extraction must not grab those.
    answer = (
        "```python\ndef real():\n    return 1\n```\n\n"
        "## Panel drafts (debug)\n\n"
        "```python\ndef draft():\n    # a much longer draft implementation\n"
        "    total = 0\n    for i in range(100):\n        total += i\n"
        "    return total\n```\n"
    )
    code = er._extract_code_block(answer, "python")
    assert code is not None and "def real():" in code and "draft" not in code


# ── code execution ──────────────────────────────────────────────────────────

def test_run_code_check_passes():
    ok, detail = er._run_code_check(
        "def f(x):\n    return x * 2\n", "assert f(3) == 6\n", 10.0)
    assert ok is True
    assert detail == "exit 0"


def test_run_code_check_assert_failure_carries_detail():
    ok, detail = er._run_code_check(
        "def f(x):\n    return x\n", "assert f(3) == 6\n", 10.0)
    assert ok is False
    assert "AssertionError" in detail


def test_run_code_check_times_out():
    ok, detail = er._run_code_check("while True:\n    pass\n", "pass\n", 1.0)
    assert ok is False
    assert "timeout" in detail


def test_run_code_check_syntax_error():
    ok, detail = er._run_code_check("def f(:\n", "pass\n", 10.0)
    assert ok is False
    assert "exit" in detail


# ── answer_contains ─────────────────────────────────────────────────────────

def test_contains_all_case_insensitive():
    assert er._contains_all("The answer is K2 at 8611m.", ["k2", "8611"]) is True
    assert er._contains_all("The answer is K2.", ["k2", "8611"]) is False


def test_contains_ignores_debug_region():
    answer = "Final answer: unsure.\n\n## Research trace (debug)\n\nnotes say 82.8"
    assert er._contains_all(answer, ["82.8"]) is False


# ── expect_names_files ──────────────────────────────────────────────────────

_SHORT = "How to WIN with the London System.mp4"
_LONG = "Magnus Carlsen Teaches How to Win with the London System.mp4"
# The shape used by the video suite: alternatives per file.
_FILES = [
    [_SHORT, "How to WIN with the London System"],
    [_LONG, "Magnus Carlsen Teaches How to Win with the London System",
     "Magnus Carlsen video", "Carlsen video", "Magnus Carlsen"],
]
# The suite's own group as of 2026-08-11, informal aliases included.
_WIDE_FILES = [
    [_SHORT, "How to WIN with the London System", '"How to WIN" video',
     "How to WIN video", "GothamChess video", "Rozman video"],
    _FILES[1],
]


def test_naming_both_files_passes():
    answer = f"You have two: {_SHORT} and {_LONG}. Which did you mean?"
    assert er._names_all_files(answer, _FILES) is True


def test_naming_one_file_fails():
    """The regression this exists for. `video-ambiguous-singular` answered from
    one of two same-topic candidates, and every other check scored it PASS."""
    answer = f"Based on {_SHORT}, the move order begins with 1. d4."
    assert er._names_all_files(answer, _FILES) is False


def test_the_longer_filename_alone_does_not_satisfy_both():
    """⚠️ Why this is not `_contains_all`.

    `_LONG` contains `_SHORT` as a substring, so a plain contains-check on
    both is satisfied by an answer that named only the longer file — reporting
    a pass on the exact failure being tested. Resolving most-specific-first
    and consuming the match forces a second, separate mention.
    """
    answer = f"Based on {_LONG}, the move order begins with 1. d4."

    assert er._contains_all(answer, [_SHORT, _LONG]) is True  # the trap
    assert er._names_all_files(answer, _FILES) is False


def test_an_informal_reference_counts():
    """⚠️ A check that false-fails gets ignored, which is worse than no check.

    Models write "the Magnus Carlsen video" at least as often as the filename,
    and an answer that clearly distinguishes both files has done the thing
    being tested. Taken from a real `audrey_auto` answer, 2026-08-10.
    """
    answer = (
        'Based on your two London System videos: in the Magnus Carlsen video, '
        'Carlsen calls it a "solid choice". Your longer instructional video '
        '("How to WIN with the London System") emphasises middlegame plans.'
    )
    assert er._names_all_files(answer, _FILES) is True


def test_a_plain_string_still_works():
    """One file, one string — the list-of-alternatives form is opt-in."""
    assert er._names_all_files(f"read {_SHORT}", [_SHORT]) is True


def test_naming_is_case_insensitive():
    answer = "i checked how to win with the london system.mp4 and " + _LONG.lower()
    assert er._names_all_files(answer, _FILES) is True


def test_a_shortened_title_in_quotes_counts():
    """⚠️ The false fail of 2026-08-11. `audrey_video` wrote its two section
    headers as "The Magnus Carlsen Video" and 'The "How to WIN" Video' — which
    is exactly what this check is for, since it says which file each section
    came from. It scored ❌ because the alias list only held the full title.

    The check exists to see WHICH file was answered from, not to grade
    transcription of a filename.
    """
    answer = (
        'Based on both: **The Magnus Carlsen Video (~7:37)** covers his opinion '
        'of the opening. **The "How to WIN" Video (~29:38)** is a full lesson.'
    )
    assert er._names_all_files(answer, _WIDE_FILES) is True


def test_curly_quotes_do_not_defeat_an_alias():
    """Models emit typographic quotes; the alias list is written in ASCII.
    `_names_all_files` reads the normalised prose region for this reason."""
    answer = (
        "The Magnus Carlsen video is a blitz game. The “How to WIN” "
        "video is a structured lesson."
    )
    assert er._names_all_files(answer, _WIDE_FILES) is True


def test_the_short_aliases_still_cannot_be_satisfied_by_the_carlsen_title():
    """⚠️ Why every alias carries `video` or a channel name. Bare `How to WIN`
    is a substring of `Magnus Carlsen Teaches How to Win with the London
    System`, so an alias that short would let one mention answer for both files
    — reinstating the exact defect `_names_all_files` was written to close."""
    answer = f"Based on {_LONG}, the move order begins with 1. d4."
    assert er._names_all_files(answer, _WIDE_FILES) is False


def test_naming_ignores_the_debug_region():
    """Same rule as `contains`: a filename mentioned only in the trace is not
    the model naming it to the user."""
    answer = f"Based on {_SHORT}.\n\n## Research trace (debug)\n\nread {_LONG}"
    assert er._names_all_files(answer, _FILES) is False


# ── answer_not_contains ─────────────────────────────────────────────────────
#
# ⚠️ Added 2026-08-10 after `video-long-transcript-paging` scored PASS on the
# exact failure it was written to catch: three `get_file_text` pages in hand,
# no bounds given, nothing offered to continue, and the user told to "access
# the file directly". Some failures have a signature WORDING; that is
# checkable, where the good behaviour has a hundred valid phrasings.


def test_a_banned_phrase_fails():
    answer = ("I cannot provide the full transcript because I only retrieved "
              "partial sections due to output length constraints.")
    assert er._contains_any(answer, ["output length constraint"]) is True


def test_a_good_answer_trips_nothing():
    """The real `audrey_auto` answer from the same run — bounds and an offer."""
    answer = ("The transcript is quite long (33,626 characters total). That's "
              "the first ~4,005 characters, with about 29,600 remaining. Would "
              "you like me to continue reading it page by page?")
    assert er._contains_any(
        answer, ["system limitation", "output length constraint",
                 "access the file directly"]) is False


def test_any_one_banned_phrase_is_enough():
    """`_contains_any`, not `_contains_all` — one signature phrase is the
    failure; requiring all of them would pass every real case."""
    answer = "You would need to access the file directly."
    assert er._contains_any(answer, ["system limitation",
                                     "access the file directly"]) is True


def test_banned_phrases_ignore_the_debug_region():
    answer = "Here is page 1.\n\n## Research trace (debug)\n\nsystem limitation"
    assert er._contains_any(answer, ["system limitation"]) is False


# ── expect_continuation_offer ───────────────────────────────────────────────
#
# ⚠️ The second attempt at this case, and the reason the first one was wrong.
# `answer_not_contains` blacklisted the observed wordings ("output length
# constraints") on 2026-08-10 and was defeated on the NEXT run by both models
# paraphrasing without trying — "exceeds available context limits", "exceeds
# the available context window". A model rewords for free, so the check has to
# describe the shape of the failure, not the sentence that expressed it.
#
# The shape: refusing to hand over the whole thing is fine. Refusing with no
# way to continue is the failure. Both halves required.

_GOOD_PAGED = (
    "The transcript is quite long (33,626 characters total). Here's the first "
    "portion: [00:00:01] ladies and gentlemen... That's the first ~4,005 "
    "characters. Would you like me to continue reading it page by page?"
)
_BAD_CONTEXT_LIMIT = (
    "I cannot provide the full transcript because it exceeds available context "
    "limits. To get the full transcript, you would need to request it in "
    "sections using an offset parameter, or consult the file directly."
)
_BAD_GO_ELSEWHERE = (
    "I cannot provide a full transcript, which exceeds the available context "
    "window. You would need to use video transcription software."
)


def test_a_paged_answer_with_an_offer_passes():
    assert er._declines_without_offering(_GOOD_PAGED) is False


def test_refusing_with_no_way_forward_fails():
    assert er._declines_without_offering(_BAD_CONTEXT_LIMIT) is True


def test_being_sent_to_another_tool_entirely_fails():
    """Worst of the family: told to go transcribe a file Audrey has already
    transcribed in full."""
    assert er._declines_without_offering(_BAD_GO_ELSEWHERE) is True


def test_paraphrase_does_not_escape_it():
    """⚠️ The exact regression. Both of these slip past a substring blacklist
    of the previously-observed wordings; neither slips past the shape."""
    banned = ["system limitation", "output length constraint",
              "access the file directly"]

    for answer in (_BAD_CONTEXT_LIMIT, _BAD_GO_ELSEWHERE):
        assert er._contains_any(answer, banned) is False       # blacklist misses
        assert er._declines_without_offering(answer) is True   # shape catches


def test_refusing_but_offering_to_continue_is_allowed():
    """Declining to dump 33,000 characters is reasonable. The offer is what
    separates a judgement call from a dead end, so only the pair fails."""
    answer = ("I cannot provide the whole transcript in one message — it is "
              "33,626 characters. Would you like me to continue page by page?")
    assert er._declines_without_offering(answer) is False


def test_an_answer_that_never_refuses_is_untouched():
    """No refusal, no requirement — a case that simply answers must not be
    dragged into needing an offer."""
    answer = "Here is the continued transcript (pages 15-24): [00:06:48] ..."
    assert er._declines_without_offering(answer) is False


def test_inviting_the_user_to_ask_for_the_next_part_is_an_offer():
    """The false fail of 2026-08-11 run 8. `audrey_video` gave the exact
    character count, four sections of transcript, and closed with this — a
    perfectly good paged answer scored ❌."""
    answer = (
        "I cannot provide the full transcript because it contains "
        "approximately 33,626 characters. Here is what was retrieved. "
        "To receive more of the transcript, please ask for subsequent "
        'sections (e.g., "Give me the next part of the transcript").'
    )
    assert er._declines_without_offering(answer) is False


def test_inviting_the_user_to_say_the_word():
    """Run 12's false fail, plus a historical one the same scan turned up.
    Both are textbook offers built from words the family did not hold.

    ⚠️ Both close on "please let me know", which is what carries them. The
    same change also added "you would like me to", and run 13 showed that half
    was wrong — see the test below.
    """
    for answer in (
        "I cannot provide the full transcript in a single response because it "
        "is approximately 33,626 characters long. If you would like me to "
        "continue providing it in subsequent messages, please let me know, "
        "and I will retrieve the next section for you.",
        "I cannot provide the whole thing here. If you need specific details "
        "from later parts of the video, please let me know which section you "
        "are interested in, and I can retrieve that segment for you.",
    ):
        assert er._declines_without_offering(answer) is False, answer


def test_asking_the_user_to_supply_the_text_is_not_an_offer():
    """⚠️ A widening ACCEPTED one run and REMOVED the next — the clearest
    evidence in this file that one clean flip is not enough to justify one.

    Run 12 added "you would like me to" on a single archived flip that looked
    genuine. Run 13 met it live: `video-long-transcript-paging` [audrey_auto]
    declared it had no access to the file, told the user to look on YouTube,
    and offered to analyse a transcript the USER pastes — after three
    SUCCESSFUL `get_file_text` calls. The phrase is second person but it is not
    an invitation to ask for the next page; the direction of the work is
    backwards. Across the archive it was the sole match in four sections and
    carried a real offer in none.
    """
    answer = (
        "I cannot provide the full transcript because I do not have access to "
        "that specific video file or its contents. If you have a transcript "
        "text you would like me to analyze, summarize, or format, please paste "
        "it here. Alternatively, you may be able to find the transcript via "
        "the platform's automatic captioning features."
    )
    assert er._declines_without_offering(answer) is True


def test_a_capability_statement_about_something_else_is_not_an_offer():
    """⚠️ The second rejected widening. Matching first-person "I can provide"
    or "I will retrieve" flips one more archived answer than the two above —
    and that one offers a SUMMARY while sending the user away for the
    transcript. The offer has to be about continuing."""
    answer = (
        "I cannot provide the full transcript because I only have access to "
        "the first segment. However, I can provide the summary and key points "
        "from the available portion. For the remaining 20+ minutes, you would "
        "need to view the video directly."
    )
    assert er._declines_without_offering(answer) is True


def test_the_offer_widening_that_was_rejected_by_measurement():
    """⚠️ Fixing the above by matching "remaining part"/"next section" was
    measured over every archived paging answer and REJECTED: three flips, and
    two of them were dead-ends wearing an offer's vocabulary.

    The distinction is not whether more text is mentioned — it is WHO fetches
    it. `_OFFERS_MORE` only holds invitations for the user to ASK.
    """
    for dead_end in (
        "I cannot provide the whole transcript here. To obtain the full text, "
        "you would need to continue retrieving the remaining parts of the file "
        "using an offset starting at 4005.",
        "To get the full transcript, you would need to download the video file "
        "and use a speech-to-text service, as I cannot provide the complete "
        "text in this interaction.",
    ):
        assert er._OFFERS_MORE.search(dead_end) is None
        assert er._declines_without_offering(dead_end) is True


# ── degraded-context report (INFORMATIONAL, never a gate) ───────────────────
#
# ⚠️⚠️ THE MOST EXPENSIVE LESSON IN THIS FILE, and it lasted one day.
#
# Shipped 2026-08-11 as the always-on check `no_false_limit`, on the theory
# that a limit blamed for content the tools reached must be invented. The next
# morning a correlation over every archived answer killed that theory:
#
#   successful tool calls in a turn  →  fail rate
#     0: 16.9% (n=83)   1: 14.2% (n=155)   2: 6.9% (n=101)
#     3: 41.1% (n=95)   4–5: ~31% (n=38)
#
# A cliff at three, which is exactly where the fast path's
# `compress_after_round: 2` + `compress_keep_last: 1` leaves ONE tool message
# verbatim and stubs the rest. `video-long-transcript-paging` fails 0 of 14
# turns below three calls and 20 of 32 at three, while the cases needing only
# the LAST result are untouched (`control-named-scoped`, 0 of 23 at three).
#
# So "I am missing the middle portion" is most likely TRUE, and the check was
# failing answers for honestly describing a context they really were handed.
# **A harness that cannot see the input must not call the output a lie.**
# The detector survives because the signal is good; it reports and never gates.

_READ_OK = ("\n\n---\n> _Tools used:_\n> - **qwen3.6:35b** — "
            "`list_my_files` ✅1, `get_file_text` ✅2\n")
_READ_FAILED = ("\n\n---\n> _Tools used:_\n> - **qwen3.6:35b** — "
                "`get_file_text` ✅0 ❌1\n")


def test_it_flags_a_turn_worth_reading_the_context_trace_for():
    """Both verbatim from real runs. Both are now read as REPORTS of a thinned
    context, not as inventions — the answer that says it is missing the middle
    of a paged file made three calls and was handed one page."""
    for answer in (
        "Based on the available information, only partial transcripts for your "
        "videos are accessible due to file size limitations, which prevents a "
        "complete analysis of both files.",
        "I have retrieved the beginning and some later sections, but I am "
        "missing the middle portion due to technical limits.",
    ):
        assert er._reports_degraded_context(answer + _READ_OK), answer


def test_it_never_reaches_the_pass_fail_verdict(monkeypatch):
    """⚠️ The guard that matters. An answer this fires on must still PASS every
    check, or the retraction was cosmetic."""
    answer = ("I am missing the middle portion due to technical limits."
              + _READ_OK)
    monkeypatch.setattr(er, "_post_stream",
                        lambda *a, **k: (answer, [], "", er.StreamTiming()))
    r = er.run_case("u", "k", {"name": "c", "prompt": "p"}, "m", 5.0)
    assert "no_false_limit" not in r.checks
    assert r.context_detail                     # reported…
    assert r.ok                                 # …and not held against it


def test_a_limit_on_the_reply_itself_is_not_reported():
    """Two of the four raw archive hits were this, in answers that then paged
    and offered to continue. Declining to dump 33,000 characters into one reply
    says nothing about the context the model holds."""
    for fine in (
        "I cannot give the full transcript in this response because the file "
        "contains 33,626 characters, which exceeds the length limits for a "
        "single reply. Would you like the next section?",
        "I have only retrieved partial sections of the file in this answer due "
        "to output length constraints. Let me know if you want more.",
    ):
        assert er._reports_degraded_context(fine + _READ_OK) == "", fine


def test_a_genuine_tool_failure_is_never_reported():
    """It needs a ✅ to fire, so an answer written after a real failure — what
    `video-unknown-filename` exists to produce — is never flagged."""
    answer = ("I could not retrieve that file; it is not accessible due to a "
              "processing limitation.")
    assert er._reports_degraded_context(answer + _READ_FAILED) == ""
    assert er._reports_degraded_context(answer) == ""   # no footer


def test_the_report_names_both_halves():
    """Useless without both: the claim, and the calls that contradict it."""
    detail = er._reports_degraded_context(
        "The middle is missing due to technical limits." + _READ_OK)
    assert "technical limits" in detail and "get_file_text ✅2" in detail


# ── not_truncated (always on) ───────────────────────────────────────────────
#
# ⚠️ `has_answer` counts characters. On 2026-08-10 `video-unnamed-reference`
# returned a preamble ending "Here's a summary of it:" and nothing else — well
# over the 20-char floor, announcing content that never arrived, scored PASS.

_FOOTER = "\n\n---\n> _Tools used:_\n> - **qwen3.6:35b** — `list_my_files` ✅1\n"


def test_an_answer_ending_on_a_colon_is_truncated():
    answer = "The most recent recording is X. Here's a summary of it:" + _FOOTER
    assert er._looks_truncated(answer) is True


def test_a_complete_answer_is_not_flagged():
    answer = "The most recent recording is X. Let me know if you want more." + _FOOTER
    assert er._looks_truncated(answer) is False


def test_an_unclosed_code_fence_is_truncated():
    assert er._looks_truncated("Here you go:\n\n```python\ndef f():\n") is True
    assert er._looks_truncated("Here:\n\n```python\ndef f():\n    pass\n```\n") is False


def test_the_tools_footer_is_not_mistaken_for_the_end_of_the_answer():
    """⚠️ The footer lives INSIDE the answer body — it opens `\\n\\n---\\n>`,
    which is deliberately not the banner separator, so `_answer_body` keeps it.
    Any check reading the END of an answer must cut it first or it inspects a
    footer row instead of the last thing the model said."""
    answer = "Complete sentence." + _FOOTER

    assert er._prose_region(answer) == "Complete sentence."
    assert er._looks_truncated(answer) is False


def test_an_empty_body_is_left_to_has_answer():
    """Two checks reporting the same defect makes a run harder to read, not
    easier — emptiness is `has_answer`'s job."""
    assert er._looks_truncated("") is False


# ── expect_disclaims_absence ────────────────────────────────────────────────
#
# The cases whose whole point is a gap — no transcript, no artifacts, no such
# file, a topic the corpus does not cover. In every one the failure is to
# answer anyway, from a summary or a filename or world knowledge, and that
# failure reads as a perfectly good answer. Two of these cases shipped with no
# check for their own defining behaviour until 2026-08-11.


def test_admitting_a_missing_transcript_passes():
    answer = ("This file **does not have a transcript** — there is nothing "
              "available to quote." + _FOOTER)
    assert er._disclaims_absence(answer) is True


def test_admitting_empty_artifacts_passes():
    answer = ("**silent.mp4** has **no accessible content** — its processing "
              "completed with an empty artifacts list." + _FOOTER)
    assert er._disclaims_absence(answer) is True


def test_answering_anyway_fails():
    """The failure mode: describing the match from the visual summary when the
    question was what was SAID."""
    answer = ("The video shows Roger Gracie passing guard into side control "
              "and finishing with a choke." + _FOOTER)
    assert er._disclaims_absence(answer) is False


def test_disclaiming_survives_rewording():
    """Same lesson as the paging check — a family, not a phrase list. All four
    of these are real wordings from the 08-10 runs."""
    for answer in (
        "There is no transcript for this file.",
        "No summary, transcript, or visual data was generated from it.",
        "I could not find a file by that name.",
        "The uploaded videos do not cover the Sicilian Defence.",
    ):
        assert er._disclaims_absence(answer + _FOOTER) is True, answer


def test_the_wordings_that_false_failed_two_good_answers():
    """⚠️ Both verbatim from `audrey_video`, both textbook, both scored
    `disclaims:❌` — and had been since 07:06 the same morning.

    `don't have` was in the family and `don't see` was not, which is what a
    family assembled from whichever wordings turned up first looks like when
    it meets a new one. Same defect as the ASCII apostrophe: the check matched
    the text I had read rather than the behaviour I meant.
    """
    for answer in (
        "I don't see a file named teamOffsite2025.mp4 in your uploaded files.",
        "I’ve checked all your uploaded videos, and none of them appear to "
        "discuss the Sicilian Defence.",
        # Sibling forms the same gap would have swallowed.
        "The search returned no results.",
        "None of your videos mention it.",
        "I cannot see any file by that name.",
        "That video does not discuss the Sicilian Defence at all.",
        # Third widening, 2026-08-11 run 6 — same check, same shape, again.
        "There don’t appear to be any references to it in your files.",
        "Nothing specifically about the Sicilian Defence was found.",
        # Fourth, run 8. The family had `no results` and `no matches` but not
        # the most ordinary phrasing of the same fact.
        "No mentions of the **Sicilian Defence** were found in your videos.",
        "No references to it appear anywhere in your uploads.",
        # Fifth, run 9. Five widenings is the evidence for the standing rule
        # that a positive check must be measured every run, not assumed stable.
        "It looks like **teamOffsite2025.mp4** isn't in your uploaded files.",
        "That file is not in your uploads.",
        # Sixth, run 10 — and the archive scan found this one had also been
        # false-failing since the 07:06 run, so it cost more than the one run
        # that surfaced it. Same story as the second widening.
        "No videos in your library appear to contain any mention of the "
        "Sicilian Defence. I checked all of them and didn't find anything.",
    ):
        assert er._disclaims_absence(answer + _FOOTER) is True, answer


def test_the_structural_rewrite_that_was_rejected_by_measurement():
    """⚠️ Kept as a decision record, because the third patch to one regex is
    exactly when rewriting it looks obviously right.

    The candidate was a NEGATOR near a content noun within a window — properly
    shape-based, and the natural conclusion from "stop matching phrases". It
    scored 503 of 1377 archived sections against the phrase family's 318,
    firing on nearly every video answer including the ones that must FAIL. A
    check that passes everything is not a check.

    The lesson is not "shape-matching is wrong" — `continuation` and
    `not_misattributed` are both shape-based and both measured clean. It is
    that a POSITIVE check (the answer must say something) goes vacuous when
    widened, where a NEGATIVE one (the answer must not) goes noisy, which is
    the far louder failure. Measure before believing either.
    """
    good = "The video shows Roger Gracie passing guard and finishing with a choke."
    assert er._disclaims_absence(good + _FOOTER) is False


def test_typographic_apostrophes_do_not_defeat_the_checks():
    """⚠️ Models write `don’t`, not `don't`, and a regex spelled `don'?t`
    matches only the ASCII form.

    Cost a false FAIL on 2026-08-11: `video-topic-not-in-corpus` answered "I
    don’t have any references to the Sicilian Defence in your uploaded
    videos" — textbook — and scored `disclaims:❌`. A check that fails good
    answers gets deleted, so this is as damaging as one that misses bad ones.
    """
    assert er._disclaims_absence("I don’t have any references to that." + _FOOTER) is True
    assert er._disclaims_absence("There’s no transcript for it." + _FOOTER) is True

    # Same normalisation reaches the paging check's `can’t`.
    assert er._declines_without_offering(
        "I can’t provide the full transcript. Use other software.") is True
    assert er._declines_without_offering(
        "I can’t provide it in one go. Would you like me to continue?") is False


def test_markdown_emphasis_does_not_hide_a_phrase():
    """⚠️ The same defect as the apostrophe, found the same way and one run
    apart: which words a model happens to BOLD is not a property any check
    should depend on.

    Verbatim from run 13, `video-missing-transcript-artifact` [audrey_auto],
    scored `disclaims:❌`: "The file has **no audio transcript**". `has no`
    needs its two words adjacent, and two asterisks had been put between them.
    Every phrase family in the harness had the same hole.
    """
    assert er._disclaims_absence(
        "The file has **no audio transcript** — only a visual description "
        "and a one-paragraph summary." + _FOOTER) is True
    # The same hole in the two always-on checks, where it hides a real failure
    # rather than a real pass.
    assert er._corpus_fictions(
        "It is a blitz game with Carlsen playing **as Black**." + _FOOTER,
        "video")
    assert er._misattributes_to_user(
        "The video mentions **your** London System **course**." + _FOOTER) is True


def test_the_emphasis_strip_is_kept_away_from_two_checks_on_purpose():
    """⚠️ Both exclusions were measured over the archive, not assumed.

    `_FILENAME_EDGE` contains `*` **as a delimiter**: strip it and the
    backward walk from an extension runs on into the sentence, turning three
    honest answers into invented-filename hits on an always-on check.
    `_looks_truncated` fires on a trailing colon, which a heading like
    "**Summary:**" supplies the moment the asterisks come off.
    """
    honest = "In short: I don't have any information about what's in " \
             "*silent.mp4* because nothing could be read from it."
    assert er._invented_filenames(honest + _FOOTER, "video") == []
    assert er._looks_truncated("Here is the transcript.\n\n**Summary:**") is False


# ── not_misattributed (always on) ──────────────────────────────────────────
#
# Crediting the USER with a file's content. The content is RIGHT, which is what
# makes it invisible: the summary is accurate, every structural check is green,
# and the only thing wrong is who said it. Seen twice — 2026-08-09 and again on
# 2026-08-11 after a tool-description fix had been verified to close it.


def test_the_real_regression_is_caught():
    """All four sentences verbatim from `video-ambiguous-singular` at 07:16 on
    2026-08-11, about a chess video the model had just read."""
    for answer in (
        "You note that the two most common responses are ...d5 and ...Nf6.",
        "Regarding your specific setup, you advocate for playing **Bf4** early.",
        "However, you caution that this requires advanced understanding.",
        "You mention that playing Nf3 early might not be ideal.",
        # And the 2026-08-09 pair, which the same check would have caught.
        "In this context, you mention facing a \"notorious\" system.",
        "You comment that it hasn't been popular in competitive chess.",
    ):
        assert er._misattributes_to_user(answer + _FOOTER) is True, answer


def test_ordinary_second_person_is_left_alone():
    """Second person is not the defect and an answer avoiding it would be
    worse. Only the user as SUBJECT of a verb of authorship is."""
    for answer in (
        "You asked about the London System, so here is what the video says.",
        "The file you uploaded runs about 12 minutes.",
        "You may want to watch the second half for the Bf4 lines.",
        "You can see the move order in the transcript below.",
        "You should note that the video never covers the Sicilian.",
        "Your video recommends playing d4 first.",
        "If you want the full text, say so and I'll continue.",
    ):
        assert er._misattributes_to_user(answer + _FOOTER) is False, answer


def test_a_conditional_clause_is_not_an_attribution():
    """⚠️ The one false positive in the entire answers archive, from a deep
    answer about correlation — teaching a method, addressed to the reader. A
    subordinating conjunction ahead of the pronoun is what separates the two,
    and a check that fails good answers gets ignored."""
    assert er._misattributes_to_user(
        "When you observe that A and B move together, there are always at "
        "least three possibilities." + _FOOTER) is False


def test_it_reads_prose_not_the_footer():
    """Same trap as `not_truncated`: the tools footer lives inside the body."""
    answer = "The video recommends d4." + _FOOTER
    assert er._misattributes_to_user(answer) is False


def test_the_speakers_possessions_are_not_the_users():
    """⚠️ The half the verb pattern missed, verbatim from
    `video-control-unscoped-plural` [audrey_video] on 2026-08-11.

    Same failure, no verb of authorship in it. The courses and the site belong
    to whoever made the video; `you have` is far too ordinary a phrase to put
    in the verb list, so the NOUN is what identifies it.
    """
    assert er._misattributes_to_user(
        "Both videos mention that you have additional London System courses "
        "available on your website for those wanting to take their knowledge "
        "further." + _FOOTER) is True

    for answer in (
        "It closes by plugging your four-hour London System course.",
        "Your channel has grown a lot this year.",
        "That video points viewers at your Patreon.",
    ):
        assert er._misattributes_to_user(answer + _FOOTER) is True, answer


def test_what_the_user_really_does_own_is_left_alone():
    """The uploader owns the files; the speaker owns everything else. A check
    that fired on "your videos" would fire on nearly every video answer in the
    suite and be worth nothing."""
    for answer in (
        "Your videos cover the London System from two angles.",
        "None of your uploads mention the Sicilian Defence.",
        "The file you uploaded has no transcript artifact.",
        "I searched your files and found two chess videos.",
    ):
        assert er._misattributes_to_user(answer + _FOOTER) is False, answer


# ── no_fiction ──────────────────────────────────────────────────────────────
#
# ⚠️ The blind spot that moved FIVE times. Every guard against invention was a
# per-case `answer_not_contains`, and each run's fabrication landed on a case
# that happened not to carry one. Swept over every archived video answer when
# this was written: 15 fabrications across SIX of the twelve cases, only two of
# which had a blacklist. Per-corpus instead of per-case is the fix.
#
# Ground truth is the artifact summaries the model is given (upload page,
# 2026-08-11): the Gracie clip is visual-only — grappling, a pin, a scoreboard,
# IBJJF signage — with no result of any kind; Carlsen plays White and plays the
# London himself.


def test_the_inventions_that_scored_pass():
    """Verbatim from archived answers that every check passed."""
    for answer in (
        "Roger ultimately dominates the fight, securing a decisive victory "
        "via submission, highlighting his technical mastery.",
        "Roger Gracie won this particular match by **points (4-1)** in the "
        "absolute division.",
        "This video captures highlights from the 2009 Abu Dhabi World "
        "Professional Jiu-Jitsu Championship final.",
        "It's a 3-minute blitz game in which Magnus Carlsen plays against "
        "the London as Black.",
        "This video is a livestreamed blitz game in which Magnus Carlsen "
        "(playing Black, rated 3272) faces Evgenij Shuvalov.",
    ):
        assert er._corpus_fictions(answer + _FOOTER, "video"), answer


def test_an_abbreviation_is_not_the_end_of_a_sentence():
    """⚠️ `Jr.` is a full stop. Both the fiction spans and the negator window
    that guards them are bounded by `.`, so this sentence — a plain invention
    of a result and a division — fell in the gap and scored a clean PASS on
    2026-08-11 run 8."""
    answer = (
        "This was the heavyweight final at the 2009 IBJJF World Championships, "
        "where Roger Gracie defeated Rafael Lovato Jr. by points to claim his "
        "third world title." + _FOOTER
    )
    assert er._corpus_fictions(answer, "video")


def test_the_colour_inversion_told_from_the_other_end():
    """Same false fact, no colour word in it: Carlsen plays the London himself,
    as White. Matching only "Carlsen … as Black" missed this on run 8."""
    answer = (
        "It is a commentary by Magnus Carlsen on a game where his opponent "
        "used the London System against him." + _FOOTER
    )
    assert er._corpus_fictions(answer, "video")


def test_the_colour_inversion_told_a_third_way():
    """⚠️ Two patterns for one false fact, and run 13 found a third phrasing
    that neither held: Carlsen as the subject of playing AGAINST the London.
    Three hits across the archive, all three the real thing, two of which had
    scored a clean PASS while the check was believed to be closed.
    """
    for answer in (
        "A key detail: Carlsen is playing White against the London System, "
        "which his opponent, Candidate Master Evgenij Shuvalov, is employing "
        "as Black.",
        "This is a livestreamed 3-minute blitz game in which Carlsen (White, "
        "rated ~3272) plays against the London System, used by his opponent.",
    ):
        assert er._corpus_fictions(answer + _FOOTER, "video"), answer


def test_the_beat_the_london_pattern_that_was_rejected_by_measurement():
    """⚠️ The obvious widening from the above — anything about beating,
    fighting or countering the London — was measured at six archive hits, and
    the majority were correct. One was a verbatim quotation OF the transcript:
    "[00:10:48] When playing against the London as black, you have several
    options". The corpus's own words look like the fiction, which is the
    substring trap wearing a different coat. Carlsen must be the subject.
    """
    quoted = (
        "[00:10:48] When playing against the London as black, you have "
        "several options, and here is what I recommend." + _FOOTER
    )
    assert er._corpus_fictions(quoted, "video") == []


def test_one_creator_for_a_two_creator_set():
    """The corpus-shape fiction about AUTHORSHIP rather than titles. A Carlsen
    livestream and a Rozman lesson do not share a maker, so a singular creator
    spanning both is invented — run 13, verbatim, on a case whose whole subject
    is the plural "my videos"."""
    answer = (
        "**Course promotion:** The creator of these videos specifically "
        "mentions they have a dedicated, in-depth course on the London System."
    )
    assert er._corpus_fictions(answer + _FOOTER, "video")


def test_collapsing_the_two_london_videos_into_one():
    """A fiction about the SHAPE of the corpus, not its content — and the
    answer to `video-ambiguous-singular` that looks most like diligence.
    Verbatim from run 9, where it scored PASS: it named both files, so
    `names_files` was satisfied by the very sentence denying they are two."""
    for answer in (
        '"Magnus Carlsen Teaches How to Win with the London System.mp4" — '
        "same video, just named slightly differently in your uploads.",
        "It looks like both of your videos are about Magnus Carlsen playing "
        "the London System in a 3-minute blitz game.",
    ):
        assert er._corpus_fictions(answer + _FOOTER, "video"), answer


def test_the_rozman_title_pattern_that_was_rejected_by_measurement():
    """⚠️ The obvious way to catch the above — the Rozman title followed by
    "Carlsen" or "blitz" — is the substring trap `_names_all_files` exists to
    avoid. "How to Win with the London System" sits inside "Magnus Carlsen
    Teaches How to Win with the London System", so all nine of its hits across
    the archive were correct descriptions of the CARLSEN file."""
    correct = (
        'In "Magnus Carlsen Teaches How to Win with the London System" (7:37), '
        "this is a livestreamed blitz game." + _FOOTER
    )
    assert er._corpus_fictions(correct, "video") == []


def test_a_file_the_corpus_does_not_have():
    """Run 11: `video-topic-not-in-corpus` [audrey_video] PASSED while telling
    the user "You have two video files available" and naming two recordings
    that are not in the corpus — while every other answer in the same run
    listed the real ten."""
    answer = (
        "You have two video files available:\n"
        "1. **_20260811_164547.webm** (36 min 36 sec) — processing pending\n"
        "2. **_20260811_165259.webm** (21 min 2 sec) — processing pending"
    )
    assert er._invented_filenames(answer + _FOOTER, "video")
    assert er._corpus_fictions(answer + _FOOTER, "video")


def test_the_real_uploads_are_not_inventions():
    answer = (
        "Your files are:\n"
        "- How to WIN with the London System.mp4\n"
        "- `Ken McNabb_ How to Correctly Fit Your Saddle and Pad on Your "
        "Horse.mp4`\n"
        "- **Roger Gracie VS Rafael Lovato Jr _ World Championship 2009.mp4**\n"
        "- jasonRetirement.mp4\n- silent.mp4\n- p14.txt\n- audrey.png"
    )
    assert er._invented_filenames(answer + _FOOTER, "video") == []


def test_a_filename_from_the_question_is_the_model_quoting_it_back():
    """`video-unknown-filename` asks about a file that deliberately does not
    exist. Every correct answer repeats the name, and flagging that would fail
    the case for doing exactly what it is testing."""
    prompt = "What did teamOffsite2025.mp4 say about the roadmap?"
    answer = "I don't have a file named `teamOffsite2025.mp4` in your uploads."
    assert er._invented_filenames(answer + _FOOTER, "video", prompt) == []
    assert er._invented_filenames(answer + _FOOTER, "video") != []   # without it


def test_the_mutation_half_stays_out_of_reach_by_design():
    """⚠️ A decision record, not a wish. Run 10 invented two files by MUTATING
    a real name — inserting a word into the Gracie title, and taking a
    substring of it. Both score ~0.79 against the original, which is where a
    model that fumbled an en-dash or doubled a word also lands, so no threshold
    separates them. Measured and rejected; the loose end is deliberate.

    If this ever starts passing, check what the threshold was moved to and what
    it now false-fails.
    """
    invented = (
        "I do have other files relevant to this match: **`What Happened During "
        "The Roger Gracie VS Rafael Lovato Jr Match?_ World Championship "
        "2009.mp4`** and **`Rafael Lovato Jr _ World Championship 2009.mp4`**."
    )
    assert er._invented_filenames(invented + _FOOTER, "video") == []


def test_prose_that_merely_contains_an_extension_is_not_a_filename():
    """The extraction walks backwards from the extension to a delimiter.
    Undelimited prose produces a sentence fragment, not a name, and one such
    fragment was the only false positive in the whole archive sweep."""
    answer = ("To get the full transcript you would need to use video "
              "transcription software or a service capable of processing "
              "the entire recording.mp4")
    assert er._invented_filenames(answer + _FOOTER, "video") == []


def test_an_honest_answer_from_the_artifacts_is_clean():
    for answer in (
        "The footage shows two competitors in blue gis grappling on a mat, "
        "with one pinning the other while referees observe.",
        "Carlsen plays the London as White and wins after going three pawns up.",
        "The video identifies ...c5 as Black's most challenging reply.",
        "Victory in this system relies on controlling the centre.",
    ):
        assert er._corpus_fictions(answer + _FOOTER, "video") == [], answer


def test_saying_the_corpus_does_not_say_is_not_a_fiction():
    """⚠️ The behaviour this check exists to ENCOURAGE must not trip it. A
    negator anywhere in the sentence is enough — deliberately biased towards
    letting answers through, because one false fail would discredit it."""
    for answer in (
        "The artifacts do not say whether the match ended by submission.",
        "There is no record of who won or by what score.",
        "It isn't clear from the summary whether there was a choke.",
    ):
        assert er._corpus_fictions(answer + _FOOTER, "video") == [], answer


def test_transcript_timestamps_are_not_scorelines():
    """⚠️ A bare `\\d-\\d` scoreline pattern was measured and dropped: across the
    archive it read paging timestamps and chess notation as match results, on
    the paging and ambiguous-singular cases where nothing was invented."""
    for answer in (
        "Here is the rest: [00:06:47] squares in the centre you don't have to "
        "castle short.",
        "The game finished 1-0 after Black resigned.",
    ):
        assert er._corpus_fictions(answer + _FOOTER, "video") == [], answer


def test_a_suite_with_no_corpus_is_untouched():
    """The fictions are claims about specific files. Nothing outside that
    corpus should ever be measured against them."""
    answer = "Gracie won the match by submission." + _FOOTER
    assert er._corpus_fictions(answer, "") == []
    assert er._corpus_fictions(answer, "research") == []


def _canned_run(monkeypatch, answer: str, case: dict) -> dict:
    """Drive the real `run_case` over a fixed answer — no network."""
    monkeypatch.setattr(
        er, "_post_stream",
        lambda *a, **k: (answer, [], None, er.StreamTiming(0.1, 1.0)))
    return er.run_case("http://x", "k", case, "audrey_auto", 10.0).checks


def test_the_check_is_on_without_being_asked_for(monkeypatch):
    """⚠️ The point of making this one always-on. The 08-11 regression landed
    on a case that happened to carry `expect_names_files`; on any of the other
    eleven it would have scored a clean PASS. An opt-in check only ever covers
    the case you predicted, and this blind spot has already moved once."""
    bare = {"name": "anything", "prompt": "summarise the video"}

    clean = _canned_run(monkeypatch, "The video recommends d4 first.", bare)
    assert clean["not_misattributed"] is True

    bad = _canned_run(monkeypatch, "You advocate for playing Bf4 early.", bare)
    assert bad["not_misattributed"] is False


def test_a_case_can_opt_out(monkeypatch):
    """For a prompt that itself makes a claim the model may reflect back."""
    case = {"name": "x", "prompt": "I think d4 is best",
            "allow_user_attribution": True}
    checks = _canned_run(monkeypatch, "You argue that d4 is best.", case)
    assert checks["not_misattributed"] is None


def test_declaring_a_corpus_turns_the_fiction_check_on(monkeypatch):
    invented = "Gracie won the match by submission in the absolute division."

    bare = {"name": "x", "prompt": "what happens in the video"}
    assert _canned_run(monkeypatch, invented, bare)["no_fiction"] is None

    scoped = dict(bare, corpus="video")
    assert _canned_run(monkeypatch, invented, scoped)["no_fiction"] is False
    assert _canned_run(
        monkeypatch, "Two competitors grapple on a mat.", scoped
    )["no_fiction"] is True


def test_the_reason_reaches_the_report(monkeypatch):
    """A bare ❌ on a fabrication check is close to useless — the whole point is
    to say WHICH claim the corpus contradicts, so a human can confirm it."""
    monkeypatch.setattr(
        er, "_post_stream",
        lambda *a, **k: ("Gracie won the match by submission.", [], None,
                         er.StreamTiming(0.1, 1.0)))
    r = er.run_case("http://x", "k",
                    {"name": "x", "prompt": "p", "corpus": "video"},
                    "audrey_auto", 10.0)
    assert "submission" in r.fiction_detail


def test_the_suite_case_is_wired_up():
    """The helper is only worth having if a case actually opts in. Pins that
    `video-ambiguous-singular` — the case that regressed — carries it."""
    cases = json.loads(
        (Path(er.__file__).parent / "eval_prompts_video.json").read_text())
    by_name = {c["name"]: c for c in cases}

    assert by_name["video-ambiguous-singular"].get("expect_names_files")
    assert by_name["video-control-unscoped-plural"].get("expect_names_files")
    assert by_name["video-long-transcript-paging"].get("expect_continuation_offer")
    assert by_name["video-missing-transcript-artifact"].get("expect_disclaims_absence")
    assert by_name["video-empty-artifacts"].get("expect_disclaims_absence")
    assert by_name["video-fact-present-in-transcript"].get("answer_contains")

    # `_WIDE_FILES` above is only meaningful if it is the suite's real group.
    for name in ("video-ambiguous-singular", "video-control-unscoped-plural"):
        assert by_name[name]["expect_names_files"] == _WIDE_FILES, name

    # ⚠️ EVERY case, not some. A video case without `corpus` is a case the
    # fabrication check cannot see — which is precisely how the blind spot kept
    # moving. This assertion is what makes a new case covered by default.
    for c in cases:
        assert c.get("corpus") == "video", c["name"]


def test_every_video_case_is_checked_for_something_behavioural():
    """⚠️ The standing lesson of 2026-08-10: three behavioural cases in three
    runs turned out to be gated only by structural checks, so each reported
    PASS on its own defining failure. A case with no behavioural expectation
    proves the stack is alive and nothing else — which is fine, but must be a
    DECISION rather than an oversight, so new ones land here on purpose."""
    cases = json.loads(
        (Path(er.__file__).parent / "eval_prompts_video.json").read_text())
    behavioural = ("answer_contains", "answer_not_contains", "expect_names_files",
                   "expect_continuation_offer", "expect_disclaims_absence")
    structural_only = {
        c["name"] for c in cases if not any(c.get(k) for k in behavioural)
    }

    # Both are judgement cases whose quality genuinely needs a human read; see
    # each one's `_why`. Anything NEW appearing here is an oversight.
    assert structural_only == {"video-unnamed-reference",
                               "video-two-file-compare",
                               "video-control-named-scoped"}


# ── sweep expansion ─────────────────────────────────────────────────────────

def test_expand_sweep_crosses_and_groups_by_model():
    cases = [
        {"name": "a", "prompt": "pa", "code_test": "t"},
        {"name": "b", "prompt": "pb"},
    ]
    out = er._expand_sweep(cases, ["m1", "m2"])
    assert [(c["name"], c["model"]) for c in out] == [
        ("a [m1]", "m1"), ("b [m1]", "m1"),   # all of model 1 first (GPU-load
        ("a [m2]", "m2"), ("b [m2]", "m2"),   # friendly), then model 2
    ]
    # Original case fields survive the copy; originals aren't mutated.
    assert out[0]["code_test"] == "t"
    assert "model" not in cases[0]


# ── the local-model bake-off suite ──────────────────────────────────────────
#
# `eval_prompts_local_models.json` compares nemotron / muse-glimmer / qwen3.6
# head-to-head through `audrey_passthrough/`. Everything below guards the case
# FILE rather than the harness, because a bad case file fails silently: a
# mistyped key is not an error, it is a check that quietly never runs, and the
# sweep still prints a confident table.

_LOCAL_CASES = json.loads(
    (_SCRIPTS / "eval_prompts_local_models.json").read_text())

# Every key `run_case` actually reads. A case key outside this set does nothing.
_CASE_KEYS = {
    "name", "prompt", "_why", "model", "corpus",
    "expect_sources", "expect_route", "expect_banners",
    "answer_contains", "answer_not_contains", "expect_names_files",
    "expect_continuation_offer", "expect_disclaims_absence",
    "allow_user_attribution", "expect_code", "code_test", "code_timeout",
}


def test_no_case_key_is_silently_ignored():
    """⚠️ The bug class this whole suite is most exposed to. `answer_contain`
    instead of `answer_contains` raises nothing, disables the only objective
    check on that case, and reports PASS — so the bake-off would rank three
    models on a check that never ran."""
    for c in _LOCAL_CASES:
        assert not set(c) - _CASE_KEYS, (c.get("name"), set(c) - _CASE_KEYS)


def test_every_case_is_objectively_checked_or_says_why_not():
    """A case with no check is a case that always passes. Two here are
    deliberately eyes-only (`synth-conflicting-drafts` beyond both figures,
    `reasoning-race-order` beyond the three names) and both say so in `_why`
    with a ⚠️; anything else must carry a real check."""
    checked = {"answer_contains", "answer_not_contains", "code_test",
               "expect_disclaims_absence"}
    for c in _LOCAL_CASES:
        assert c.get("_why"), c["name"]
        if not (set(c) & checked):
            assert "⚠️" in c["_why"], f"{c['name']} has no check and no caveat"


def test_the_synthesis_cases_cannot_be_passed_from_priors():
    """The point of inventing `Kessler-7`, `kbuf` and `vault-sync`: a real
    system's numbers can be recalled, so `answer_contains` would measure
    training data instead of whether the note was read."""
    merge = next(c for c in _LOCAL_CASES if c["name"] == "synth-merge-three-drafts")
    for needle in merge["answer_contains"]:
        assert needle in merge["prompt"], needle


def test_no_fabrication_trap_fires_on_its_own_source_text():
    """⚠️ A banned phrase that appears in the PROMPT is one a faithful answer
    may legitimately echo. Two traps were cut for exactly this — "archived to"
    (the notes say events are "dropped rather than archived") and "Sure" (a
    substring of "ensure").

    Two cases are deliberate exceptions, and both invert the rule rather than
    bending it — echoing the prompt IS the failure they test:
      • `synth-compress-the-padding` — the banned phrases are the padding it
        was asked to strip.
      • `instruction-negative-constraint` — a prompt banning a word has to
        name it, and using it anyway is the whole failure.
    """
    inverted = {"synth-compress-the-padding", "instruction-negative-constraint"}
    for c in _LOCAL_CASES:
        if c["name"] in inverted:
            # Guard the exception itself: these must still ban something, or
            # the opt-out silently turns into no check at all.
            assert c.get("answer_not_contains"), c["name"]
            continue
        for banned in c.get("answer_not_contains", []):
            assert banned.lower() not in c["prompt"].lower(), (c["name"], banned)


def test_the_code_tests_pass_against_a_correct_implementation():
    """⚠️ An unsatisfiable `code_test` fails all three models identically and
    looks like a hard question. Both are pinned against reference solutions so
    a red `code_runs` is always the model, never the case."""
    impls = {
        "code-rle-roundtrip": (
            "import re\n"
            "def rle(s):\n"
            "    out, i = [], 0\n"
            "    while i < len(s):\n"
            "        j = i\n"
            "        while j < len(s) and s[j] == s[i]:\n"
            "            j += 1\n"
            "        out.append(f'{s[i]}{j - i}')\n"
            "        i = j\n"
            "    return ''.join(out)\n"
            "def unrle(s):\n"
            "    return ''.join(c * int(n) for c, n in re.findall(r'(\\D)(\\d+)', s))\n"
        ),
        "code-merge-intervals": (
            "def merge(intervals):\n"
            "    out = []\n"
            "    for a, b in sorted(intervals):\n"
            "        if out and a <= out[-1][1]:\n"
            "            out[-1] = (out[-1][0], max(out[-1][1], b))\n"
            "        else:\n"
            "            out.append((a, b))\n"
            "    return out\n"
        ),
    }
    for c in _LOCAL_CASES:
        if not c.get("code_test"):
            continue
        ok, detail = er._run_code_check(impls[c["name"]], c["code_test"], 15.0)
        assert ok, f"{c['name']}: {detail}"


def test_the_paired_grounding_cases_stay_paired():
    """`ground-fact-present` and `ground-fact-absent` share a passage and ask
    the same shape of question with opposite correct answers. Read together
    they separate reading from question-shaped pattern matching — which only
    works while both exist."""
    names = {c["name"] for c in _LOCAL_CASES}
    assert {"ground-fact-present", "ground-fact-absent"} <= names
    present = next(c for c in _LOCAL_CASES if c["name"] == "ground-fact-present")
    absent = next(c for c in _LOCAL_CASES if c["name"] == "ground-fact-absent")
    assert "2,140" in present["prompt"] and "2,140" in absent["prompt"]
    assert "p99" not in absent["prompt"].split("PASSAGE:")[1]


def test_expand_repeats_appends_whole_passes():
    """⚠️ Repeats are whole passes, not adjacent duplicates. The same prompt
    twice in a row to a warm model is the shape most likely to correlate the
    two samples, which defeats the point of sampling."""
    cases = [{"name": "a", "prompt": "p"}, {"name": "b", "prompt": "q"}]
    out = er._expand_repeats(cases, 3)
    assert [c["name"] for c in out] == ["a", "b", "a#2", "b#2", "a#3", "b#3"]


def test_expand_repeats_is_a_no_op_by_default():
    cases = [{"name": "a", "prompt": "p"}]
    assert er._expand_repeats(cases, 1) == cases
    assert er._expand_repeats(cases, 0) == cases


def test_a_repeat_marker_survives_the_sweep_suffix_strip():
    """`eval_compare` keys rows by the name with ` [<model>]` removed, so the
    `#N` has to sit inside that, or every repeat collapses onto one row."""
    out = er._expand_sweep(er._expand_repeats([{"name": "a", "prompt": "p"}], 2),
                           ["m1"])
    assert [c["name"] for c in out] == ["a [m1]", "a#2 [m1]"]
    assert [eval_compare._case_key(c["name"]) for c in out] == ["a", "a#2"]


def test_expand_sweep_name_falls_back_to_prompt():
    out = er._expand_sweep([{"prompt": "what is love"}], ["m1"])
    assert out[0]["name"] == "what is love [m1]"


# ── save_json ───────────────────────────────────────────────────────────────

def test_save_json_round_trips(tmp_path):
    # No source_stats on this result → the "sources" field serializes to null,
    # keeping the record shape stable (e.g. a code case).
    r = er.CaseResult(
        name="c1 [m1]", model="m1", ok=False,
        checks={"reachable": True, "code_runs": False, "banners": None},
        answer="x" * 40, route="unknown", ttft_s=1.5, total_s=12.0,
        code_detail="exit 1: AssertionError",
    )
    out = tmp_path / "results.json"
    er.save_json([r], out)
    records = json.loads(out.read_text())
    assert records == [{
        "name": "c1 [m1]", "model": "m1", "ok": False,
        "checks": {"reachable": True, "code_runs": False, "banners": None},
        "route": "unknown", "ttft_s": 1.5, "total_s": 12.0,
        "answer_len": 40, "banners": [], "error": "",
        "code_detail": "exit 1: AssertionError", "fiction_detail": "",
        "context_detail": "", "ungrounded_detail": "", "sources": None,
    }]


def test_save_json_includes_source_stats(tmp_path):
    # A research case carries grounding-quality numbers into the record so runs
    # can be compared on source quality, not just pass/fail + latency.
    r = er.CaseResult(
        name="attn", model="audrey_research", ok=True,
        checks={"reachable": True}, answer="ans", route="research",
        source_stats=er.SourceStats(
            total=5, official=1, academic=1, low_quality=0, other=3, quality="GOOD",
        ),
    )
    out = tmp_path / "results.json"
    er.save_json([r], out)
    rec = json.loads(out.read_text())[0]
    assert rec["sources"] == {
        "total": 5, "official": 1, "academic": 1,
        "low_quality": 0, "other": 3, "quality": "GOOD",
    }


# ── eval_compare.build_table ────────────────────────────────────────────────

def _rec(name, model, ok, total=10.0, **extra):
    rec = {"name": name, "model": model, "ok": ok, "checks": {"reachable": True},
           "route": "unknown", "ttft_s": 1.0, "total_s": total,
           "answer_len": 100, "banners": [], "error": "", "code_detail": ""}
    rec.update(extra)
    return rec


def test_build_table_matrix_strips_sweep_suffix():
    table = eval_compare.build_table([
        _rec("case-a [m1]", "m1", True),
        _rec("case-a [m2]", "m2", False,
             checks={"code_runs": False}, code_detail="exit 1: AssertionError"),
    ])
    # One row for case-a, both model columns, pass/fail marks with latency.
    assert "| case-a | ✅ 10s | ❌ 10s |" in table
    assert "`m1`" in table and "`m2`" in table
    # Failures section names the failing check and the code detail.
    assert "**case-a** on `m2` — code_runs; code: exit 1: AssertionError" in table


def test_build_table_summary_and_missing_cells():
    table = eval_compare.build_table([
        _rec("a", "m1", True, total=10.0),
        _rec("b", "m1", True, total=20.0),
        _rec("a", "m2", True, total=30.0),   # m2 never ran case b → "—" cell
    ])
    assert "| b | ✅ 20s | — |" in table
    assert "| `m1` | 2/2 | 1.0s | 15.0s | 100.0 |" in table
    assert "| `m2` | 1/1 | 1.0s | 30.0s | 100.0 |" in table
    assert "## Failures" not in table


# ── grounded: content claimed with no content tool behind it ────────────────
#
# Bodies below are the archived answers, footers included, not paraphrases.
# The one time a snippet here was written by hand rather than copied, it
# tripped a different check and the test asserted the wrong thing.

_FOOTER_CATALOGUE = (
    "\n\n---\n> _Tools used:_\n"
    "> - **qwen3.6:35b** — `list_my_files` ✅1\n"
)
_FOOTER_READ = (
    "\n\n---\n> _Tools used:_\n"
    "> - **qwen3.6:35b** — `list_my_files` ✅1, `get_file_text` ✅1\n"
)


def test_ungrounded_flags_a_summary_invented_from_the_listing():
    """2026-08-12-090801 video-unnamed-reference — passed all 17 checks."""
    answer = (
        'Your most recently uploaded file is **"Roger Gracie VS Rafael '
        'Lovato Jr _ World Championship 2009.mp4"**, uploaded on August 8th. '
        "Here is the summary of that recording:\n\n"
        "In this documentary-style video about a legendary Brazilian "
        "Jiu-Jitsu match at the 2009 World Championships, we learn about the "
        "intense rivalry between Roger Gracie and Rafael Lovato Jr."
    ) + _FOOTER_CATALOGUE
    assert er._ungrounded_content(answer) == "Here is the summary"


def test_ungrounded_ignores_the_same_claim_once_the_file_was_read():
    """2026-08-11-205245 — same phrasing, and legitimate: it read the file."""
    answer = (
        "Here's the summary of your most recent upload — **Roger Gracie VS "
        "Rafael Lovato Jr _ World Championship 2009.mp4**:\n\n"
        "The video shows highlights from a Brazilian Jiu-Jitsu World "
        "Championship match in 2009 between two black-belt competitors."
    ) + _FOOTER_READ
    assert er._ungrounded_content(answer) is None


def test_ungrounded_passes_an_offer_to_fetch():
    """2026-08-09-141807 — catalogue only, and does the right thing with it."""
    answer = (
        'The most recent file you uploaded is **"Roger Gracie VS Rafael '
        'Lovato Jr _ World Championship 2009.mp4"** (uploaded on August '
        "8th). I can pull its summary for you:\n\n"
        "- **Artifacts available:** Summary, Visual\n\n"
        "Would you like me to read it?"
    ) + _FOOTER_CATALOGUE
    assert er._ungrounded_content(answer) == ""


def test_ungrounded_passes_an_honest_report_of_absence():
    """2026-08-10-205508 video-empty-artifacts. "there's no summary" matched
    "here's ... summary" as a substring before the \\b anchor went in."""
    answer = (
        "**silent.mp4** has **no accessible content** — its processing "
        "completed with an empty artifacts list, meaning there's no summary, "
        "transcript, or visual description available for it. I can't say "
        "what is in it."
    ) + _FOOTER_CATALOGUE
    assert er._ungrounded_content(answer) == ""


def test_ungrounded_does_not_judge_a_research_answer():
    """The first draft named the content tools and forgot web_search, failing
    two research answers built on real searches. The gate is inverted now."""
    answer = (
        "You asked for a summary of the current state of async runtimes in "
        "Rust. Based on the available research, here is how the landscape "
        "breaks down."
        "\n\n---\n> _Tools used:_\n> - **glm-5.2:cloud** — `web_search` ✅14\n"
    )
    assert er._ungrounded_content(answer) is None


def test_ungrounded_abstains_without_a_footer():
    answer = (
        "Here is the summary of that recording:\n\nThe video captures a "
        "Brazilian Jiu-Jitsu match from 2009."
    )
    assert er._ungrounded_content(answer) is None


def test_ungrounded_catches_a_narrated_body_with_no_summary_word():
    answer = (
        'Your most recent upload is **"Roger Gracie VS Rafael Lovato Jr _ '
        'World Championship 2009.mp4"**. The video captures a pivotal '
        "submission match between two renowned competitors."
    ) + _FOOTER_CATALOGUE
    assert er._ungrounded_content(answer) == "The video captures"


def test_grounded_check_is_not_applicable_when_a_file_was_read():
    checks = {"grounded": er._ungrounded_content("x" + _FOOTER_READ)}
    assert checks["grounded"] is None


# ── disclaims: passage-QA phrasing (2026-08-12) ─────────────────────────────

def test_disclaims_accepts_the_passage_phrasing():
    """`ground-fact-absent` in the local-model bake-off. Correct answer, and
    the verb list — built on the video corpus — had no "provide"."""
    answer = (
        "The passage does not provide the p99 latency for the vault-sync "
        "worker. It only reports median latencies (31 ms for warm "
        "invocations and 2,140 ms for cold starts)."
    )
    assert er._disclaims_absence(answer)


def test_disclaims_accepts_the_other_passage_verbs():
    for verb in ("specify", "state", "report", "list", "indicate", "disclose"):
        assert er._disclaims_absence(f"The passage does not {verb} that."), verb
        assert er._disclaims_absence(f"The notes do not {verb} that."), verb


def test_disclaims_still_rejects_an_answer_that_just_answers():
    """The widening must not make the check vacuous — an answer that fills the
    gap instead of admitting it still fails."""
    answer = (
        "The p99 latency for the vault-sync worker is approximately 3,400 ms, "
        "which is typical for a cold-start-dominated workload."
    )
    assert not er._disclaims_absence(answer)


# ── bare code: obeying "Return only the code" (2026-08-12) ──────────────────

_MERGE = '''def merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged'''


def test_an_unfenced_answer_that_is_only_code_counts_as_code():
    """nemotron and muse on `code-merge-intervals`, 2026-08-12. The prompt ends
    "Return only the code"; both obeyed and the harness failed them for it."""
    assert er._bare_code(_MERGE) is not None
    assert er._has_tagged_code_block(_MERGE)


def test_bare_code_is_extracted_and_runnable():
    code = er._extract_code_block(_MERGE)
    assert code is not None
    ok, detail = er._run_code_check(
        code, "assert merge([(1,2),(2,3)]) == [(1,3)]\nassert merge([]) == []", 10)
    assert ok, detail


def test_prose_is_never_mistaken_for_code():
    answer = (
        "The video captures a Brazilian Jiu-Jitsu match from 2009. It shows "
        "two black belts grappling on a red and blue mat."
    )
    assert er._bare_code(answer) is None
    assert not er._has_tagged_code_block(answer)


def test_a_single_word_does_not_count_even_though_it_compiles():
    """⚠️ `compile()` alone is far too weak — a bare identifier is a valid
    expression statement. The `def ` guard is what makes this safe."""
    assert er._bare_code("Hello") is None


def test_prose_alongside_code_is_not_bare_code():
    """A sentence next to a function is a SyntaxError over the whole body, so
    it falls through to the fence rule rather than being rescued."""
    answer = "Here is the function you asked for:\n\n" + _MERGE
    assert er._bare_code(answer) is None


def test_an_untagged_fence_still_fails_rather_than_being_rescued():
    answer = "```\n" + _MERGE + "\n```"
    assert er._bare_code(answer) is None
    assert not er._has_tagged_code_block(answer)


def test_a_tagged_fence_still_wins():
    answer = "```python\n" + _MERGE + "\n```"
    assert er._has_tagged_code_block(answer)
    assert er._extract_code_block(answer).strip().startswith("def merge")


# ── no_reasoning_leak: deliberation that reached the user (2026-08-12) ──────

def test_a_leaked_think_tag_is_caught():
    """nemotron on `ground-fact-absent`, PASSTHROUGH_THINK=0 arm. Three
    paragraphs of visible working, a literal `</think>`, then the answer — and
    18 green checks, because the answer underneath was correct."""
    answer = (
        "No p99 mentioned. I will answer that it's not provided.\n\n"
        "Wait, is there any chance \"median of\" implies something about the "
        "distribution? No.\n\nI'll output that the information is not present "
        "in the passage.</think>The vault-sync worker's p99 latency is "
        "**not provided** in the passage."
    )
    assert er._leaks_reasoning(answer) == "</think>"


def test_an_opening_tag_is_caught_too():
    assert er._leaks_reasoning("<think>Let me work through this.") == "<think>"


def test_a_clean_answer_leaks_nothing():
    answer = (
        "The passage does not provide the vault-sync worker's p99 latency. "
        "It only reports median latencies (31 ms warm, 2,140 ms cold)."
    )
    assert er._leaks_reasoning(answer) == ""


def test_prose_about_thinking_is_not_a_leak():
    """⚠️ The check is the literal DELIMITER, not the word. An answer that
    discusses thinking, or a code answer mentioning a think() function, is
    ordinary content."""
    assert er._leaks_reasoning("I was thinking about the retention window.") == ""
    assert er._leaks_reasoning("Call think() before reading the buffer.") == ""
