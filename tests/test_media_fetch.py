"""The yt-dlp layer (Phase 41, steps 2 and 4).

`fetcher.py` is a claim loop and is tested next door; everything with a
judgement in it lives in `fetch.py` and is tested here. Three groups, and the
weight is deliberately on the last two:

  - **What runs.** The argv actually handed to yt-dlp, checked against a fake
    binary rather than asserted on a string — a flag that stops being passed is
    invisible to a mock and obvious to a subprocess.
  - **What the user is told.** `friendly_reason` is the deliverable of step 3.
    Private, deleted, members-only and region-blocked are the *common* cases,
    and the one thing that must never happen is all four collapsing into
    "download failed".
  - **What a caption track becomes.** `parse_vtt` is where step 4's quality
    win is won or lost. YouTube's auto-captions repeat themselves by design,
    and a literal parse produces a transcript that says everything twice —
    which does not merely read badly, it poisons retrieval.

The fake yt-dlp is a real executable script, so these exercise the actual
subprocess call, argument order and exit-code handling.
"""

from __future__ import annotations

import stat
import textwrap
from pathlib import Path

import pytest

from audrey.media.fetch import (
    SOURCE_AUTO_CAPTIONS,
    SOURCE_SUBTITLES,
    FetchFailedError,
    FetchRefusedError,
    UrlInfo,
    YtDlpMissingError,
    check_limits,
    download,
    friendly_reason,
    parse_progress_line,
    parse_vtt,
    probe_url,
)

URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
FILE_ID = "11111111-2222-3333-4444-555555555555"


def _fake_ytdlp(tmp_path: Path, body: str, name: str = "yt-dlp") -> str:
    """Write an executable stand-in for yt-dlp and return its path.

    A real script rather than a monkeypatched `subprocess.run`: the things most
    likely to break here are the argv — a flag dropped, `--` lost, the output
    template misspelled — and a mock asserts on the argv we *think* we built
    rather than the one a process is handed.
    """
    path = tmp_path / name
    path.write_text("#!/usr/bin/env python3\n" + textwrap.dedent(body))
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IRUSR)
    return str(path)


#: Answers `-J` with usable metadata and records the argv it was given.
_PROBE_OK = """
    import json, sys
    (sys.argv[0] + '.argv').__str__()
    with open(sys.argv[0] + '.argv', 'w') as f:
        f.write('\\n'.join(sys.argv[1:]))
    print(json.dumps({
        "title": "A Retirement Speech",
        "ext": "mp4",
        "duration": 565.0,
        "filesize_approx": 288 * 1024 * 1024,
        "subtitles": {"en": [{"ext": "vtt"}], "fr": [{"ext": "vtt"}]},
        "automatic_captions": {"en": [{"ext": "vtt"}]},
        "chapters": [{"start_time": 0.0, "title": "Intro"}],
    }))
"""


def _argv_of(binary: str) -> list[str]:
    return Path(binary + ".argv").read_text().splitlines()


class TestProbe:
    def test_metadata_comes_back_without_downloading_anything(self, tmp_path):
        binary = _fake_ytdlp(tmp_path, _PROBE_OK)
        info = probe_url(URL, binary=binary)
        assert info.title == "A Retirement Speech"
        assert info.duration_s == 565.0
        assert info.filesize_approx == 288 * 1024 * 1024
        assert info.subtitle_langs == ("en", "fr")
        assert info.auto_caption_langs == ("en",)

    def test_the_probe_never_downloads_and_never_expands_a_playlist(self, tmp_path):
        binary = _fake_ytdlp(tmp_path, _PROBE_OK)
        probe_url(URL, binary=binary)
        argv = _argv_of(binary)
        assert "--skip-download" in argv
        # A link with `&list=` is one video the user pasted, not 200 videos
        # they did not ask for — and 200 downloads is a quota problem wearing
        # an ingest costume.
        assert "--no-playlist" in argv

    def test_the_url_is_passed_after_a_double_dash(self, tmp_path):
        binary = _fake_ytdlp(tmp_path, _PROBE_OK)
        probe_url("https://www.youtube.com/watch?v=-x", binary=binary)
        argv = _argv_of(binary)
        # The URL is the one caller-supplied element. After `--` it cannot be
        # read as an option however it is spelled, which is the whole reason a
        # leading-dash URL is not a special case anywhere else in this file.
        assert argv[-2] == "--"
        assert argv[-1] == "https://www.youtube.com/watch?v=-x"

    def test_a_refusal_becomes_the_reason_not_the_exit_code(self, tmp_path):
        binary = _fake_ytdlp(tmp_path, """
            import sys
            print("ERROR: [youtube] xyz: Private video. Sign in if you've "
                  "been granted access to this video", file=sys.stderr)
            sys.exit(1)
        """)
        with pytest.raises(FetchRefusedError) as e:
            probe_url(URL, binary=binary)
        assert "private" in str(e.value)

    def test_unreadable_metadata_is_a_failure_not_a_refusal(self, tmp_path):
        # The distinction matters downstream: a refusal is a fact about the
        # video and is shown to the user, a failure is a fact about us.
        binary = _fake_ytdlp(tmp_path, "print('not json at all')")
        with pytest.raises(FetchFailedError):
            probe_url(URL, binary=binary)

    def test_a_playlist_response_takes_the_first_entry(self, tmp_path):
        binary = _fake_ytdlp(tmp_path, """
            import json
            print(json.dumps({"_type": "playlist", "entries": [
                {"title": "First", "ext": "mp4", "duration": 10.0},
                {"title": "Second", "ext": "mp4", "duration": 20.0},
            ]}))
        """)
        assert probe_url(URL, binary=binary).title == "First"

    def test_a_missing_binary_says_the_image_is_wrong(self, tmp_path, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _n: None)
        with pytest.raises(YtDlpMissingError) as e:
            probe_url(URL)
        # Not "this URL failed". It is every URL, and it is fixed by rebuilding.
        assert "image" in str(e.value)


class TestLimits:
    def test_a_long_video_is_refused_before_any_bytes_move(self):
        info = UrlInfo(title="t", ext="mp4", duration_s=8000, filesize_approx=0)
        with pytest.raises(FetchRefusedError) as e:
            check_limits(info, max_duration_s=7200)
        # The number in the message is the one the user can act on.
        assert "133 minutes" in str(e.value)

    def test_an_oversized_estimate_is_refused(self):
        info = UrlInfo(
            title="t", ext="mp4", duration_s=60,
            filesize_approx=3 * 1024**3,
        )
        with pytest.raises(FetchRefusedError) as e:
            check_limits(info, max_bytes=2 * 1024**3)
        assert "3072MB" in str(e.value)

    def test_a_live_stream_is_refused_with_the_reason_it_cannot_work(self):
        info = UrlInfo(title="t", ext="mp4", duration_s=0, filesize_approx=0, is_live=True)
        with pytest.raises(FetchRefusedError) as e:
            check_limits(info)
        # Not "too long" — it has no length. Saying so is what stops someone
        # raising the duration cap to fix a stream that will never end.
        assert "no end" in str(e.value)

    def test_zero_disables_a_cap_rather_than_refusing_everything(self):
        info = UrlInfo(title="t", ext="mp4", duration_s=99999, filesize_approx=10**12)
        check_limits(info, max_duration_s=0, max_bytes=0)

    def test_a_missing_size_estimate_is_not_treated_as_zero_bytes(self):
        # Plenty of sites report no `filesize_approx`. That must mean "unknown"
        # and defer to `--max-filesize` on the download, not "it fits".
        info = UrlInfo(title="t", ext="mp4", duration_s=60, filesize_approx=0)
        check_limits(info, max_bytes=1024)


class TestFriendlyReason:
    @pytest.mark.parametrize(("stderr", "expected"), [
        ("ERROR: [youtube] x: Private video. Sign in if you've been granted access",
         "private"),
        ("ERROR: [youtube] x: Video unavailable", "unavailable"),
        ("ERROR: [youtube] x: This video is available to this channel's members",
         "members only"),
        ("ERROR: [youtube] x: Sign in to confirm your age", "age-restricted"),
        ("ERROR: [youtube] x: The uploader has not made this video available in "
         "your country", "region"),
        ("ERROR: [youtube] x: This live event will begin in 3 hours", "live stream"),
        ("ERROR: Unable to download webpage: HTTP Error 429: Too Many Requests",
         "rate-limiting"),
        ("ERROR: Unsupported URL: https://example.test/thing", "does not know how"),
    ])
    def test_the_common_failures_read_as_sentences(self, stderr, expected):
        assert expected in friendly_reason(stderr)

    def test_a_stale_downloader_is_not_reported_as_a_broken_video(self):
        # Observed on the first real fetch this feature ever did. YouTube's
        # wording is about the video; the cause is entirely ours, and the
        # difference decides whether someone goes and checks the link or
        # rebuilds the image. The one case where passing yt-dlp's own text
        # through would be actively misleading rather than merely terse.
        reason = friendly_reason(
            "ERROR: [youtube] ebfzL_GwiIE: The following content is not "
            "available on this app. Watch on the latest version of YouTube.",
        )
        assert "out of date" in reason
        assert "Nothing is wrong with this video" in reason

    def test_four_different_failures_do_not_produce_one_message(self):
        # The whole point. A generic "download failed" for all of these is the
        # message that generates the support question the field exists to
        # prevent, and it is what this function is here instead of.
        messages = {
            friendly_reason("ERROR: Private video"),
            friendly_reason("ERROR: Video unavailable"),
            friendly_reason("ERROR: Sign in to confirm your age"),
            friendly_reason("ERROR: not available in your country"),
        }
        assert len(messages) == 4

    def test_an_unmapped_error_is_passed_through_rather_than_replaced(self):
        reason = friendly_reason(
            "WARNING: something cosmetic\n"
            "ERROR: [somesite] the codec negotiation went sideways",
        )
        # The failure modes are not a closed set. A new one must reach the user
        # as whatever yt-dlp said, not as a message that tells them nothing.
        assert reason == "[somesite] the codec negotiation went sideways"

    def test_the_last_error_line_wins_over_earlier_warnings(self):
        assert friendly_reason(
            "ERROR: first thing\nERROR: the actual reason",
        ) == "the actual reason"

    def test_silence_is_reported_as_silence(self):
        assert "said nothing" in friendly_reason("")


class TestDownload:
    #: Writes a file at the `-o` template with the extension substituted, and
    #: prints its path — which is what `--print after_move:filepath` does.
    BODY = """
        import sys
        from pathlib import Path
        argv = sys.argv[1:]
        with open(sys.argv[0] + '.argv', 'w') as f:
            f.write('\\n'.join(argv))
        out = Path(argv[argv.index('-o') + 1].replace('%(ext)s', 'mp4'))
        out.write_bytes(b'video bytes')
        print(out)
    """

    def test_it_lands_in_staging_under_the_file_id(self, tmp_path):
        binary = _fake_ytdlp(tmp_path, self.BODY)
        stage = tmp_path / "staging"
        got = download(URL, stage, FILE_ID, timeout_s=30, binary=binary)
        # The naming contract with `fetch/{id}/result`, which rebuilds this
        # path from `file_id` plus the reported extension.
        assert got.path == stage / f"{FILE_ID}.mp4"
        assert got.path.read_bytes() == b"video bytes"

    def test_the_output_path_is_read_from_yt_dlp_not_guessed(self, tmp_path):
        binary = _fake_ytdlp(tmp_path, """
            import sys
            from pathlib import Path
            argv = sys.argv[1:]
            out = Path(argv[argv.index('-o') + 1].replace('%(ext)s', 'mkv'))
            out.write_bytes(b'x')
            print(out)
        """)
        got = download(URL, tmp_path / "staging", FILE_ID, timeout_s=30, binary=binary)
        # The container is yt-dlp's to choose. Deriving it from the metadata
        # `ext` instead would be a guess that is wrong exactly when the format
        # selector fell through — and a wrong extension is a path to nothing.
        assert got.path.suffix == ".mkv"

    def test_the_size_cap_is_passed_to_the_downloader(self, tmp_path):
        binary = _fake_ytdlp(tmp_path, self.BODY)
        download(
            URL, tmp_path / "staging", FILE_ID,
            timeout_s=30, max_bytes=1234, binary=binary,
        )
        argv = _argv_of(binary)
        # The metadata estimate is advisory and often absent; this is the hard
        # stop that a lying one cannot get past.
        assert argv[argv.index("--max-filesize") + 1] == "1234"

    def test_the_output_is_remuxed_to_mp4(self, tmp_path):
        binary = _fake_ytdlp(tmp_path, self.BODY)
        download(URL, tmp_path / "staging", FILE_ID, timeout_s=30, binary=binary)
        argv = _argv_of(binary)
        # ALLOWED_VIDEO_MIMES is exactly {"video/mp4"}, so a webm would be
        # refused by the same libmagic gate that stops an HTML error page —
        # correctly, but with a message that reads as "the download broke".
        assert argv[argv.index("--remux-video") + 1] == "mp4"
        assert argv[argv.index("--merge-output-format") + 1] == "mp4"

    def test_a_stale_part_file_is_never_resumed(self, tmp_path):
        binary = _fake_ytdlp(tmp_path, self.BODY)
        download(URL, tmp_path / "staging", FILE_ID, timeout_s=30, binary=binary)
        # A `.part` left by a swept attempt belongs to a download nobody is
        # holding any more. Resuming it appends new bytes to old ones.
        assert "--no-continue" in _argv_of(binary)

    def test_success_with_no_file_written_is_a_failure_not_a_success(self, tmp_path):
        binary = _fake_ytdlp(tmp_path, "pass")
        with pytest.raises(FetchFailedError):
            download(URL, tmp_path / "staging", FILE_ID, timeout_s=30, binary=binary)

    def test_the_size_cap_tripping_reads_as_a_size_refusal(self, tmp_path):
        # `--max-filesize` aborts with returncode 0 and writes nothing, which
        # is the one success-shaped failure this call has. Reported as "too
        # large" rather than as the downloader misbehaving.
        binary = _fake_ytdlp(tmp_path, """
            import sys
            print("File is larger than max-filesize (99 > 10)", file=sys.stderr)
        """)
        with pytest.raises(FetchRefusedError) as e:
            download(
                URL, tmp_path / "staging", FILE_ID,
                timeout_s=30, max_bytes=10 * 1024 * 1024, binary=binary,
            )
        assert "larger than" in str(e.value)

    def test_a_timeout_says_so_rather_than_hanging_the_lease(self, tmp_path):
        binary = _fake_ytdlp(tmp_path, "import time; time.sleep(10)")
        with pytest.raises(FetchFailedError) as e:
            download(URL, tmp_path / "staging", FILE_ID, timeout_s=0.5, binary=binary)
        assert "did not finish" in str(e.value)


class TestCaptionChoice:
    def test_human_subtitles_win_over_auto_captions(self):
        info = UrlInfo(
            title="t", ext="mp4", duration_s=1, filesize_approx=0,
            subtitle_langs=("en",), auto_caption_langs=("en",),
        )
        # Manual subs are human-authored and routinely better than whisper.
        # Auto-captions are neither, but still beat 74 seconds of GPU.
        assert info.caption_choice() == SOURCE_SUBTITLES

    def test_auto_captions_are_taken_when_there_are_no_real_ones(self):
        info = UrlInfo(
            title="t", ext="mp4", duration_s=1, filesize_approx=0,
            auto_caption_langs=("en",),
        )
        assert info.caption_choice() == SOURCE_AUTO_CAPTIONS

    def test_a_regional_english_tag_still_counts_as_english(self):
        info = UrlInfo(
            title="t", ext="mp4", duration_s=1, filesize_approx=0,
            subtitle_langs=("en-GB",),
        )
        assert info.caption_choice() == SOURCE_SUBTITLES

    def test_another_language_is_not_english(self):
        # The permissive failure is the bad one: claiming English subtitles
        # exist on a video that has only Portuguese ones, then ingesting them
        # as if they were what was asked for.
        info = UrlInfo(
            title="t", ext="mp4", duration_s=1, filesize_approx=0,
            subtitle_langs=("pt", "pt-BR", "engineering"),
        )
        assert info.caption_choice() == ""

    def test_no_captions_at_all_means_whisper(self):
        info = UrlInfo(title="t", ext="mp4", duration_s=1, filesize_approx=0)
        assert info.caption_choice() == ""


class TestParseVtt:
    SIMPLE = """WEBVTT

00:00:01.000 --> 00:00:04.000
Hello and welcome.

00:00:12.000 --> 00:00:15.000
Today we are talking about retirement.
"""

    def test_cues_become_segments(self):
        segments = parse_vtt(self.SIMPLE)
        assert len(segments) == 2
        assert segments[0]["t_start"] == 1.0
        assert segments[0]["t_end"] == 4.0
        assert segments[0]["text"] == "Hello and welcome."

    def test_the_header_is_not_a_segment(self):
        assert all("WEBVTT" not in s["text"] for s in parse_vtt(self.SIMPLE))

    def test_an_hour_long_timestamp_is_read_as_an_hour(self):
        segments = parse_vtt("""WEBVTT

01:02:03.500 --> 01:02:06.000
Late in the video.
""")
        assert segments[0]["t_start"] == pytest.approx(3723.5)

    def test_inline_word_timings_are_stripped(self):
        segments = parse_vtt("""WEBVTT

00:00:01.000 --> 00:00:04.000
hello<00:00:01.500><c> there</c><00:00:02.000><c> friend</c>
""")
        # The `<c>` spans are how auto-captions animate word by word. Left in,
        # every chunk of the transcript is half markup — and the embedding is
        # of the markup too.
        assert segments[0]["text"] == "hello there friend"

    def test_the_rolling_repetition_of_auto_captions_is_removed(self):
        # This is the shape YouTube actually serves: each cue repeats the tail
        # of the last so the words appear to accumulate on screen.
        segments = parse_vtt("""WEBVTT

00:00:01.000 --> 00:00:03.000
so today we are

00:00:03.000 --> 00:00:03.010
so today we are
going to talk about

00:00:03.010 --> 00:00:06.000
going to talk about
what retirement means
""", min_segment_s=0)
        joined = " ".join(s["text"] for s in segments)
        # Parsed literally this says "so today we are" twice and "going to
        # talk about" twice — which does not merely read badly. A chunk of
        # triplicated text matches a query about that phrasing far more
        # strongly than the sentence deserves.
        assert joined.count("so today we are") == 1
        assert joined.count("going to talk about") == 1
        assert "what retirement means" in joined

    def test_short_cues_are_merged_into_whisper_shaped_segments(self):
        segments = parse_vtt("""WEBVTT

00:00:00.000 --> 00:00:01.000
one

00:00:01.000 --> 00:00:02.000
two

00:00:02.000 --> 00:00:03.000
three

00:00:30.000 --> 00:00:31.000
much later
""", min_segment_s=5.0)
        # Everything downstream — chunking, the [HH:MM:SS] sidecar, the frame
        # description context — was built against whisper's sentence-shaped
        # segments. A per-second caption cue is not that.
        assert len(segments) == 2
        assert segments[0]["text"] == "one two three"
        assert segments[0]["t_end"] == 3.0
        assert segments[1]["text"] == "much later"

    def test_html_entities_are_decoded(self):
        segments = parse_vtt("""WEBVTT

00:00:01.000 --> 00:00:02.000
rock &amp; roll
""")
        assert segments[0]["text"] == "rock & roll"

    def test_comma_decimals_are_accepted(self):
        # SRT-style separators turn up in .vtt files in the wild.
        segments = parse_vtt("""WEBVTT

00:00:01,500 --> 00:00:02,000
comma timing
""")
        assert segments[0]["t_start"] == pytest.approx(1.5)

    def test_cue_identifiers_are_not_mistaken_for_speech(self):
        segments = parse_vtt("""WEBVTT

intro-cue-1
00:00:01.000 --> 00:00:02.000
The actual words.
""")
        assert segments[0]["text"] == "The actual words."

    def test_junk_parses_to_nothing_rather_than_raising(self):
        # The caller treats an empty result as "let whisper do it", which is
        # the right answer to a caption file we cannot read. Raising here
        # would fail a download that succeeded.
        assert parse_vtt("") == []
        assert parse_vtt("not a vtt file at all") == []


class TestCaptionsOnDisk:
    """The bridge between `download` and `parse_vtt`: what yt-dlp leaves behind."""

    BODY = """
        import sys
        from pathlib import Path
        argv = sys.argv[1:]
        with open(sys.argv[0] + '.argv', 'w') as f:
            f.write('\\n'.join(argv))
        template = argv[argv.index('-o') + 1]
        out = Path(template.replace('%(ext)s', 'mp4'))
        out.write_bytes(b'video bytes')
        Path(template.replace('.%(ext)s', '.en.vtt')).write_text(
            "WEBVTT\\n\\n00:00:01.000 --> 00:00:04.000\\nSpoken words.\\n")
        print(out)
    """

    def test_a_caption_track_is_parsed_and_attributed(self, tmp_path):
        binary = _fake_ytdlp(tmp_path, self.BODY)
        got = download(
            URL, tmp_path / "staging", FILE_ID,
            timeout_s=30, caption_source=SOURCE_SUBTITLES, binary=binary,
        )
        assert got.transcript_source == SOURCE_SUBTITLES
        assert got.segments == [
            {"t_start": 1.0, "t_end": 4.0, "text": "Spoken words."},
        ]

    def test_the_right_flag_is_used_for_each_kind(self, tmp_path):
        binary = _fake_ytdlp(tmp_path, self.BODY)
        download(
            URL, tmp_path / "staging", FILE_ID,
            timeout_s=30, caption_source=SOURCE_AUTO_CAPTIONS, binary=binary,
        )
        argv = _argv_of(binary)
        # `--write-subs` and `--write-auto-subs` produce files with identical
        # names, so which one was asked for is the only thing that knows which
        # one arrived. Asking for both would make the attribution a guess.
        assert "--write-auto-subs" in argv
        assert "--write-subs" not in argv

    def test_no_captions_are_requested_when_there_are_none(self, tmp_path):
        binary = _fake_ytdlp(tmp_path, self.BODY)
        got = download(URL, tmp_path / "staging", FILE_ID, timeout_s=30, binary=binary)
        argv = _argv_of(binary)
        assert "--write-subs" not in argv and "--write-auto-subs" not in argv
        # The file the fake writes anyway is ignored: nothing was asked for,
        # so nothing is attributed.
        assert got.segments == []
        assert got.transcript_source == ""

    def test_a_caption_file_that_never_arrived_is_not_an_error(self, tmp_path):
        binary = _fake_ytdlp(tmp_path, TestDownload.BODY)
        got = download(
            URL, tmp_path / "staging", FILE_ID,
            timeout_s=30, caption_source=SOURCE_SUBTITLES, binary=binary,
        )
        # The video downloaded fine. Whisper does the transcript, which is what
        # would have happened without step 4 at all.
        assert got.path.exists()
        assert got.segments == []
        assert got.transcript_source == ""


def test_staging_holds_everything_yt_dlp_writes(tmp_path):
    """Nothing lands outside the one directory the fetcher may write to."""
    binary = _fake_ytdlp(tmp_path, TestCaptionsOnDisk.BODY)
    stage = tmp_path / "staging"
    before = {p.name for p in tmp_path.iterdir()}
    download(
        URL, stage, FILE_ID,
        timeout_s=30, caption_source=SOURCE_SUBTITLES, binary=binary,
    )
    # The container has write access to staging and to nothing else; a path
    # escaping it would fail on the box rather than here, so the check that
    # earns its keep is that every artifact is named under the file_id.
    assert all(p.name.startswith(FILE_ID) for p in stage.iterdir())
    # `yt-dlp.argv` is the fake recording its own arguments, not something the
    # code under test wrote.
    assert {p.name for p in tmp_path.iterdir()} - before == {"staging", "yt-dlp.argv"}


class TestProgressLine:
    """Parsing what `--progress-template` emits.

    Given a name and a test of its own because it is the one part of the
    streaming path with a decision in it: a yt-dlp upgrade that changes how a
    field renders should fail here, loudly, rather than quietly produce a
    download that reports 0 bytes for its entire run.
    """

    def test_a_normal_line_yields_bytes_and_total(self):
        assert parse_progress_line("@AUDREYP 1048576 10485760 NA") == (1048576, 10485760)

    def test_an_unknown_total_falls_back_to_the_estimate(self):
        # Sites that will not commit to a size still usually offer a guess. A
        # roughly-right denominator beats none: a counter with no total can
        # only climb, which does not answer "how much longer".
        assert parse_progress_line("@AUDREYP 500 NA 9000") == (500, 9000)

    def test_no_total_at_all_is_reported_as_unknown(self):
        assert parse_progress_line("@AUDREYP 500 NA NA") == (500, None)

    def test_a_float_byte_count_is_accepted(self):
        # `total_bytes_estimate` is a float in yt-dlp's own data.
        assert parse_progress_line("@AUDREYP 500 NA 9000.5") == (500, 9000)

    def test_the_filepath_line_is_not_progress(self):
        # Both share stdout. Mistaking one for the other loses the download's
        # location, which is the only thing `download` has to return.
        assert parse_progress_line("/data/uploads/.staging/abc.mp4") is None

    def test_a_malformed_progress_line_is_ignored_rather_than_raising(self):
        assert parse_progress_line("@AUDREYP") is None
        assert parse_progress_line("@AUDREYP not-a-number 5") is None


class TestProgressStreaming:
    """Progress arrives *during* the download, not after it."""

    BODY = """
        import sys, time
        from pathlib import Path
        argv = sys.argv[1:]
        with open(sys.argv[0] + '.argv', 'w') as f:
            f.write('\\n'.join(argv))
        for done in (25, 50, 100):
            print(f"@AUDREYP {done} 100 NA", flush=True)
        out = Path(argv[argv.index('-o') + 1].replace('%(ext)s', 'mp4'))
        out.write_bytes(b'video bytes')
        print(out)
    """

    def test_every_update_reaches_the_callback(self, tmp_path):
        binary = _fake_ytdlp(tmp_path, self.BODY)
        seen = []
        got = download(
            URL, tmp_path / "staging", FILE_ID, timeout_s=30, binary=binary,
            on_progress=lambda d, t: seen.append((d, t)),
        )
        assert seen == [(25, 100), (50, 100), (100, 100)]
        # And the download still returns what it always did — the progress
        # lines must not be mistaken for the printed filepath.
        assert got.path.name == f"{FILE_ID}.mp4"

    def test_the_template_and_newline_are_both_passed(self, tmp_path):
        binary = _fake_ytdlp(tmp_path, self.BODY)
        download(
            URL, tmp_path / "staging", FILE_ID, timeout_s=30, binary=binary,
            on_progress=lambda d, t: None,
        )
        argv = _argv_of(binary)
        # Without `--newline` yt-dlp redraws one line with a carriage return,
        # so `for line in stdout` yields nothing until the process exits — the
        # progress would all arrive at once, after the download it describes.
        assert "--newline" in argv
        assert "--no-progress" not in argv
        assert "%(progress.downloaded_bytes)s" in argv[argv.index("--progress-template") + 1]

    def test_progress_is_suppressed_when_nobody_is_listening(self, tmp_path):
        binary = _fake_ytdlp(tmp_path, TestDownload.BODY)
        download(URL, tmp_path / "staging", FILE_ID, timeout_s=30, binary=binary)
        assert "--no-progress" in _argv_of(binary)

    def test_a_throwing_callback_does_not_lose_the_download(self, tmp_path):
        def boom(_d, _t):
            raise RuntimeError("the reporting side fell over")

        binary = _fake_ytdlp(tmp_path, self.BODY)
        got = download(
            URL, tmp_path / "staging", FILE_ID, timeout_s=30, binary=binary,
            on_progress=boom,
        )
        # Progress is a courtesy; the bytes are the job. Failing the download
        # because nobody could be told about it would be the tail wagging.
        assert got.path.exists()

    def test_a_hung_downloader_is_killed_at_the_deadline(self, tmp_path):
        # The reason the timeout is a watchdog rather than a deadline checked
        # between lines: `readline` blocks, so a downloader that stalls with no
        # output would sit past any between-lines check forever.
        binary = _fake_ytdlp(tmp_path, "import time; time.sleep(30)")
        with pytest.raises(FetchFailedError) as e:
            download(
                URL, tmp_path / "staging", FILE_ID, timeout_s=0.5, binary=binary,
                on_progress=lambda d, t: None,
            )
        assert "did not finish" in str(e.value)

    def test_a_large_stderr_does_not_deadlock_the_stream(self, tmp_path):
        # The classic pipe deadlock: reading stdout line by line while stderr
        # fills its own 64 KB buffer. stderr goes to a temp file for exactly
        # this, and 200 KB of warnings is a real yt-dlp run on a bad day.
        binary = _fake_ytdlp(tmp_path, """
            import sys
            sys.stderr.write("WARNING: something verbose\\n" * 8000)
            print("ERROR: and then it failed", file=sys.stderr)
            sys.exit(1)
        """)
        with pytest.raises(FetchRefusedError) as e:
            download(
                URL, tmp_path / "staging", FILE_ID, timeout_s=20, binary=binary,
                on_progress=lambda d, t: None,
            )
        # It got the *last* error line, from the far end of 200 KB.
        assert "and then it failed" in str(e.value)
