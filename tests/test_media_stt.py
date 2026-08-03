"""Tests for the whisper driver (Phase 35).

faster-whisper is not a dependency of this repo — it lives only in the
media-worker image, because installing it here would put CTranslate2 and a
half-gigabyte of weights into every developer's venv for code that runs in a
container. So these tests drive a **fake engine** with the same surface.

That is a real limit and worth naming: nothing here proves whisper transcribes
correctly. What it proves is everything wrapped *around* whisper — the budget
that stops a transcription outliving its lease, the repetition collapse, the
refusal to return a partial transcript — which is where the logic that can be
wrong actually lives. Transcription quality is verified on the box against a
video whose contents are known.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from audrey.media.stt import (
    Segment,
    TranscriptionFailedError,
    WhisperUnavailableError,
    collapse_repeats,
    load_model,
    transcribe,
)


class _FakeSegment:
    def __init__(self, start: float, end: float, text: str):
        self.start, self.end, self.text = start, end, text


class _FakeInfo:
    language = "en"
    language_probability = 0.99


class _FakeEngine:
    """Same call surface as `faster_whisper.WhisperModel`."""

    def __init__(self, segments, *, delay: float = 0.0, raise_on_open=None,
                 raise_at: int | None = None):
        self._segments = segments
        self._delay = delay
        self._raise_on_open = raise_on_open
        self._raise_at = raise_at
        self.kwargs: dict = {}

    def transcribe(self, path, **kwargs):
        self.kwargs = kwargs
        if self._raise_on_open:
            raise self._raise_on_open

        def gen():
            for i, s in enumerate(self._segments):
                if self._raise_at is not None and i == self._raise_at:
                    raise RuntimeError("decoder exploded")
                if self._delay:
                    time.sleep(self._delay)
                yield s

        return gen(), _FakeInfo()


@pytest.fixture
def wav(tmp_path: Path) -> Path:
    p = tmp_path / "audio.wav"
    p.write_bytes(b"RIFF....WAVE")
    return p


class TestTranscribe:
    def test_it_maps_segments_and_strips_whitespace(self, wav: Path):
        engine = _FakeEngine([
            _FakeSegment(0.0, 2.5, "  Hello there.  "),
            _FakeSegment(2.5, 4.0, "General Kenobi."),
        ])
        out = transcribe(wav, model=engine)

        assert [s.text for s in out] == ["Hello there.", "General Kenobi."]
        assert out[0].t_start == 0.0
        assert out[0].t_end == 2.5

    def test_empty_segments_are_dropped(self, wav: Path):
        """VAD gaps and pure-silence windows come back as blank text. Ingesting
        them would create chunks with a timestamp and nothing else."""
        engine = _FakeEngine([
            _FakeSegment(0.0, 1.0, "real speech"),
            _FakeSegment(1.0, 2.0, "   "),
            _FakeSegment(2.0, 3.0, ""),
        ])
        assert len(transcribe(wav, model=engine)) == 1

    def test_the_vad_filter_is_on(self, wav: Path):
        """Whisper's signature failure is inventing speech in silence — a
        musical intro reliably yields 'Thanks for watching!'. The VAD gates
        audio before the model sees it."""
        engine = _FakeEngine([])
        transcribe(wav, model=engine)
        assert engine.kwargs["vad_filter"] is True

    def test_conditioning_on_previous_text_is_off(self, wav: Path):
        """Conditioning each window on its own previous output is how one
        mis-decode becomes a repetition loop that eats the rest of the file."""
        engine = _FakeEngine([])
        transcribe(wav, model=engine)
        assert engine.kwargs["condition_on_previous_text"] is False

    def test_a_silent_file_yields_no_segments_and_no_error(self, wav: Path):
        assert transcribe(wav, model=_FakeEngine([])) == []


class TestBudget:
    def test_it_aborts_once_the_budget_is_spent(self, wav: Path):
        """A transcription that outlives `lease_minutes` gets swept and
        re-claimed while still running — the same file transcribed twice,
        concurrently, until its attempts run out. Failing at the budget turns
        that into one honest `failed` row."""
        engine = _FakeEngine(
            [_FakeSegment(i, i + 1, f"line {i}") for i in range(10)], delay=0.05,
        )
        with pytest.raises(TranscriptionFailedError, match="budget"):
            transcribe(wav, model=engine, budget_s=0.1)

    def test_the_budget_message_says_how_to_fix_it(self, wav: Path):
        engine = _FakeEngine(
            [_FakeSegment(i, i + 1, f"line {i}") for i in range(10)], delay=0.05,
        )
        with pytest.raises(TranscriptionFailedError) as e:
            transcribe(wav, model=engine, budget_s=0.1)
        assert "lease_minutes" in str(e.value)
        assert "smaller model" in str(e.value)

    def test_no_budget_means_no_cap(self, wav: Path):
        engine = _FakeEngine([_FakeSegment(0, 1, "x")], delay=0.05)
        assert len(transcribe(wav, model=engine, budget_s=None)) == 1


class TestPartialTranscripts:
    def test_a_mid_file_failure_raises_rather_than_truncating(self, wav: Path):
        """The finding this guards: half a video ingested and reported `ready`
        is wrong in the way nobody notices. The row looks healthy; the missing
        half is only found by someone searching for something that was said."""
        engine = _FakeEngine(
            [_FakeSegment(i, i + 1, f"line {i}") for i in range(10)], raise_at=5,
        )
        with pytest.raises(TranscriptionFailedError):
            transcribe(wav, model=engine)

    def test_a_file_whisper_cannot_open_raises(self, wav: Path):
        engine = _FakeEngine([], raise_on_open=RuntimeError("bad wav"))
        with pytest.raises(TranscriptionFailedError, match="could not open"):
            transcribe(wav, model=engine)


class TestCollapseRepeats:
    def test_a_long_run_collapses_to_one(self):
        segs = [Segment(float(i), i + 1.0, "same line") for i in range(12)]
        out = collapse_repeats(segs)
        assert len(out) == 1
        assert out[0].t_start == 0.0

    def test_a_short_run_is_left_alone(self):
        """Two or three identical lines can be genuine — a chant, a countdown,
        someone repeating themselves. Only runaway runs are trimmed."""
        segs = [Segment(0, 1, "yes"), Segment(1, 2, "yes")]
        assert collapse_repeats(segs) == segs

    def test_distinct_lines_are_untouched(self):
        segs = [Segment(0, 1, "a"), Segment(1, 2, "b"), Segment(2, 3, "c")]
        assert collapse_repeats(segs) == segs

    def test_a_repeat_run_between_real_content_keeps_the_surroundings(self):
        segs = (
            [Segment(0, 1, "intro")]
            + [Segment(float(i), i + 1.0, "loop") for i in range(1, 10)]
            + [Segment(10, 11, "outro")]
        )
        out = collapse_repeats(segs)
        assert [s.text for s in out] == ["intro", "loop", "outro"]

    def test_an_empty_list_is_fine(self):
        assert collapse_repeats([]) == []


class TestMissingWhisper:
    def test_a_missing_package_is_its_own_error_type(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        """Distinct from a transcription failure, and handled like
        `FFmpegMissingError`: the row must NOT be failed, or every queued video
        burns its attempts while the image is being fixed."""
        import builtins
        real_import = builtins.__import__

        def no_whisper(name, *args, **kwargs):
            if name == "faster_whisper":
                raise ImportError("no module named faster_whisper")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_whisper)
        monkeypatch.setattr("audrey.media.stt._MODEL_CACHE", {})

        with pytest.raises(WhisperUnavailableError, match="not installed"):
            load_model("small")


class TestSegmentPayload:
    def test_the_payload_matches_what_the_route_expects(self):
        """`IngestResultRequest.segments` is a list of
        `{t_start, t_end, text}` — a mismatch here is a 422 the worker would
        report as a failed job."""
        assert Segment(1.5, 2.5, "hi").as_payload() == {
            "t_start": 1.5, "t_end": 2.5, "text": "hi",
        }
