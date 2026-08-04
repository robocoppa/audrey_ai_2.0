"""Tests for the video summary (Phase 37).

Two things carry the phase. What the model is *shown* — because a summariser
handed a truncated transcript will confidently describe a video it only saw
the first third of — and what happens when the call fails, because by the time
it runs the transcript and descriptions are already ingested and already
useful, so a failure here must cost a field and never a row.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from audrey.kb.ingest import ingest_summary
from audrey.kb.qdrant import point_id
from audrey.pipeline.summarise import (
    SummaryUnavailableError,
    build_input,
    summarise_video,
)


def _segments(n: int, text: str = "spoken line") -> list[dict]:
    return [{"t_start": i * 2.0, "t_end": i * 2.0 + 2, "text": f"{text} {i}"}
            for i in range(n)]


def _frames(n: int) -> list[dict]:
    return [{"t_start": i * 30.0, "t_end": i * 30.0 + 30, "text": f"a shot of scene {i}"}
            for i in range(n)]


class TestBuildInput:
    def test_the_two_artifacts_are_labelled_separately(self):
        """They answer different questions and the model must not blend them.
        A summary reporting a whiteboard as something a person *stated* is
        worse than one that omits it."""
        out = build_input(_segments(2), _frames(2))

        assert "WHAT WAS SAID" in out
        assert "WHAT WAS ON SCREEN" in out
        assert out.index("WHAT WAS SAID") < out.index("WHAT WAS ON SCREEN")

    def test_a_silent_video_gives_its_whole_budget_to_the_descriptions(self):
        """Reserving half the budget for a transcript that does not exist
        would summarise a silent video from half the frames it has."""
        out = build_input([], _frames(40), budget=600)

        assert "WHAT WAS SAID" not in out
        assert "scene 39" in out or len(out) > 400

    def test_a_video_with_no_frames_still_summarises(self):
        out = build_input(_segments(5), [])
        assert "WHAT WAS SAID" in out
        assert "WHAT WAS ON SCREEN" not in out

    def test_nothing_at_all_produces_nothing(self):
        assert build_input([], []) == ""

    def test_long_input_is_cut_to_the_budget(self):
        out = build_input(_segments(5000), [], budget=2000)
        assert len(out) < 4000

    def test_the_cut_samples_across_the_video_rather_than_truncating(self):
        """A summary built from the first fifteen minutes is confidently wrong
        about the other forty-five and says nothing to indicate it."""
        out = build_input(_segments(500), [], budget=1500)

        assert "spoken line 0" in out
        # Something from the last fifth has to survive.
        assert any(f"spoken line {i}" in out for i in range(400, 500))

    def test_a_cut_input_tells_the_model_it_is_reading_excerpts(self):
        """A model that thinks it has the whole transcript will happily assert
        what the video concluded."""
        assert "excerpts" in build_input(_segments(5000), [], budget=2000)

    def test_an_uncut_input_does_not_claim_to_be_excerpts(self):
        assert "excerpts" not in build_input(_segments(3), _frames(2))


class _Ollama:
    def __init__(self, content="A retirement party for Jason.", boom=None):
        self.content = content
        self.boom = boom
        self.calls: list[dict] = []

    async def chat(self, *, model, messages, timeout_s):
        self.calls.append({"model": model, "messages": messages})
        if self.boom:
            raise self.boom
        return {"message": {"content": self.content}}


class _Gate:
    def __init__(self) -> None:
        self.acquired: list[tuple[str, str, str | None]] = []

    def acquire(self, model, *, location, user_id=None):
        self.acquired.append((model, location, user_id))

        class _Ctx:
            async def __aenter__(self): return None
            async def __aexit__(self, *a): return False
        return _Ctx()


def _cfg(**video):
    return SimpleNamespace(raw={"kb": {"video": video}})


class _Registry:
    def __init__(self, location="cloud"):
        self._location = location

    def location_of(self, model):
        return self._location


class TestSummariseVideo:
    @pytest.mark.asyncio
    async def test_it_returns_the_model_text(self):
        got = await summarise_video(
            _segments(3), _frames(2), filename="v.mp4", duration_s=565.0,
            ollama=_Ollama(), registry=_Registry(), gate=_Gate(), cfg=_cfg())

        assert got == "A retirement party for Jason."

    @pytest.mark.asyncio
    async def test_the_default_summariser_is_a_cloud_model(self):
        """The one stage of video ingest that costs the box no GPU."""
        ollama = _Ollama()
        await summarise_video(
            _segments(3), [], filename="v.mp4", duration_s=0.0,
            ollama=ollama, registry=_Registry(), gate=_Gate(), cfg=_cfg())

        assert ollama.calls[0]["model"] == "glm-5.2:cloud"

    @pytest.mark.asyncio
    async def test_a_local_summariser_still_takes_the_gate(self):
        """Naming a local model works and must queue like anything else, in
        the uploader's own slice — not jump it."""
        gate = _Gate()
        await summarise_video(
            _segments(3), [], filename="v.mp4", duration_s=0.0,
            ollama=_Ollama(), registry=_Registry("local"), gate=gate,
            cfg=_cfg(summarise_model="qwen3.6:35b"), user_id="bart@proton.me")

        assert gate.acquired == [("qwen3.6:35b", "local", "bart@proton.me")]

    @pytest.mark.asyncio
    async def test_the_filename_and_length_reach_the_model(self):
        ollama = _Ollama()
        await summarise_video(
            _segments(3), [], filename="jasonRetirement.mp4", duration_s=565.0,
            ollama=ollama, registry=_Registry(), gate=_Gate(), cfg=_cfg())

        prompt = ollama.calls[0]["messages"][1]["content"]
        assert "jasonRetirement.mp4" in prompt
        assert "9 minutes" in prompt

    @pytest.mark.asyncio
    async def test_nothing_to_summarise_raises_rather_than_calling_the_model(self):
        ollama = _Ollama()
        with pytest.raises(SummaryUnavailableError):
            await summarise_video(
                [], [], filename="v.mp4", duration_s=0.0,
                ollama=ollama, registry=_Registry(), gate=_Gate(), cfg=_cfg())

        assert ollama.calls == []

    @pytest.mark.asyncio
    async def test_an_empty_summary_is_a_failure_not_a_success(self):
        """It would otherwise be stored as a blank field and shown as one."""
        with pytest.raises(SummaryUnavailableError):
            await summarise_video(
                _segments(3), [], filename="v.mp4", duration_s=0.0,
                ollama=_Ollama(content="   "), registry=_Registry(),
                gate=_Gate(), cfg=_cfg())

    @pytest.mark.asyncio
    async def test_a_model_failure_propagates_for_the_caller_to_swallow(self):
        with pytest.raises(RuntimeError, match="upstream"):
            await summarise_video(
                _segments(3), [], filename="v.mp4", duration_s=0.0,
                ollama=_Ollama(boom=RuntimeError("upstream is down")),
                registry=_Registry(), gate=_Gate(), cfg=_cfg())


class _Qdrant:
    def __init__(self) -> None:
        self.points: list = []

    async def has_sparse(self, collection) -> bool:
        return False

    async def upsert_text(self, points, *, collection=None) -> None:
        self.points.extend(points)


class _Embedder:
    async def embed_many(self, texts):
        return [[0.1] * 8 for _ in texts]


class TestIngestSummary:
    @pytest.mark.asyncio
    async def test_it_becomes_exactly_one_chunk(self, tmp_path: Path):
        """A summary that needed splitting would no longer be a summary, and
        a half-summary answers nothing."""
        qdrant = _Qdrant()
        n = await ingest_summary(
            "A retirement party. " * 400,
            sidecar=tmp_path / "f1.summary.txt", qdrant=qdrant,
            embedder=_Embedder(), collection="c", user="a@b.c", file_id="f1",
            filename="v.mp4", mime="video/mp4", source_bytes=301936597)

        assert n == 1
        assert len(qdrant.points) == 1

    @pytest.mark.asyncio
    async def test_the_payload_marks_it_as_a_summary(self, tmp_path: Path):
        qdrant = _Qdrant()
        await ingest_summary(
            "A retirement party.", sidecar=tmp_path / "f1.summary.txt",
            qdrant=qdrant, embedder=_Embedder(), collection="c", user="a@b.c",
            file_id="f1", filename="v.mp4", mime="video/mp4", source_bytes=99)

        assert qdrant.points[0].payload["artifact"] == "summary"

    @pytest.mark.asyncio
    async def test_it_does_not_collide_with_the_other_two_artifacts(
        self, tmp_path: Path,
    ):
        """All three live under one file_id in one collection, and the point
        id is `(source, kind, chunk_idx)` — three chunk-0s sharing a source
        would be one point wearing three hats."""
        qdrant = _Qdrant()
        await ingest_summary(
            "A retirement party.", sidecar=tmp_path / "f1.summary.txt",
            qdrant=qdrant, embedder=_Embedder(), collection="c", user="a@b.c",
            file_id="f1", filename="v.mp4", mime="video/mp4", source_bytes=99)

        got = qdrant.points[0].id
        for other in ("f1.transcript.txt", "f1.frames.txt"):
            assert got != point_id(
                source=str((tmp_path / other).resolve()), kind="text", idx=0)

    @pytest.mark.asyncio
    async def test_an_empty_summary_ingests_nothing(self, tmp_path: Path):
        qdrant = _Qdrant()
        n = await ingest_summary(
            "   ", sidecar=tmp_path / "f1.summary.txt", qdrant=qdrant,
            embedder=_Embedder(), collection="c", user="a@b.c", file_id="f1",
            filename="v.mp4", mime="video/mp4", source_bytes=99)

        assert n == 0
        assert qdrant.points == []
        assert not (tmp_path / "f1.summary.txt").exists()

    @pytest.mark.asyncio
    async def test_the_source_video_size_is_recorded(self, tmp_path: Path):
        """Third time this trap appears: `reconcile_with_qdrant` copies
        payload bytes onto the uploads row at every boot."""
        qdrant = _Qdrant()
        await ingest_summary(
            "A retirement party.", sidecar=tmp_path / "f1.summary.txt",
            qdrant=qdrant, embedder=_Embedder(), collection="c", user="a@b.c",
            file_id="f1", filename="v.mp4", mime="video/mp4",
            source_bytes=301936597)

        assert qdrant.points[0].payload["bytes"] == 301936597


class TestConfig:
    def test_the_shipped_summariser_is_a_cloud_model(self):
        """A local default would put a summary in the same queue as the chat
        turn waiting behind it, for a stage nobody is waiting on."""
        import yaml
        cfg = yaml.safe_load(
            (Path(__file__).resolve().parent.parent / "config.yaml").read_text())
        video = cfg["kb"]["video"]

        assert video["summarise_model"].endswith(":cloud")
        assert video["summary_input_chars"] > 0
