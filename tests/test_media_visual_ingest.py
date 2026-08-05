"""Tests for the visual pass's two halves that never touch ffmpeg (Phase 36).

`describe_frames` — the budget and the failure policy, which decide whether a
video comes back partly described or not at all.

`ingest_frame_descriptions` — how descriptions land in Qdrant next to a
transcript without either artifact deleting or overwriting the other. That
coexistence is the part with a real trap in it: both live under one `file_id`
in one collection, and `point_id` is derived from `(source, kind, chunk_idx)`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from audrey.kb.ingest import ingest_frame_descriptions
from audrey.kb.qdrant import point_id
from audrey.media.describe import DescribeFailedError, describe_frames, spoken_during
from audrey.media.frames import SelectedFrame

TOKEN = "svc-token"  # noqa: S105  (test fixture, not a real secret)


def _frame(i: int, tmp_path: Path) -> SelectedFrame:
    path = tmp_path / f"f{i}.jpg"
    path.write_bytes(b"\xff\xd8\xff\xe0 fake jpeg")
    return SelectedFrame(path=path, t_start=i * 30.0, t_end=i * 30.0 + 30, represents=1)


class _Post:
    """Stands in for the worker's `post`, with a scripted reply per call."""

    def __init__(self, replies=None, *, delay: float = 0.0):
        self.calls: list[dict] = []
        self._replies = list(replies or [])
        self._delay = delay
        self.clock = 0.0

    def __call__(self, endpoint, path, token, body, **kwargs):
        self.calls.append(body)
        self.clock += self._delay
        if self._replies:
            return self._replies.pop(0)
        return 200, {"description": f"frame {len(self.calls)}", "model": "vl"}


class TestDescribeFrames:
    def test_each_keyframe_becomes_one_description(self, tmp_path: Path):
        frames = [_frame(i, tmp_path) for i in range(3)]
        post = _Post()

        described, planned = describe_frames(
            frames, user="a@b.c", post=post, endpoint="http://x", token=TOKEN)

        assert planned == 3
        assert [d["text"] for d in described] == ["frame 1", "frame 2", "frame 3"]

    def test_the_span_travels_with_the_description(self, tmp_path: Path):
        """A description of a static stretch belongs to the stretch, not to
        the instant the frame happened to be sampled at."""
        described, _ = describe_frames(
            [_frame(2, tmp_path)], user="a@b.c", post=_Post(),
            endpoint="http://x", token=TOKEN)

        assert (described[0]["t_start"], described[0]["t_end"]) == (60.0, 90.0)

    def test_the_uploading_user_is_sent_with_every_frame(self, tmp_path: Path):
        """Both fairness layers key on it. A frame posted without it would run
        in the anonymous bucket and contend with everyone."""
        post = _Post()
        describe_frames([_frame(0, tmp_path), _frame(1, tmp_path)],
                        user="bart@proton.me", post=post,
                        endpoint="http://x", token=TOKEN)

        assert [c["user"] for c in post.calls] == ["bart@proton.me"] * 2

    def test_one_rejected_frame_does_not_lose_the_others(self, tmp_path: Path):
        """A single unreadable frame is a gap in coverage. Refusing the whole
        video over it would throw away every good description and the
        transcript besides."""
        post = _Post(replies=[
            (200, {"description": "good", "model": "vl"}),
            (502, {"detail": "model barfed"}),
            (200, {"description": "also good", "model": "vl"}),
        ])

        described, planned = describe_frames(
            [_frame(i, tmp_path) for i in range(3)], user="a@b.c", post=post,
            endpoint="http://x", token=TOKEN)

        assert planned == 3
        assert [d["text"] for d in described] == ["good", "also good"]

    def test_vision_being_down_stops_the_pass_immediately(self, tmp_path: Path):
        """503 will not change within this job, and grinding through the rest
        would spend the lease collecting the same error N more times."""
        post = _Post(replies=[(503, {"detail": "no healthy vl model"})])

        with pytest.raises(DescribeFailedError, match="unavailable"):
            describe_frames([_frame(i, tmp_path) for i in range(20)],
                            user="a@b.c", post=post, endpoint="http://x", token=TOKEN)

        assert len(post.calls) == 1

    def test_an_empty_description_is_dropped_rather_than_ingested(self, tmp_path: Path):
        post = _Post(replies=[(200, {"description": "   ", "model": "vl"})])

        described, planned = describe_frames(
            [_frame(0, tmp_path)], user="a@b.c", post=post,
            endpoint="http://x", token=TOKEN)

        assert described == []
        assert planned == 1

    def test_the_budget_stops_the_pass_and_keeps_what_it_has(
        self, tmp_path: Path, monkeypatch,
    ):
        """`keyframes_max` frames at `vision.timeout_s` each can exceed the
        lease, so an unbudgeted pass would have its job swept out from under
        it and retried forever."""
        ticks = iter([0.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
        monkeypatch.setattr("audrey.media.describe.time.monotonic", lambda: next(ticks))
        post = _Post()

        described, planned = describe_frames(
            [_frame(i, tmp_path) for i in range(5)], user="a@b.c", post=post,
            endpoint="http://x", token=TOKEN, budget_s=25.0)

        assert planned == 5
        assert 0 < len(described) < 5

    def test_no_budget_describes_everything(self, tmp_path: Path):
        described, _ = describe_frames(
            [_frame(i, tmp_path) for i in range(4)], user="a@b.c", post=_Post(),
            endpoint="http://x", token=TOKEN, budget_s=None)

        assert len(described) == 4

    def test_a_zero_budget_describes_nothing(self, tmp_path: Path):
        """0.0 means the lease is already spent. Under a truthiness check that
        reads as "no budget configured" and describes *everything* — the exact
        failure the caller computed a 0.0 to prevent."""
        post = _Post()

        described, planned = describe_frames(
            [_frame(i, tmp_path) for i in range(3)], user="a@b.c", post=post,
            endpoint="http://x", token=TOKEN, budget_s=0.0)

        assert described == []
        assert planned == 3
        assert post.calls == []

    def test_an_unreadable_frame_file_is_skipped(self, tmp_path: Path):
        missing = SelectedFrame(path=tmp_path / "gone.jpg", t_start=0.0,
                                t_end=30.0, represents=1)

        described, planned = describe_frames(
            [missing, _frame(1, tmp_path)], user="a@b.c", post=_Post(),
            endpoint="http://x", token=TOKEN)

        assert planned == 2
        assert len(described) == 1


class TestLeaseAwareBudget:
    """The configured budget is a ceiling, not an entitlement.

    `TRANSCRIBE_BUDGET_S` 1440 plus `FRAME_BUDGET_S` 900 is 39 minutes against
    a 30-minute lease. Taken at face value, a long video is swept mid-describe,
    re-claimed, and burns its attempts doing the same thing every time.
    """

    def _budget(self, *, lease_s, elapsed, configured):
        import audrey.media.worker as worker
        now = worker.time.monotonic()
        return worker._frame_budget(
            {"lease_seconds": lease_s}, now - elapsed, configured)

    def test_a_fresh_lease_allows_the_configured_budget(self):
        assert self._budget(lease_s=1800, elapsed=0, configured=900) == 900

    def test_a_mostly_spent_lease_cuts_the_budget(self):
        """24 minutes of transcription leaves 6, minus the reserve."""
        got = self._budget(lease_s=1800, elapsed=1440, configured=900)
        assert got == pytest.approx(1800 - 1440 - 120, abs=1)

    def test_a_spent_lease_yields_zero_not_a_negative(self):
        """Negative would be worse than useless — `describe_frames` compares
        elapsed against it, and every elapsed exceeds a negative immediately,
        which happens to be right for the wrong reason."""
        assert self._budget(lease_s=1800, elapsed=2000, configured=900) == 0.0

    def test_a_claim_without_a_lease_falls_back_to_the_configured_budget(self):
        """An audrey-ai that predates this field still hands out jobs."""
        import audrey.media.worker as worker
        assert worker._frame_budget({}, worker.time.monotonic(), 900) == 900


class _Qdrant:
    def __init__(self) -> None:
        self.points: list = []
        self.deleted: list[str] = []

    async def delete_by_file_id(self, file_id, *, user, collection) -> None:
        self.deleted.append(file_id)

    async def has_sparse(self, collection) -> bool:
        return False

    async def upsert_text(self, points, *, collection=None) -> None:
        self.points.extend(points)


class _Embedder:
    async def embed_many(self, texts):
        return [[0.1] * 8 for _ in texts]


async def _ingest(tmp_path: Path, frames, **over):
    sidecar = tmp_path / "f1.frames.txt"
    sidecar.write_text("whatever")
    qdrant = _Qdrant()
    n = await ingest_frame_descriptions(
        frames, sidecar=sidecar, qdrant=qdrant, embedder=_Embedder(),
        collection="c", user="a@b.c", file_id="f1", filename="v.mp4",
        mime="video/mp4", source_bytes=301936597, **over,
    )
    return n, qdrant


class TestIngestFrameDescriptions:
    @pytest.mark.asyncio
    async def test_each_description_becomes_a_chunk(self, tmp_path: Path):
        n, qdrant = await _ingest(tmp_path, [
            {"t_start": 0.0, "t_end": 30.0, "text": "a whiteboard reading DEPLOY"},
            {"t_start": 30.0, "t_end": 60.0, "text": "two people in chairs"},
        ])

        assert n == 2
        assert len(qdrant.points) == 2

    @pytest.mark.asyncio
    async def test_the_payload_marks_it_as_visual_not_transcript(self, tmp_path: Path):
        """The discriminator that lets a caller tell 'this was said' from
        'this was shown' — two different answers about the same second."""
        _, qdrant = await _ingest(tmp_path, [
            {"t_start": 0.0, "t_end": 30.0, "text": "a whiteboard"}])

        assert qdrant.points[0].payload["artifact"] == "visual"

    @pytest.mark.asyncio
    async def test_the_timestamps_ride_in_the_payload(self, tmp_path: Path):
        _, qdrant = await _ingest(tmp_path, [
            {"t_start": 60.0, "t_end": 90.0, "text": "a whiteboard"}])

        payload = qdrant.points[0].payload
        assert (payload["t_start"], payload["t_end"]) == (60.0, 90.0)
        assert "60" not in payload["text"]

    @pytest.mark.asyncio
    async def test_the_source_video_size_is_recorded(self, tmp_path: Path):
        """Same trap as the transcript path: `reconcile_with_qdrant` copies
        payload bytes onto the uploads row at every boot, so the sidecar's own
        size here would bill a 288 MB video as a few KB."""
        _, qdrant = await _ingest(tmp_path, [
            {"t_start": 0.0, "t_end": 30.0, "text": "a whiteboard"}])

        assert qdrant.points[0].payload["bytes"] == 301936597

    @pytest.mark.asyncio
    async def test_it_does_not_delete_the_transcript_by_default(self, tmp_path: Path):
        """`delete_by_file_id` removes *every* point for the file. If this
        deleted on its own way in, it would take out the transcript written
        moments earlier — the route clears once for both."""
        _, qdrant = await _ingest(tmp_path, [
            {"t_start": 0.0, "t_end": 30.0, "text": "a whiteboard"}])

        assert qdrant.deleted == []

    @pytest.mark.asyncio
    async def test_frame_points_do_not_collide_with_transcript_points(
        self, tmp_path: Path,
    ):
        """Both artifacts live under one file_id in one collection, and the
        point id is `(source, kind, chunk_idx)`. Sharing a source would make
        frame chunk 0 and transcript chunk 0 the same point, each silently
        overwriting the other."""
        _, qdrant = await _ingest(tmp_path, [
            {"t_start": 0.0, "t_end": 30.0, "text": "a whiteboard"}])

        transcript_source = str((tmp_path / "f1.transcript.txt").resolve())
        assert qdrant.points[0].id != point_id(
            source=transcript_source, kind="text", idx=0)

    @pytest.mark.asyncio
    async def test_a_long_description_is_chunked_with_unique_ids(self, tmp_path: Path):
        """A dense slide transcribed verbatim can be long. Chunk numbering
        runs across the whole set, not per frame — restarting at 0 for each
        frame would collapse every frame's first chunk onto one point."""
        long_text = "word " * 900
        n, qdrant = await _ingest(tmp_path, [
            {"t_start": 0.0, "t_end": 30.0, "text": long_text},
            {"t_start": 30.0, "t_end": 60.0, "text": long_text},
        ], chunk_tokens=100, overlap_tokens=10)

        assert n > 2
        assert len({p.id for p in qdrant.points}) == n

    @pytest.mark.asyncio
    async def test_chunks_never_straddle_two_frames(self, tmp_path: Path):
        """Two descriptions are about two different moments. A chunk spanning
        them would be text that was never true of either, attached to
        whichever timestamp came first."""
        _, qdrant = await _ingest(tmp_path, [
            {"t_start": 0.0, "t_end": 30.0, "text": "alpha " * 200},
            {"t_start": 30.0, "t_end": 60.0, "text": "bravo " * 200},
        ], chunk_tokens=50, overlap_tokens=5)

        for p in qdrant.points:
            text, start = p.payload["text"], p.payload["t_start"]
            assert ("alpha" in text) == (start == 0.0), (start, text[:40])

    @pytest.mark.asyncio
    async def test_no_frames_ingests_nothing(self, tmp_path: Path):
        n, qdrant = await _ingest(tmp_path, [])
        assert n == 0
        assert qdrant.points == []

    @pytest.mark.asyncio
    async def test_blank_descriptions_are_skipped(self, tmp_path: Path):
        n, _ = await _ingest(tmp_path, [
            {"t_start": 0.0, "t_end": 30.0, "text": "   "},
            {"t_start": 30.0, "t_end": 60.0, "text": "real prose"},
        ])
        assert n == 1


# ─── Transcript context for a keyframe (Phase 38) ──────────────────────

class TestSpokenDuring:
    """A description is written at ingest, so there is no user and no question
    to steer it — whatever gets asked arrives hours or days later. The speech
    over a frame is the closest available proxy for what matters in it.

    The `hint` field on `/v1/media/describe` was built for this in phase 36
    and went unpopulated until now.
    """

    def test_speech_overlapping_the_frame_window_is_picked_up(self):
        segments = [
            {"t_start": 0.0, "t_end": 4.0, "text": "Before the frame."},
            {"t_start": 30.0, "t_end": 34.0, "text": "This slide shows Q3 revenue."},
            {"t_start": 300.0, "t_end": 304.0, "text": "Long after."},
        ]

        assert spoken_during(segments, 28.0, 40.0) == "This slide shows Q3 revenue."

    def test_a_segment_straddling_the_boundary_counts(self):
        """Overlap, not containment. A keyframe spans everything it stands in
        for — after the gate that can be minutes — while a segment is a few
        seconds, so a containment test would almost always come back empty."""
        segments = [{"t_start": 25.0, "t_end": 35.0, "text": "Straddles the start."}]

        assert spoken_during(segments, 30.0, 60.0) == "Straddles the start."

    def test_several_segments_join_in_order(self):
        segments = [
            {"t_start": 30.0, "t_end": 32.0, "text": "First."},
            {"t_start": 33.0, "t_end": 35.0, "text": "Second."},
        ]

        assert spoken_during(segments, 30.0, 60.0) == "First. Second."

    def test_no_segments_yields_no_hint(self):
        """A silent video is an ordinary case, not an error — phase 35. Its
        frames are described with no context at all."""
        assert spoken_during(None, 0.0, 30.0) == ""
        assert spoken_during([], 0.0, 30.0) == ""

    def test_nothing_said_over_this_frame_yields_no_hint(self):
        segments = [{"t_start": 0.0, "t_end": 4.0, "text": "Elsewhere."}]

        assert spoken_during(segments, 100.0, 130.0) == ""

    def test_the_hint_is_truncated_on_a_segment_boundary(self):
        """Cut mid-sentence, the context reads as garbled speech rather than
        as an excerpt — and the model is being asked to judge relevance from
        it, not to transcribe it."""
        segments = [
            {"t_start": 30.0, "t_end": 31.0, "text": "A" * 40},
            {"t_start": 31.0, "t_end": 32.0, "text": "B" * 40},
            {"t_start": 32.0, "t_end": 33.0, "text": "C" * 40},
        ]

        out = spoken_during(segments, 30.0, 60.0, max_chars=90)

        assert out == f"{'A' * 40} {'B' * 40}"

    def test_a_segment_with_no_text_is_skipped(self):
        segments = [
            {"t_start": 30.0, "t_end": 31.0, "text": "   "},
            {"t_start": 31.0, "t_end": 32.0, "text": "Real words."},
        ]

        assert spoken_during(segments, 30.0, 60.0) == "Real words."

    def test_a_segment_missing_its_end_time_is_treated_as_an_instant(self):
        """faster-whisper always supplies both, but the payload crosses a
        network boundary and this must not raise on a partial one."""
        segments = [{"t_start": 30.0, "text": "No end time."}]

        assert spoken_during(segments, 29.0, 31.0) == "No end time."


class TestTheHintReachesTheRoute:
    def test_describe_frames_sends_the_surrounding_speech(self, monkeypatch):
        posted: list[dict] = []

        def _post(endpoint, path, token, body, *, timeout=None):
            posted.append(body)
            return 200, {"description": "A slide reading Q3 REVENUE.",
                         "model": "qwen3-vl:32b", "elapsed_s": 1.0}

        monkeypatch.setattr("audrey.media.describe._read_b64", lambda p: "AAAA")
        frames = [SelectedFrame(
            path=Path("/nope.jpg"), t_start=30.0, t_end=60.0, represents=1)]
        segments = [{"t_start": 31.0, "t_end": 34.0, "text": "Q3 revenue is up."}]

        described, planned = describe_frames(
            frames, user="a@b.c", post=_post,
            endpoint="http://audrey-ai:8000", token=TOKEN, segments=segments,
        )

        assert planned == 1
        assert len(described) == 1
        assert posted[0]["hint"] == "Q3 revenue is up."

    def test_a_silent_video_sends_an_empty_hint(self, monkeypatch):
        posted: list[dict] = []

        def _post(endpoint, path, token, body, *, timeout=None):
            posted.append(body)
            return 200, {"description": "A title card.", "model": "m", "elapsed_s": 1.0}

        monkeypatch.setattr("audrey.media.describe._read_b64", lambda p: "AAAA")
        frames = [SelectedFrame(
            path=Path("/nope.jpg"), t_start=0.0, t_end=30.0, represents=1)]

        describe_frames(
            frames, user="a@b.c", post=_post,
            endpoint="http://audrey-ai:8000", token=TOKEN, segments=None,
        )

        assert posted[0]["hint"] == ""
