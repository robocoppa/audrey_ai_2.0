"""Tests for the media worker's claim loop (Phase 34).

The loop's job is to survive things going wrong without losing a job or
spinning. What's pinned here is mostly the unhappy paths:

  - a source path the worker can't open (the mount is wrong),
  - ffmpeg rejecting the file (the video is bad),
  - ffmpeg missing entirely (the *image* is bad),

and the fact that those three get different treatment. The middle one fails
the row. The last one must not, or a broken image quietly burns every job's
attempts while the operator is still working out what's wrong.

HTTP is stubbed at `worker.post`. The transport itself is exercised for real
against the deployed box; what these tests are for is the decision logic
sitting on top of it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from audrey.media import worker
from audrey.media.audio import FFmpegMissingError
from audrey.media.stt import Segment, TranscriptionFailedError, WhisperUnavailableError

TOKEN = "not-a-real-token"  # noqa: S105  (test fixture)

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


class _Calls:
    """Records what the worker posted, and replies with a script."""

    def __init__(self, replies: list[tuple[int, dict]] | None = None):
        self.posted: list[tuple[str, dict | None]] = []
        self._replies = list(replies or [])

    def __call__(self, endpoint: str, path: str, token: str, body: dict | None,
                 **kwargs):
        # **kwargs absorbs `timeout`, which the result post overrides — a
        # transcript for a long video is a much bigger body than a claim.
        self.posted.append((path, body))
        if self._replies:
            return self._replies.pop(0)
        return 200, {}

    def paths(self) -> list[str]:
        return [p for p, _ in self.posted]

    def body_for(self, needle: str) -> dict:
        for path, body in self.posted:
            if needle in path:
                return body or {}
        raise AssertionError(f"nothing was posted to a path containing {needle!r}")


@pytest.fixture(autouse=True)
def stub_whisper(monkeypatch: pytest.MonkeyPatch):
    """faster-whisper is not installed outside the media-worker image.

    Autouse so every test in this file exercises the loop's decisions rather
    than whisper's. The driver's own logic is covered in test_media_stt.py,
    against a fake engine for the same reason.
    """
    def fake(wav, **kwargs):
        return [Segment(0.0, 1.0, "spoken words")]

    monkeypatch.setattr(worker, "transcribe", fake)
    return fake


@pytest.fixture
def video(tmp_path: Path) -> Path:
    """One second with a tone, written where a job would point."""
    import subprocess
    path = tmp_path / "clip.mp4"
    subprocess.run(
        [shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=10",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-c:v", "mpeg4", "-c:a", "aac", "-shortest", str(path)],
        check=True, capture_output=True,
    )
    return path


def _job(path: Path, **over) -> dict:
    job = {
        "file_id": "f1", "filename": path.name, "path": str(path),
        "lease_id": "L1", "user": "a@b.c", "bytes": 10, "attempts": 1,
    }
    job.update(over)
    return job


class TestHandleJob:
    def test_a_good_video_posts_a_result_with_its_duration(
        self, video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        calls = _Calls()
        monkeypatch.setattr(worker, "post", calls)

        worker.handle_job(
            _job(video), endpoint="http://x", token=TOKEN, work_dir=tmp_path / "w",
        )

        assert calls.paths() == ["/v1/files/f1/ingest-result"]
        body = calls.body_for("ingest-result")
        assert body["lease_id"] == "L1"
        assert body["duration_s"] == pytest.approx(1.0, abs=0.3)
        assert body["segments"] == [
            {"t_start": 0.0, "t_end": 1.0, "text": "spoken words"},
        ]

    def test_the_intermediate_wav_is_cleaned_up(
        self, video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """Left behind, these accumulate at video size until the container is
        recreated — and the worker's scratch dir is inside the container, so
        nobody would go looking."""
        work = tmp_path / "w"
        monkeypatch.setattr(worker, "post", _Calls())

        worker.handle_job(_job(video), endpoint="http://x", token=TOKEN, work_dir=work)

        assert list(work.glob("*.wav")) == []

    def test_a_missing_source_is_reported_as_a_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """The mount is wrong. Retrying cannot fix it, so the row should say so
        rather than cycling until it runs out of attempts."""
        calls = _Calls()
        monkeypatch.setattr(worker, "post", calls)

        worker.handle_job(
            _job(tmp_path / "gone.mp4"), endpoint="http://x", token=TOKEN,
            work_dir=tmp_path / "w",
        )

        assert calls.paths() == ["/v1/files/f1/ingest-failed"]
        assert "mount" in calls.body_for("ingest-failed")["reason"]

    def test_an_unreadable_video_is_reported_as_a_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        bad = tmp_path / "bad.mp4"
        bad.write_bytes(b"not a container")
        calls = _Calls()
        monkeypatch.setattr(worker, "post", calls)

        worker.handle_job(
            _job(bad), endpoint="http://x", token=TOKEN, work_dir=tmp_path / "w",
        )

        assert calls.paths() == ["/v1/files/f1/ingest-failed"]
        # ffmpeg's own words, so the failed row explains itself.
        assert calls.body_for("ingest-failed")["reason"].strip() != ""

    def test_a_missing_ffmpeg_does_not_fail_the_row(
        self, video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """The image is broken, not the video. Failing the row here would burn
        attempts on every queued job while the operator debugs the image, and
        they would all be `failed` by the time it was fixed. Raising instead
        leaves the lease to expire and the job to survive."""
        calls = _Calls()
        monkeypatch.setattr(worker, "post", calls)
        monkeypatch.setattr(shutil, "which", lambda _name: None)

        with pytest.raises(FFmpegMissingError):
            worker.handle_job(
                _job(video), endpoint="http://x", token=TOKEN, work_dir=tmp_path / "w",
            )

        assert calls.posted == []

    def test_a_rejected_result_is_logged_not_raised(
        self, video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """A 409 means the lease was swept while we worked. The job is gone,
        which is fine — the worker's next move is to claim another, not die."""
        calls = _Calls(replies=[(409, {"detail": "reclaimed"})])
        monkeypatch.setattr(worker, "post", calls)

        worker.handle_job(
            _job(video), endpoint="http://x", token=TOKEN, work_dir=tmp_path / "w",
        )


class TestRunLoop:
    def test_an_empty_queue_exits_cleanly_in_once_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(worker, "post", _Calls(replies=[(204, {})]))
        assert worker.run(
            endpoint="http://x", token=TOKEN, poll_seconds=0,
            work_dir=tmp_path, once=True,
        ) == 0

    def test_it_claims_then_handles_then_exits(
        self, video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        calls = _Calls(replies=[(200, _job(video)), (200, {})])
        monkeypatch.setattr(worker, "post", calls)

        assert worker.run(
            endpoint="http://x", token=TOKEN, poll_seconds=0,
            work_dir=tmp_path, once=True,
        ) == 0
        assert calls.paths() == ["/v1/files/jobs/claim", "/v1/files/f1/ingest-result"]

    def test_a_rejected_claim_is_not_treated_as_an_empty_queue(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """401 from a bad token must not look like 'nothing to do' — that is
        how a misconfigured worker sits quiet for a week."""
        monkeypatch.setattr(worker, "post", _Calls(replies=[(401, {"detail": "no"})]))
        assert worker.run(
            endpoint="http://x", token=TOKEN, poll_seconds=0,
            work_dir=tmp_path, once=True,
        ) == 1

    def test_a_connection_error_does_not_kill_the_worker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """Audrey restarting is routine. The worker must outlive it."""
        state = {"calls": 0}

        def flaky(endpoint, path, token, body):
            state["calls"] += 1
            if state["calls"] == 1:
                raise OSError("connection refused")
            return 204, {}

        monkeypatch.setattr(worker, "post", flaky)
        monkeypatch.setattr(worker.time, "sleep", lambda _s: None)

        assert worker.run(
            endpoint="http://x", token=TOKEN, poll_seconds=0,
            work_dir=tmp_path, once=True,
        ) == 0
        assert state["calls"] == 2

    def test_sigterm_stops_the_loop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """A compose stop should drain, not abandon a job mid-flight — that
        costs a full lease expiry before anything picks it up again."""
        stopper = worker.Stopping()

        def claim_then_stop(endpoint, path, token, body):
            stopper.requested = True
            return 204, {}

        monkeypatch.setattr(worker, "post", claim_then_stop)
        monkeypatch.setattr(worker, "Stopping", lambda: stopper)
        monkeypatch.setattr(worker.time, "sleep", lambda _s: None)

        assert worker.run(
            endpoint="http://x", token=TOKEN, poll_seconds=0, work_dir=tmp_path,
        ) == 0


class TestMain:
    def test_it_refuses_to_start_without_a_token(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        """Every claim would 401. Failing at startup is a one-line log; the
        alternative is an error every POLL_SECONDS forever."""
        monkeypatch.delenv("KB_SERVICE_TOKEN", raising=False)
        assert worker.main() == 2


class TestShutdownLatency:
    """A graceful stop has to be faster than Docker's patience.

    The worker sleeps between polls. If the stop flag is only read at the top
    of the loop, an idle shutdown takes up to a full poll interval — measured
    at 7.3s against a 10s poll on the box. Docker's default
    `stop_grace_period` is 10s, so a POLL_SECONDS of 30 would have turned
    every graceful stop into a SIGKILL, and the drain logic would have been
    dead code that still looked correct in review.
    """

    def test_wait_returns_immediately_once_stopping(self):
        import time as _time
        stopper = worker.Stopping()
        stopper.requested = True

        t0 = _time.monotonic()
        stopper.wait(30)
        assert _time.monotonic() - t0 < 0.5

    def test_wait_notices_a_stop_raised_partway_through(self):
        import threading
        import time as _time
        stopper = worker.Stopping()
        threading.Timer(0.2, lambda: setattr(stopper, "requested", True)).start()

        t0 = _time.monotonic()
        stopper.wait(30)
        elapsed = _time.monotonic() - t0
        # Not the full 30s, and not instant either — it actually waited.
        assert 0.1 < elapsed < 2.0

    def test_wait_sleeps_the_whole_interval_when_not_stopping(self):
        import time as _time
        stopper = worker.Stopping()

        t0 = _time.monotonic()
        stopper.wait(0.3)
        assert _time.monotonic() - t0 >= 0.25


class TestTranscriptStage:
    """Phase 35. The worker now posts what was said, not just how long it was."""

    def test_a_silent_video_is_not_sent_to_whisper(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """Feeding whisper an audio file that does not exist would fail a job
        that has nothing wrong with it. `extract_audio` returns 0.0 and writes
        nothing for a file with no audio stream, and that is the signal."""
        import subprocess
        silent = tmp_path / "silent.mp4"
        subprocess.run(
            [shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-y",
             "-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=10",
             "-c:v", "mpeg4", str(silent)],
            check=True, capture_output=True,
        )
        called = []
        monkeypatch.setattr(worker, "transcribe", lambda *a, **k: called.append(1) or [])
        calls = _Calls()
        monkeypatch.setattr(worker, "post", calls)

        worker.handle_job(
            _job(silent), endpoint="http://x", token=TOKEN, work_dir=tmp_path / "w",
        )

        assert called == []
        assert calls.paths() == ["/v1/files/f1/ingest-result"]
        body = calls.body_for("ingest-result")
        assert body["segments"] == []
        assert body["duration_s"] == 0.0

    def test_a_transcription_failure_fails_the_row(
        self, video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """Budget overrun or a mid-file decoder crash. The video is the
        problem, so the row carries the reason."""
        def boom(wav, **kwargs):
            raise TranscriptionFailedError("exceeded its 1440s budget")

        monkeypatch.setattr(worker, "transcribe", boom)
        calls = _Calls()
        monkeypatch.setattr(worker, "post", calls)

        worker.handle_job(
            _job(video), endpoint="http://x", token=TOKEN, work_dir=tmp_path / "w",
        )

        assert calls.paths() == ["/v1/files/f1/ingest-failed"]
        assert "budget" in calls.body_for("ingest-failed")["reason"]

    def test_missing_whisper_does_not_fail_the_row(
        self, video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """The image is broken, not the video — the same distinction ffmpeg
        gets. Failing rows here would mark every queued video `failed` while
        the image is being rebuilt."""
        def missing(wav, **kwargs):
            raise WhisperUnavailableError("weights are not baked into this image")

        monkeypatch.setattr(worker, "transcribe", missing)
        calls = _Calls()
        monkeypatch.setattr(worker, "post", calls)

        with pytest.raises(WhisperUnavailableError):
            worker.handle_job(
                _job(video), endpoint="http://x", token=TOKEN, work_dir=tmp_path / "w",
            )

        assert calls.posted == []

    def test_the_wav_is_removed_even_when_transcription_fails(
        self, video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """A failed job is retried. Without cleanup each attempt leaves another
        copy of the audio inside the container until the disk fills."""
        work = tmp_path / "w"
        monkeypatch.setattr(worker, "transcribe",
                            lambda *a, **k: (_ for _ in ()).throw(
                                TranscriptionFailedError("nope")))
        monkeypatch.setattr(worker, "post", _Calls())

        worker.handle_job(_job(video), endpoint="http://x", token=TOKEN, work_dir=work)

        assert list(work.glob("*.wav")) == []

    def test_the_model_size_reaches_the_driver(
        self, video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """WHISPER_MODEL only selects among models baked into the image, so a
        value that never arrives is a silent fall back to the default."""
        seen = {}

        def capture(wav, **kwargs):
            seen.update(kwargs)
            return []

        monkeypatch.setattr(worker, "transcribe", capture)
        monkeypatch.setattr(worker, "post", _Calls())

        worker.handle_job(
            _job(video), endpoint="http://x", token=TOKEN, work_dir=tmp_path / "w",
            model_size="medium", budget_s=99.0,
        )

        assert seen["model_size"] == "medium"
        assert seen["budget_s"] == 99.0
