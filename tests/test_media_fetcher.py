"""The media fetcher's claim loop (Phase 41, step 2).

`fetch.py` owns the judgement and is tested next door. This file owns the
sequence, and the sequence is where the two-container contract lives:

  - **Nothing reaches the final path until it is finished.** The rename is the
    last thing before the post, so a crash anywhere earlier leaves a file in
    staging that the next sweep collects — never a partial video at the path
    the media worker derives from the row.
  - **Every failure is reported.** A fetcher that dies quietly leaves the row
    in `fetching` until the lease expires: it recovers, but it costs a full
    lease period during which the page says "downloading" about nothing.
  - **The reported filename carries the real extension.** `fetch/{id}/result`
    rebuilds both paths from it, so this is the one field the two containers
    have to agree on.

`post` is patched at `audrey.media.fetcher.post`, and the download and probe
are stubbed — this is about what happens between them, not about yt-dlp.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from audrey.media import fetcher
from audrey.media.fetch import (
    SOURCE_AUTO_CAPTIONS,
    SOURCE_SUBTITLES,
    Downloaded,
    FetchFailedError,
    FetchRefusedError,
    UrlInfo,
    YtDlpMissingError,
)

TOKEN = "not-a-real-token"  # noqa: S105  (test fixture)
FILE_ID = "11111111-2222-3333-4444-555555555555"
URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    stage = tmp_path / "uploads" / ".staging"
    dest = tmp_path / "uploads" / "a_b_c"
    stage.mkdir(parents=True)
    dest.mkdir(parents=True)
    return stage, dest


@pytest.fixture
def job(dirs: tuple[Path, Path]) -> dict:
    stage, dest = dirs
    return {
        "file_id": FILE_ID,
        "user": "a@b.c",
        "source_url": URL,
        "lease_id": "lease-1",
        "attempts": 1,
        "stage_dir": str(stage),
        "dest_dir": str(dest),
        "lease_seconds": 1200,
        "max_bytes": 2 * 1024**3,
        "max_duration_s": 7200,
    }


@pytest.fixture
def posted(monkeypatch) -> list[tuple[str, dict]]:
    """Record every POST the fetcher makes, and answer 200 to all of them."""
    calls: list[tuple[str, dict]] = []

    def fake_post(_endpoint, path, _token, body, **_kw):
        calls.append((path, body or {}))
        return 200, {}

    monkeypatch.setattr(fetcher, "post", fake_post)
    return calls


def _info(**kw) -> UrlInfo:
    base = {
        "title": "A Retirement Speech",
        "ext": "mp4",
        "duration_s": 565.0,
        "filesize_approx": 100 * 1024 * 1024,
    }
    return UrlInfo(**{**base, **kw})


def _stub(monkeypatch, *, info=None, result=None, probe_raises=None, download_raises=None):
    def fake_probe(_url, **_kw):
        if probe_raises is not None:
            raise probe_raises
        return info or _info()

    def fake_download(_url, stage_dir, file_id, **_kw):
        if download_raises is not None:
            raise download_raises
        if result is not None:
            return result
        path = Path(stage_dir) / f"{file_id}.mp4"
        path.write_bytes(b"video bytes")
        return Downloaded(path=path)

    monkeypatch.setattr(fetcher, "probe_url", fake_probe)
    monkeypatch.setattr(fetcher, "download", fake_download)


def _run(job: dict) -> None:
    fetcher.handle_job(
        job, endpoint="http://audrey", token=TOKEN, probe_timeout_s=30,
    )


class TestHappyPath:
    def test_the_file_is_moved_into_the_users_directory(self, job, dirs, posted, monkeypatch):
        stage, dest = dirs
        _stub(monkeypatch)
        _run(job)
        assert (dest / f"{FILE_ID}.mp4").read_bytes() == b"video bytes"
        # Nothing left behind in staging: the rename took the video, and the
        # tidy-up took everything else this job wrote.
        assert list(stage.iterdir()) == []

    def test_the_result_reports_the_title_with_the_real_extension(
        self, job, posted, monkeypatch,
    ):
        _stub(monkeypatch)
        _run(job)
        path, body = posted[-1]
        assert path == f"/v1/files/fetch/{FILE_ID}/result"
        # The extension is not decoration: `fetch/{id}/result` rebuilds the
        # staged and final paths from it. The stem is the video's real title,
        # which is what replaces the placeholder in the file list.
        assert body["filename"] == "A Retirement Speech.mp4"
        assert body["duration_s"] == 565.0

    def test_an_awkward_title_still_produces_a_usable_filename(
        self, job, posted, monkeypatch,
    ):
        _stub(monkeypatch, info=_info(title="10/10 — Q&A: \"why?\" (part 2)"))
        _run(job)
        name = posted[-1][1]["filename"]
        # A title is arbitrary text from someone else's website. Path
        # separators are the part that matters — the rest is legibility.
        assert "/" not in name
        assert name.endswith(".mp4")

    def test_a_title_that_sanitizes_to_nothing_falls_back_to_the_id(
        self, job, posted, monkeypatch,
    ):
        _stub(monkeypatch, info=_info(title="///"))
        # An empty filename is refused by the route, which would fail the row
        # for a download that worked perfectly.
        assert posted or True
        _run(job)
        assert posted[-1][1]["filename"] == f"{FILE_ID}.mp4"

    def test_a_very_long_title_is_cut_rather_than_refused(self, job, posted, monkeypatch):
        _stub(monkeypatch, info=_info(title="word " * 200))
        _run(job)
        name = posted[-1][1]["filename"]
        assert len(name) <= 130
        assert name.endswith(".mp4")

    def test_the_extension_follows_the_file_not_the_metadata(
        self, job, dirs, posted, monkeypatch,
    ):
        stage, dest = dirs
        landed = stage / f"{FILE_ID}.mkv"
        landed.write_bytes(b"matroska")
        _stub(
            monkeypatch,
            info=_info(ext="mp4"),          # metadata said mp4 …
            result=Downloaded(path=landed),  # … and the format selector fell through
        )
        _run(job)
        assert posted[-1][1]["filename"].endswith(".mkv")
        assert (dest / f"{FILE_ID}.mkv").exists()


class TestFailuresAreReported:
    def test_a_refused_url_fails_the_row_with_the_reason(self, job, posted, monkeypatch):
        _stub(monkeypatch, probe_raises=FetchRefusedError("this video is private"))
        _run(job)
        path, body = posted[-1]
        assert path == f"/v1/files/fetch/{FILE_ID}/failed"
        # The reason is the deliverable. It is stored verbatim and shown on the
        # page, and "download failed" is what this exists instead of.
        assert body["reason"] == "this video is private"

    def test_a_video_over_the_duration_cap_never_downloads(self, job, posted, monkeypatch):
        calls: list[str] = []

        def fake_download(*_a, **_kw):
            calls.append("downloaded")
            raise AssertionError("must not be reached")

        monkeypatch.setattr(fetcher, "probe_url", lambda *_a, **_k: _info(duration_s=99999))
        monkeypatch.setattr(fetcher, "download", fake_download)
        _run(job)
        assert calls == []
        assert posted[-1][0].endswith("/failed")
        assert "refuses anything over" in posted[-1][1]["reason"]

    def test_a_download_failure_is_reported_rather_than_raised(
        self, job, posted, monkeypatch,
    ):
        _stub(monkeypatch, download_raises=FetchFailedError("the connection dropped"))
        _run(job)
        # Raising would leave the row in `fetching` until the lease expires —
        # recoverable, but the page says "downloading" about nothing for 20
        # minutes and nobody learns why.
        assert posted[-1][0].endswith("/failed")

    def test_an_empty_download_is_caught_here_too(self, job, dirs, posted, monkeypatch):
        stage, dest = dirs
        empty = stage / f"{FILE_ID}.mp4"
        empty.write_bytes(b"")
        _stub(monkeypatch, result=Downloaded(path=empty))
        _run(job)
        assert posted[-1][0].endswith("/failed")
        assert "empty" in posted[-1][1]["reason"]
        # And it never reached the user's directory.
        assert not (dest / f"{FILE_ID}.mp4").exists()

    def test_an_oversized_download_is_refused_before_the_move(
        self, job, dirs, posted, monkeypatch,
    ):
        stage, dest = dirs
        big = stage / f"{FILE_ID}.mp4"
        big.write_bytes(b"x" * 5000)
        job["max_bytes"] = 1000
        # No estimate from the metadata pass, which is the case this backstop
        # is for: plenty of sites report no size, and `--max-filesize` does not
        # apply to every downloader path.
        _stub(monkeypatch, info=_info(filesize_approx=0), result=Downloaded(path=big))
        _run(job)
        assert posted[-1][0].endswith("/failed")
        # Reported as a size problem, not left to trip the quota check on the
        # far side — which would tell the user their account was full.
        assert "over this server's" in posted[-1][1]["reason"]
        assert not (dest / f"{FILE_ID}.mp4").exists()

    def test_staging_is_tidied_even_when_the_job_fails(
        self, job, dirs, posted, monkeypatch,
    ):
        stage, _dest = dirs
        (stage / f"{FILE_ID}.f137.mp4.part").write_bytes(b"partial")
        _stub(monkeypatch, download_raises=FetchRefusedError("nope"))
        _run(job)
        assert list(stage.iterdir()) == []

    def test_another_jobs_staged_file_is_left_alone(self, job, dirs, posted, monkeypatch):
        stage, _dest = dirs
        other = stage / "99999999-0000-0000-0000-000000000000.mp4"
        other.write_bytes(b"someone else's download in progress")
        _stub(monkeypatch, download_raises=FetchRefusedError("nope"))
        _run(job)
        # Two fetchers can run at once. Tidying by directory rather than by
        # file_id would delete a live download out from under one of them.
        assert other.exists()

    def test_a_broken_image_propagates_instead_of_failing_rows(
        self, job, posted, monkeypatch,
    ):
        _stub(monkeypatch, probe_raises=YtDlpMissingError("yt-dlp is not on PATH"))
        with pytest.raises(YtDlpMissingError):
            _run(job)
        # Nothing reported: failing rows for an image that is built wrong
        # would burn every queued URL's attempts, and they would all be
        # 'failed' by the time anyone noticed.
        assert posted == []


class TestCaptionsRideAlong:
    def test_a_caption_track_is_posted_with_the_result(self, job, dirs, posted, monkeypatch):
        stage, _dest = dirs
        path = stage / f"{FILE_ID}.mp4"
        path.write_bytes(b"video")
        _stub(monkeypatch, result=Downloaded(
            path=path,
            segments=[{"t_start": 0.0, "t_end": 3.0, "text": "Hello."}],
            transcript_source=SOURCE_SUBTITLES,
        ))
        _run(job)
        body = posted[-1][1]
        assert body["transcript_source"] == SOURCE_SUBTITLES
        assert body["segments"] == [{"t_start": 0.0, "t_end": 3.0, "text": "Hello."}]

    def test_the_kind_of_captions_asked_for_comes_from_the_metadata(
        self, job, posted, monkeypatch,
    ):
        seen: list[str] = []

        def fake_download(_url, stage_dir, file_id, *, caption_source="", **_kw):
            seen.append(caption_source)
            path = Path(stage_dir) / f"{file_id}.mp4"
            path.write_bytes(b"v")
            return Downloaded(path=path)

        monkeypatch.setattr(
            fetcher, "probe_url",
            lambda *_a, **_k: _info(auto_caption_langs=("en",)),
        )
        monkeypatch.setattr(fetcher, "download", fake_download)
        _run(job)
        assert seen == [SOURCE_AUTO_CAPTIONS]

    def test_no_captions_posts_nothing_rather_than_an_empty_attribution(
        self, job, posted, monkeypatch,
    ):
        _stub(monkeypatch)
        body = posted[-1][1] if posted else {}
        _run(job)
        body = posted[-1][1]
        assert body["segments"] == []
        # An empty source with empty segments is what the route treats as "the
        # worker should transcribe", which is exactly right.
        assert body["transcript_source"] == ""


class TestLeaseBudget:
    def test_the_download_gets_the_lease_minus_a_reserve(self, job, posted, monkeypatch):
        seen: list[float] = []

        def fake_download(_url, stage_dir, file_id, *, timeout_s, **_kw):
            seen.append(timeout_s)
            path = Path(stage_dir) / f"{file_id}.mp4"
            path.write_bytes(b"v")
            return Downloaded(path=path)

        monkeypatch.setattr(fetcher, "probe_url", lambda *_a, **_k: _info())
        monkeypatch.setattr(fetcher, "download", fake_download)
        _run(job)
        # The reserve is what the result post is paid for. A job that spends
        # its last second downloading has nowhere to put the answer.
        assert 1200 - fetcher.LEASE_RESERVE_S - 5 < seen[0] <= 1200 - fetcher.LEASE_RESERVE_S

    def test_a_claim_without_a_lease_length_still_runs(self, job, posted, monkeypatch):
        job.pop("lease_seconds")
        _stub(monkeypatch)
        _run(job)
        # An absent field is allowed to be absent. Treating it as zero would
        # refuse every job on an older Audrey rather than degrading.
        assert posted[-1][0].endswith("/result")


class TestLoop:
    def test_an_empty_queue_is_not_an_error(self, monkeypatch):
        monkeypatch.setattr(fetcher, "post", lambda *_a, **_k: (204, {}))
        assert fetcher.run(endpoint="http://a", token=TOKEN, poll_seconds=0, once=True) == 0

    def test_a_rejected_claim_is_reported_and_does_not_crash(self, monkeypatch):
        monkeypatch.setattr(fetcher, "post", lambda *_a, **_k: (401, {"detail": "nope"}))
        assert fetcher.run(endpoint="http://a", token=TOKEN, poll_seconds=0, once=True) == 1

    def test_audrey_being_down_is_survivable(self, monkeypatch):
        def boom(*_a, **_k):
            raise OSError("connection refused")

        monkeypatch.setattr(fetcher, "post", boom)
        calls: list[float] = []
        monkeypatch.setattr(
            fetcher.Stopping, "wait",
            lambda self, seconds, **_kw: (calls.append(seconds), setattr(self, "requested", True)),
        )
        # A restart of audrey-ai is normal. The fetcher waits and polls again
        # rather than exiting into a restart loop.
        assert fetcher.run(endpoint="http://a", token=TOKEN, poll_seconds=7) == 0
        assert calls == [7]

    def test_a_claimed_job_is_handled_then_the_loop_exits_on_once(self, monkeypatch):
        handled: list[dict] = []
        monkeypatch.setattr(fetcher, "post", lambda *_a, **_k: (200, {"file_id": "x"}))
        monkeypatch.setattr(
            fetcher, "handle_job", lambda job, **_kw: handled.append(job),
        )
        assert fetcher.run(endpoint="http://a", token=TOKEN, poll_seconds=0, once=True) == 0
        assert handled == [{"file_id": "x"}]


def test_the_fetcher_refuses_to_start_without_a_token(monkeypatch):
    monkeypatch.delenv("KB_SERVICE_TOKEN", raising=False)
    # Every claim would 401. Failing at startup says so once, instead of once
    # every ten seconds forever.
    assert fetcher.main() == 2
