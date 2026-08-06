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
from audrey.routes.files import FetchClaim

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
    """A claim built from `FetchClaim` itself, never hand-written.

    **This is the fix for the bug that shipped.** The fixture used to spell the
    payload out by hand, and when the claim dropped `dest_dir` for `stage_dir`
    it kept both — so every test here passed against a job shape the route had
    stopped sending, while `handle_job` still read the field that no longer
    existed. On the box that was `KeyError: 'dest_dir'` on the first claim.

    Two test files each testing one end against its own idea of the contract
    will agree with each other forever and with reality never. Deriving the
    payload from the model the route actually serialises makes a field that
    only one side knows about impossible to write down.
    """
    stage, _dest = dirs
    return FetchClaim(
        file_id=FILE_ID,
        user="a@b.c",
        source_url=URL,
        lease_id="lease-1",
        attempts=1,
        stage_dir=str(stage),
        lease_seconds=1200,
        max_bytes=2 * 1024**3,
        max_duration_s=7200,
    ).model_dump()


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
    def test_the_file_is_left_in_staging_for_audrey_to_move(
        self, job, dirs, monkeypatch,
    ):
        stage, dest = dirs
        staged_at_post: list[bool] = []

        def fake_post(_endpoint, path, _token, _body, **_kw):
            if path.endswith("/result"):
                staged_at_post.append((stage / f"{FILE_ID}.mp4").exists())
            return 200, {}

        monkeypatch.setattr(fetcher, "post", fake_post)
        _stub(monkeypatch)
        _run(job)

        # **This container never writes outside staging.** Audrey rebuilds the
        # staged path from the reported filename, re-derives everything it was
        # told, and does the rename itself — so the file is still here when the
        # result is posted, and the user's directory is untouched by this end.
        assert staged_at_post == [True]
        assert list(dest.iterdir()) == []

    def test_the_tidy_up_runs_after_the_handover(self, job, dirs, posted, monkeypatch):
        stage, _dest = dirs
        _stub(monkeypatch)
        _run(job)
        # In production the file is gone from staging by now because the result
        # route moved it. Here the post is faked, so the tidy-up collects it —
        # which is the correct behaviour for the case that really matters: a
        # rejected handover leaves bytes nobody owns.
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
        # The reported extension is the one Audrey rebuilds the staged path
        # from, so it has to follow the file rather than the metadata — they
        # differ exactly when the format selector fell through, and a wrong
        # extension makes the result route look for a file that is not there.
        assert posted[-1][1]["filename"].endswith(".mkv")
        assert list(dest.iterdir()) == []


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


class TestProgressReporting:
    """What the page learns while a download is running.

    Two things it could not know before: what the video is actually called,
    and how far along it is. The title is the one that matters — until it
    arrives the row shows a video id, so "is it fetching the one I meant?" is
    unanswerable for the whole download.
    """

    def _progress(self, posted) -> list[dict]:
        return [b for p, b in posted if p.endswith("/progress")]

    def test_the_title_is_sent_before_a_byte_is_downloaded(
        self, job, posted, monkeypatch,
    ):
        order: list[str] = []

        def fake_post(_endpoint, path, _token, body, **_kw):
            order.append("progress" if path.endswith("/progress") else "other")
            posted.append((path, body or {}))
            return 200, {}

        def fake_download(_url, stage_dir, file_id, **_kw):
            order.append("download")
            path = Path(stage_dir) / f"{file_id}.mp4"
            path.write_bytes(b"v")
            return Downloaded(path=path)

        monkeypatch.setattr(fetcher, "post", fake_post)
        monkeypatch.setattr(fetcher, "probe_url", lambda *_a, **_k: _info())
        monkeypatch.setattr(fetcher, "download", fake_download)
        _run(job)

        assert order[0] == "progress"
        assert order[1] == "download"
        assert self._progress(posted)[0]["title"] == "A Retirement Speech"

    def test_the_first_update_carries_the_size_estimate(self, job, posted, monkeypatch):
        _stub(monkeypatch)
        _run(job)
        first = self._progress(posted)[0]
        # So the very first poll has a denominator. The exact figures from
        # yt-dlp replace it a couple of seconds later.
        assert first["downloaded_bytes"] == 0
        assert first["total_bytes"] == 100 * 1024 * 1024

    def test_progress_from_the_downloader_is_forwarded(self, job, posted, monkeypatch):
        def fake_download(_url, stage_dir, file_id, *, on_progress=None, **_kw):
            on_progress(50, 200)
            path = Path(stage_dir) / f"{file_id}.mp4"
            path.write_bytes(b"v")
            return Downloaded(path=path)

        monkeypatch.setattr(fetcher, "probe_url", lambda *_a, **_k: _info())
        monkeypatch.setattr(fetcher, "download", fake_download)
        monkeypatch.setattr(fetcher, "PROGRESS_INTERVAL_S", 0.0)
        _run(job)
        assert (50, 200) == (
            self._progress(posted)[-1]["downloaded_bytes"],
            self._progress(posted)[-1]["total_bytes"],
        )

    def test_a_failed_progress_post_does_not_fail_the_download(
        self, job, monkeypatch,
    ):
        calls: list[str] = []

        def fake_post(_endpoint, path, _token, _body, **_kw):
            calls.append(path)
            if path.endswith("/progress"):
                return 409, {"detail": "reclaimed"}
            return 200, {}

        monkeypatch.setattr(fetcher, "post", fake_post)
        _stub(monkeypatch)
        _run(job)
        # The row keeps showing elapsed time, which is what it did before any
        # of this existed. Aborting a download over a display field would be
        # the tail wagging.
        assert calls[-1].endswith("/result")


class TestProgressReporter:
    """The accounting, tested directly — it is arithmetic with a trap in it."""

    def _reporter(self, sent, **kw):
        r = fetcher._ProgressReporter(
            "http://a", TOKEN,
            {"file_id": FILE_ID, "lease_id": "L1"}, "A Title", **kw,
        )
        r.send = lambda done, total: sent.append((done, total))
        return r

    def test_updates_are_throttled(self):
        sent: list[tuple[int, int]] = []
        r = self._reporter(sent, interval_s=1000.0)
        for i in range(1, 101):
            r(i, 100)
        # yt-dlp emits several times a second and the page polls every five, so
        # forwarding every one is a POST nobody sees and a sqlite write on the
        # connection every other request shares. The first goes straight
        # through, though — waiting out the interval before the first real byte
        # count would leave the row on elapsed time for no reason.
        assert sent == [(1, 100)]

    def test_an_unthrottled_reporter_sends_everything(self):
        sent: list[tuple[int, int]] = []
        r = self._reporter(sent, interval_s=0.0)
        r(10, 100)
        r(20, 100)
        assert sent == [(10, 100), (20, 100)]

    def test_a_second_stream_does_not_restart_the_count(self):
        sent: list[tuple[int, int]] = []
        r = self._reporter(sent, interval_s=0.0)
        r(90, 100)
        r(100, 100)
        # Audio begins; yt-dlp counts it from zero again.
        r(2, 10)
        r(8, 10)
        # Naively forwarded this reads 100%, then 20%, then 80% — a download
        # that appears to have restarted. Folding the finished stream in keeps
        # the numerator climbing.
        assert [d for d, _t in sent] == [90, 100, 102, 108]

    def test_the_denominator_grows_rather_than_the_percentage_falling(self):
        sent: list[tuple[int, int]] = []
        r = self._reporter(sent, interval_s=0.0)
        r(100, 100)
        r(5, 10)
        # The job really did get bigger when the second stream's size became
        # known. Showing that in the total is honest; showing it as a
        # percentage dropping from 100% would not be.
        assert sent == [(100, 100), (105, 110)]

    def test_a_stream_with_no_total_still_counts_upwards(self):
        sent: list[tuple[int, int]] = []
        r = self._reporter(sent, interval_s=0.0)
        r(50, None)
        r(120, None)
        r(10, None)     # a second file, size still unknown
        assert [d for d, _t in sent] == [50, 120, 130]
        # No denominator anywhere, so none is invented.
        assert {t for _d, t in sent} == {0}

    def test_one_unknown_stream_makes_the_whole_total_unknown(self):
        sent: list[tuple[int, int]] = []
        r = self._reporter(sent, interval_s=0.0)
        r(50, None)     # a stream that would not say how big it is
        r(10, 40)       # a second one that would
        # Summing only the parts we know gives a denominator the numerator
        # will overtake — "100% (60 of 40)", still downloading. Better to have
        # no percentage than one that says the download finished twice.
        assert sent == [(50, 0), (60, 0)]


class TestABugDoesNotBecomeSilence:
    """What a crash in this container looks like from the outside.

    Written after `KeyError: 'dest_dir'` shipped and cost an afternoon. The
    KeyError itself is prevented by the `job` fixture now being built from
    `FetchClaim`; this is about the *second* failure, which was worse and is
    the general one: an unhandled exception killed the process, so the row sat
    in 'fetching' with nobody holding it until the lease expired, was
    re-queued, and crashed identically on the next claim.

    From the upload page that is a download that has been running for an hour,
    indistinguishable from a slow one, with the only evidence in a container
    log nobody has reason to open.
    """

    def _harness(self, monkeypatch, boom, *, jobs: int = 2):
        """Hand out `jobs` claims, then go idle and stop."""
        posted: list[tuple[str, dict]] = []
        claims = [0]

        def fake_post(_endpoint, path, _token, body, **_kw):
            posted.append((path, body or {}))
            if path.endswith("/claim"):
                claims[0] += 1
                if claims[0] > jobs:
                    return 204, {}
                return 200, {"file_id": "x", "lease_id": "L1"}
            return 200, {}

        monkeypatch.setattr(fetcher, "post", fake_post)
        monkeypatch.setattr(fetcher, "handle_job", boom)
        # The idle wait is where this loop ends. Without it the 204 branch
        # spins, which is what a no-op `wait` turns a poll loop into.
        monkeypatch.setattr(
            fetcher.Stopping, "wait",
            lambda self, _s, **_k: setattr(self, "requested", True),
        )
        return posted

    def test_an_unexpected_error_does_not_kill_the_loop(self, monkeypatch):
        def boom(_job, **_kw):
            raise KeyError("dest_dir")

        posted = self._harness(monkeypatch, boom, jobs=2)
        assert fetcher.run(endpoint="http://a", token=TOKEN, poll_seconds=0) == 0
        # It took the second job too, rather than exiting into a restart that
        # repeats the same crash twenty minutes later.
        assert len([p for p, _b in posted if p.endswith("/claim")]) == 3

    def test_the_row_is_told_rather_than_left_to_time_out(self, monkeypatch):
        def boom(_job, **_kw):
            raise KeyError("dest_dir")

        posted = self._harness(monkeypatch, boom, jobs=1)
        fetcher.run(endpoint="http://a", token=TOKEN, poll_seconds=0)

        failed = [b for p, b in posted if p.endswith("/failed")]
        # A sentence on the row beats an hour of silence. It costs a retry that
        # was going to fail anyway — max_attempts is 3 — and it is the
        # difference between "this is broken" and "is this stalled?".
        assert len(failed) == 1
        assert "internal error" in failed[0]["reason"]

    def test_a_reporting_failure_does_not_resurrect_the_crash(self, monkeypatch):
        def boom(_job, **_kw):
            raise KeyError("dest_dir")

        posted: list[str] = []

        def fake_post(_endpoint, path, _token, _body, **_kw):
            posted.append(path)
            if path.endswith("/failed"):
                raise OSError("audrey is restarting")
            if path.endswith("/claim"):
                return (200, {"file_id": "x", "lease_id": "L1"}) \
                    if len([p for p in posted if p.endswith("/claim")]) == 1 else (204, {})
            return 200, {}

        monkeypatch.setattr(fetcher, "post", fake_post)
        monkeypatch.setattr(fetcher, "handle_job", boom)
        monkeypatch.setattr(
            fetcher.Stopping, "wait",
            lambda self, _s, **_k: setattr(self, "requested", True),
        )
        # Telling Audrey about the bug can itself fail. Letting that propagate
        # would put the crash-loop back, one layer out.
        assert fetcher.run(endpoint="http://a", token=TOKEN, poll_seconds=0) == 0

    def test_a_broken_image_still_crashes_loudly(self, monkeypatch):
        def missing(_job, **_kw):
            raise YtDlpMissingError("yt-dlp is not on PATH")

        self._harness(monkeypatch, missing, jobs=1)
        # Not caught by the blanket handler: an image built wrong is fixed by
        # rebuilding, and a container restart-looping in `docker ps` is the
        # right signal. Failing every queued URL on the way past is not.
        with pytest.raises(YtDlpMissingError):
            fetcher.run(endpoint="http://a", token=TOKEN, poll_seconds=0)


CLIENTS = ["youtube:player_client=android_vr", "youtube:player_client=tv", ""]


class TestClientFallback:
    """Walking the client list when YouTube stops serving one.

    The point is not the retry, it is *when* it retries. YouTube decides which
    download clients it will serve and changes that on its own schedule; when
    it does, every fetch fails at once and stays failed until someone edits
    config. A list turns that outage into a slower first download.

    What makes it affordable is the discrimination: a private video is private
    to every client, and walking the list would spend a lease learning that six
    times over — six times the requests, from a server that would rather not
    look like something worth blocking.
    """

    @pytest.fixture(autouse=True)
    def _no_waiting(self, monkeypatch):
        monkeypatch.setattr(fetcher.time, "sleep", lambda _s: None)

    def _tries(self, monkeypatch, outcomes: dict[str, Exception | None]):
        """Fail or succeed per client, recording the order they were tried."""
        tried: list[str] = []

        def fake_probe(_url, *, extractor_args="", **_kw):
            tried.append(extractor_args)
            problem = outcomes.get(extractor_args)
            if problem is not None:
                raise problem
            return _info()

        def fake_download(_url, stage_dir, file_id, *, extractor_args="", **_kw):
            problem = outcomes.get(extractor_args)
            if problem is not None:
                raise problem
            path = Path(stage_dir) / f"{file_id}.mp4"
            path.write_bytes(b"video bytes")
            return Downloaded(path=path)

        monkeypatch.setattr(fetcher, "probe_url", fake_probe)
        monkeypatch.setattr(fetcher, "download", fake_download)
        return tried

    def test_a_403_moves_on_to_the_next_client(self, job, posted, monkeypatch):
        job["extractor_args"] = CLIENTS
        tried = self._tries(monkeypatch, {
            CLIENTS[0]: FetchRefusedError("403", client_related=True),
        })
        _run(job)
        assert tried == [CLIENTS[0], CLIENTS[1]]
        assert posted[-1][0].endswith("/result")

    def test_a_private_video_stops_after_one_client(self, job, posted, monkeypatch):
        job["extractor_args"] = CLIENTS
        tried = self._tries(monkeypatch, {
            c: FetchRefusedError("this video is private") for c in CLIENTS
        })
        _run(job)
        # Every other client would say the same thing, more slowly and with
        # three times the requests.
        assert tried == [CLIENTS[0]]
        assert "private" in posted[-1][1]["reason"]

    def test_our_own_failures_do_not_walk_the_list_either(self, job, posted, monkeypatch):
        job["extractor_args"] = CLIENTS
        tried = self._tries(monkeypatch, {
            c: FetchFailedError("the download did not finish in time") for c in CLIENTS
        })
        _run(job)
        # A timeout is ours, not theirs. Another client is not the answer, and
        # the lease has already been spent once.
        assert tried == [CLIENTS[0]]

    def test_exhausting_every_client_says_so(self, job, posted, monkeypatch):
        job["extractor_args"] = CLIENTS
        job["max_client_attempts"] = len(CLIENTS)
        self._tries(monkeypatch, {
            c: FetchRefusedError("403", client_related=True) for c in CLIENTS
        })
        _run(job)
        reason = posted[-1][1]["reason"]
        # The last client's message alone reads as a fact about the video. The
        # news is that every way in was refused, which is a different thing to
        # go and fix.
        assert "tried 3 download clients" in reason
        assert "server-side problem" in reason

    def test_staging_is_cleared_between_attempts(self, job, dirs, posted, monkeypatch):
        stage, _dest = dirs
        job["extractor_args"] = CLIENTS
        leftover = stage / f"{FILE_ID}.f137.mp4.part"
        leftover.write_bytes(b"half a download from the failed client")
        self._tries(monkeypatch, {
            CLIENTS[0]: FetchRefusedError("403", client_related=True),
        })
        _run(job)
        # A retry re-downloads from scratch. A stale `.part` under the same
        # file_id would be picked up by the caption glob or confuse the size
        # check on the way out.
        assert not leftover.exists()

    def test_one_client_configured_behaves_as_before(self, job, posted, monkeypatch):
        job["extractor_args"] = [""]
        tried = self._tries(monkeypatch, {})
        _run(job)
        assert tried == [""]
        assert posted[-1][0].endswith("/result")


class TestClientRing:
    def test_the_winner_is_tried_first_next_time(self):
        ring = fetcher.ClientRing(["a", "b", "c"])
        ring.succeeded("c")
        assert ring.candidates() == ["c", "a", "b"]

    def test_a_winner_already_first_is_left_alone(self):
        ring = fetcher.ClientRing(["a", "b"])
        ring.succeeded("a")
        assert ring.candidates() == ["a", "b"]

    def test_an_empty_list_still_tries_once(self):
        # `[""]` is yt-dlp's own default. A downloader configured to try
        # nothing is not a safe default, it is a broken one.
        assert fetcher.ClientRing([]).candidates() == [""]

    def test_the_preference_survives_between_jobs(self):
        # Not between restarts: losing it costs one wasted attempt on the next
        # download, and persisting it would cost a file, a format, and a way
        # for it to go stale.
        assert fetcher._reordered(["a", "b", "c"], "b") == ["b", "a", "c"]
        assert fetcher._reordered(["a", "b"], "gone") == ["a", "b"]
        assert fetcher._reordered(["a", "b"], None) == ["a", "b"]


class TestFallbackDepthIsBounded:
    """How many clients one job may try, and why the number is small.

    Every extra client is another request made at the moment something is
    already wrong. The list exists to survive YouTube retiring a client, not to
    sweep every option — and most of the configured entries were measured dead
    the day they were written, so a deep sweep is mostly requests we already
    know will fail.
    """

    @pytest.fixture(autouse=True)
    def _no_waiting(self, monkeypatch):
        monkeypatch.setattr(fetcher.time, "sleep", lambda _s: None)

    def _all_refuse(self, monkeypatch):
        tried: list[str] = []

        def fake_probe(_url, *, extractor_args="", **_kw):
            tried.append(extractor_args)
            raise FetchRefusedError("403", client_related=True)

        monkeypatch.setattr(fetcher, "probe_url", fake_probe)
        return tried

    def test_the_default_depth_stops_at_two(self, job, posted, monkeypatch):
        job["extractor_args"] = CLIENTS
        job["max_client_attempts"] = 2
        tried = self._all_refuse(monkeypatch)
        _run(job)
        assert len(tried) == 2

    def test_one_disables_fallback_entirely(self, job, posted, monkeypatch):
        job["extractor_args"] = CLIENTS
        job["max_client_attempts"] = 1
        tried = self._all_refuse(monkeypatch)
        _run(job)
        # A reasonable setting, not a degraded one: one attempt, a clear reason
        # on the row, and a human edits the config when YouTube moves.
        assert tried == [CLIENTS[0]]

    def test_a_missing_or_zero_depth_never_means_zero_attempts(
        self, job, posted, monkeypatch,
    ):
        job["extractor_args"] = CLIENTS
        job["max_client_attempts"] = 0
        tried = self._all_refuse(monkeypatch)
        _run(job)
        # A downloader configured to try nothing is not a safe default; it is
        # a broken one that reports every video as unavailable.
        assert len(tried) == 1

    def test_the_reason_counts_what_was_actually_tried(self, job, posted, monkeypatch):
        job["extractor_args"] = CLIENTS
        job["max_client_attempts"] = 2
        self._all_refuse(monkeypatch)
        _run(job)
        # Not "tried 3" when the depth allowed 2 — the message is the evidence
        # someone will act on.
        assert "tried 2 download clients" in posted[-1][1]["reason"]
