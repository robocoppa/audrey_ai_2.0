"""yt-dlp: metadata, download, and caption parsing (Phase 41, steps 2 and 4).

Separated from `fetcher.py` for the same reason `audio.py` is separated from
`worker.py` — so the parts with judgement in them can be tested without a
container, a network, or a real video. The loop is plumbing; the judgement is
here, and there are three pieces of it:

  - **What to refuse before downloading.** Duration and estimated size come
    from a metadata-only pass, which costs one request against a download that
    could cost two gigabytes.
  - **What to tell the user when it fails.** `friendly_reason` is the
    deliverable of step 3, not a nicety: "private video", "members only" and
    "region blocked" are the *common* cases, and a generic "download failed"
    is the message that generates the support question the whole feature
    exists to prevent.
  - **Which transcript is real.** Manual subtitles are human-authored and
    routinely better than whisper; auto-captions are neither, but still arrive
    in seconds against whisper's minutes. They are different enough that the
    row records which one it got — "the transcript is wrong" has completely
    different answers for the two, and after the fact nothing else can tell
    them apart.

Nothing here imports from `audrey`. Stdlib plus one subprocess.
"""

from __future__ import annotations

import html
import json
import logging
import re
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_PROBE_TIMEOUT_S = 120

#: Cap the download at 720p rather than taking the best available, and prefer
#: streams that are already mp4.
#:
#: **720p** because the visual pass downscales to `kb.video.frame_max_width`
#: (1280) before a frame ever reaches the vision model, so anything above it is
#: bytes and download seconds spent on pixels that get thrown away — and the
#: transcript, which is what most questions are actually about, does not depend
#: on the video track at all. 1080p roughly doubles the download for no
#: downstream difference.
#:
#: **mp4** because `ALLOWED_VIDEO_MIMES` is exactly `{"video/mp4"}`, and a
#: webm arriving at `fetch/{id}/result` is refused by the same libmagic gate
#: that stops an HTML error page — correctly, but for a reason that would read
#: as "the download is broken" rather than "this server stores mp4". The
#: preference here plus `--remux-video` below means that never comes up.
#:
#: The `+ba` branch merges separate video and audio streams, which is the
#: reason the image installs ffmpeg. The single-file fallbacks after it are
#: what a site without adaptive streams provides.
DEFAULT_FORMAT = (
    "bv*[height<=720][ext=mp4]+ba[ext=m4a]/"
    "b[height<=720][ext=mp4]/"
    "bv*[height<=720]+ba/"
    "b[height<=720]/b"
)

#: The container everything ends up in. `--merge-output-format` covers the
#: merge branches above; `--remux-video` covers the single-file fallbacks,
#: where the site served one file and it was not mp4. Both are container
#: changes — a stream copy, not a re-encode — so the cost is bounded by disk
#: rather than by CPU.
OUTPUT_CONTAINER = "mp4"

#: Caption languages to ask for, in preference order. `en.*` catches `en-US`
#: and `en-GB`; the bare `en` catches the plain tag.
DEFAULT_SUB_LANGS = "en,en.*"

#: How the transcript was produced. Stored on the row and shown in the file
#: list — see this module's docstring for why the distinction is kept.
SOURCE_SUBTITLES = "subtitles"
SOURCE_AUTO_CAPTIONS = "auto_captions"
SOURCE_WHISPER = "whisper"


class YtDlpMissingError(RuntimeError):
    """yt-dlp is not on PATH — an image defect, not a problem with this URL.

    Propagated rather than reported as a failed row, exactly as
    `FFmpegMissingError` is in the worker: failing rows for an image that is
    built wrong burns every queued fetch's attempts, and they would all be
    'failed' by the time anyone noticed.
    """


class FetchRefusedError(RuntimeError):
    """A fact about the video that means we will not download it.

    Too long, too large, or a site saying no. The message is shown to the user
    verbatim, so it is written for them rather than for a log.

    `client_related` says whether trying a *different* download client could
    plausibly succeed. It is the difference between a retry that might work and
    six identical failures: a private video is private to every client, while a
    403 on the media URL is entirely about which client asked.
    """

    def __init__(self, message: str, *, client_related: bool = False) -> None:
        super().__init__(message)
        self.client_related = client_related


class FetchFailedError(RuntimeError):
    """yt-dlp ran and could not produce the file."""


@dataclass(frozen=True)
class UrlInfo:
    """What the metadata pass learned, before a byte of media is downloaded."""

    title: str
    ext: str
    duration_s: float
    filesize_approx: int
    is_live: bool = False
    #: Language tags with human-authored subtitles.
    subtitle_langs: tuple[str, ...] = ()
    #: Language tags with machine-generated captions.
    auto_caption_langs: tuple[str, ...] = ()
    #: Chapter boundaries, if the uploader marked any. Carried but not yet
    #: used — see `docs/campaign-2/phase-41-url-video-ingest.md`.
    chapters: tuple[dict, ...] = ()

    def caption_choice(self, langs: str = DEFAULT_SUB_LANGS) -> str:
        """Which caption track to ask for: manual, automatic, or neither.

        Returns one of `SOURCE_SUBTITLES`, `SOURCE_AUTO_CAPTIONS` or `""`.

        Decided from *metadata* rather than from what lands on disk, because
        `--write-subs` and `--write-auto-subs` produce files with identical
        names — `<stem>.en.vtt` either way. Asking for both and then guessing
        which one arrived is how a row ends up claiming a human wrote its
        auto-captions.
        """
        wanted = _lang_matchers(langs)
        if any(_lang_matches(tag, wanted) for tag in self.subtitle_langs):
            return SOURCE_SUBTITLES
        if any(_lang_matches(tag, wanted) for tag in self.auto_caption_langs):
            return SOURCE_AUTO_CAPTIONS
        return ""


@dataclass
class Downloaded:
    """Where the media landed and what came with it."""

    path: Path
    #: Parsed caption track, in `IngestResultRequest.segments` shape. Empty
    #: when there were none to fetch or parsing produced nothing usable.
    segments: list[dict] = field(default_factory=list)
    #: Which of the two produced `segments`, or "" when whisper will have to.
    transcript_source: str = ""


def _binary(name: str = "yt-dlp") -> str:
    found = shutil.which(name)
    if found is None:
        raise YtDlpMissingError(
            f"{name} is not on PATH — the media-fetcher image is built wrong",
        )
    return found


#: Prefix on the progress lines we ask yt-dlp to emit. Progress and
#: `--print after_move:filepath` both go to stdout, so one of them has to be
#: recognisable; tagging the one whose format we control is the cheaper half.
_PROGRESS_TAG = "@AUDREYP"

#: What a progress line carries. `total_bytes` is exact and present for most
#: downloads; `total_bytes_estimate` is what a site that will not commit to a
#: size gives instead. yt-dlp renders an unknown field as the string `NA`.
_PROGRESS_TEMPLATE = (
    f"download:{_PROGRESS_TAG} %(progress.downloaded_bytes)s "
    "%(progress.total_bytes)s %(progress.total_bytes_estimate)s"
)


def _int_or_none(token: str) -> int | None:
    """yt-dlp writes `NA` for a field it does not know. So does an empty run."""
    try:
        value = int(float(token))
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def parse_progress_line(line: str) -> tuple[int, int | None] | None:
    """`(downloaded_bytes, total_or_None)` from one progress line, or None.

    Split out and given a name because it is the only part of the streaming
    path with a decision in it, and because a yt-dlp upgrade that changes the
    field rendering should fail a test here rather than silently produce a
    download that reports 0 bytes forever.
    """
    if not line.startswith(_PROGRESS_TAG):
        return None
    parts = line.split()[1:]
    if not parts:
        return None
    downloaded = _int_or_none(parts[0])
    if downloaded is None:
        return None
    total = _int_or_none(parts[1]) if len(parts) > 1 else None
    if total is None and len(parts) > 2:
        # Fall back to the estimate. A number that is roughly right beats no
        # denominator at all — a progress display with no total can only count
        # upwards, which does not answer "how much longer".
        total = _int_or_none(parts[2])
    return downloaded, total


def _stream(
    argv: list[str], *, timeout: float,
    on_progress: Callable[[int, int | None], None] | None,
) -> tuple[int, list[str], str]:
    """Run yt-dlp, feeding progress out as it arrives. Returns
    `(returncode, printed_lines, stderr)`.

    `Popen` rather than `subprocess.run` because progress that arrives after
    the process exits is not progress. Two things this has to get right that
    `run` was doing for us:

    **stderr goes to a file, not a pipe.** Reading stdout line by line while
    stderr fills its own 64 KB pipe buffer deadlocks — the classic one. A
    temporary file has no buffer to fill.

    **The timeout is a watchdog, not a read deadline.** `readline` blocks, so a
    yt-dlp that hangs with no output would sit past any deadline checked
    between lines. A timer that kills the process turns that into an EOF we are
    already waiting for.
    """
    killed = False

    def expire() -> None:
        nonlocal killed
        killed = True
        proc.kill()

    printed: list[str] = []
    with tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace") as errfile:
        proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=errfile,
            text=True, bufsize=1,
        )
        watchdog = threading.Timer(timeout, expire)
        watchdog.start()
        try:
            for raw in proc.stdout or ():
                line = raw.strip()
                if not line:
                    continue
                progress = parse_progress_line(line)
                if progress is None:
                    printed.append(line)
                elif on_progress is not None:
                    # A reporter that throws must not take the download with
                    # it. Progress is a courtesy; the bytes are the job.
                    try:
                        on_progress(*progress)
                    except Exception as e:  # noqa: BLE001
                        log.warning("fetch: progress callback failed: %s", e)
            proc.wait()
        finally:
            watchdog.cancel()
            if proc.stdout is not None:
                proc.stdout.close()
        errfile.seek(0)
        stderr = errfile.read()

    if killed:
        raise FetchFailedError(
            f"the download did not finish within {timeout:.0f}s and was stopped",
        )
    return proc.returncode, printed, stderr


def _run(argv: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    # `argv` is a list with an absolute binary path from `shutil.which` and no
    # shell. The URL is caller-supplied but is passed after `--`, so it cannot
    # be read as an option however it is spelled.
    try:
        return subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise FetchFailedError(
            f"the download did not finish within {timeout:.0f}s and was stopped",
        ) from e


#: yt-dlp's stderr, mapped to something a person can act on. Matched
#: case-insensitively against the whole of stderr, first hit wins, so the more
#: specific patterns come first.
#:
#: These are the COMMON cases, not edge cases. Every one of them was chosen
#: because the honest answer differs: "wait and retry" is right for a rate
#: limit and wrong for a deleted video, and "ask the owner" is right for a
#: private one and useless for a region block.
_REASON_MAP: tuple[tuple[str, str], ...] = (
    # First, because it is the one message that is about US and reads as
    # though it is about the video. YouTube says "not available on this app"
    # when it rejects the client an out-of-date yt-dlp impersonates — nothing
    # is wrong with the video, and passing the wording through sends whoever
    # reads it to check the link instead of the downloader. Observed on the
    # first real fetch this feature ever did, against a pin thirteen months
    # stale; see `docker/media-fetcher.Dockerfile`.
    (
        "not available on this app",
        "the server's downloader is out of date — YouTube refused the version "
        "it identifies as. Nothing is wrong with this video; media-fetcher "
        "needs its yt-dlp bumped and the image rebuilt",
    ),
    # Second, and for the same reason: a 403 on the *media* URL after the
    # metadata pass succeeded is not the video refusing you, it is YouTube
    # refusing the download client. Observed once the yt-dlp pin was current,
    # which is the tell — an out-of-date downloader fails earlier, at the
    # metadata pass, with the message above. The fix is `kb.fetch.extractor_args`
    # and it is a config edit plus a restart, not a rebuild.
    (
        "http error 403",
        "YouTube refused to serve the video data to the download client. This "
        "is a server-side setting, not a problem with the link: "
        "`kb.fetch.extractor_args` needs a player client YouTube currently "
        "accepts",
    ),
    ("private video", "this video is private — only its owner can see it"),
    (
        "members-only",
        "this video is for channel members only, and the server has no account",
    ),
    (
        "available to this channel's members",
        "this video is for channel members only, and the server has no account",
    ),
    (
        "join this channel",
        "this video is for channel members only, and the server has no account",
    ),
    (
        "sign in to confirm your age",
        "this video is age-restricted and the server cannot sign in to confirm an age",
    ),
    (
        "age-restricted",
        "this video is age-restricted and the server cannot sign in to confirm an age",
    ),
    (
        "available in your country",
        "this video is blocked in the server's region",
    ),
    (
        "blocked it in your country",
        "this video is blocked in the server's region",
    ),
    (
        "who has blocked it on copyright grounds",
        "this video is blocked on copyright grounds",
    ),
    (
        "this live event will begin",
        "this is a scheduled live stream that has not started yet",
    ),
    (
        "is live",
        "this is a live stream — it has no end, so there is nothing to finish "
        "downloading. Try again once it has been published as a recording",
    ),
    (
        "sign in to confirm you're not a bot",
        "the site asked the server to prove it is not a bot, which it cannot do",
    ),
    (
        "http error 429",
        "the site is rate-limiting this server — try again later",
    ),
    (
        "video unavailable",
        "the video is unavailable — it may have been deleted or made private",
    ),
    (
        "removed by the uploader",
        "the video was removed by the uploader",
    ),
    (
        "account associated with this video has been terminated",
        "the account that posted this video has been terminated",
    ),
    (
        "unsupported url",
        "this server does not know how to download from that link",
    ),
    (
        "unable to download webpage",
        "the site could not be reached from this server",
    ),
)


#: Failures where a *different* download client could plausibly succeed.
#:
#: The discrimination matters more than the list. Without it, a private video
#: would be attempted once per configured client — six identical refusals, six
#: times the requests at YouTube, and a lease spent proving something the first
#: attempt already knew. Everything here is a property of who asked; everything
#: absent from it is a property of the video.
_CLIENT_FAILURE_NEEDLES: tuple[str, ...] = (
    "http error 403",
    "not available on this app",
    "sign in to confirm you're not a bot",
    "unable to download video data",
    "failed to extract any player response",
    "nsig extraction failed",
    # A client that does not offer the formats we asked for. Not a fact about
    # the video — the next client lists a different set.
    "requested format is not available",
    # Age gates are enforced per client, so this is genuinely worth another
    # attempt even though it reads like a property of the video.
    "sign in to confirm your age",
    "age-restricted",
)


def is_client_failure(stderr: str) -> bool:
    """Could another download client plausibly get past this?"""
    lowered = (stderr or "").lower()
    return any(needle in lowered for needle in _CLIENT_FAILURE_NEEDLES)


def friendly_reason(stderr: str, *, limit: int = 300) -> str:
    """Turn yt-dlp's stderr into a sentence the user can act on.

    **Unmatched output is passed through, not replaced.** A generic "download
    failed" is precisely the message this exists to avoid, and the failure
    modes yt-dlp has are not a closed set — a new one should reach the user as
    whatever yt-dlp said about it, truncated, rather than being flattened into
    a message that tells them nothing. The mapped cases are the ones where
    yt-dlp's own wording is worse than a sentence written for a person.
    """
    text = (stderr or "").strip()
    if not text:
        return "the download failed and the downloader said nothing about why"

    lowered = text.lower()
    for needle, message in _REASON_MAP:
        if needle in lowered:
            return message

    # No match: hand back what it actually said. The last ERROR line is where
    # yt-dlp puts the reason, after any number of warnings.
    errors = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("ERROR:")]
    line = errors[-1] if errors else text.splitlines()[-1].strip()
    line = line.removeprefix("ERROR:").strip()
    return line[:limit]


def _extractor_argv(extractor_args: str) -> list[str]:
    """`--extractor-args` if configured, nothing if not.

    Config-driven and empty by default, which leaves yt-dlp's own client
    selection alone. It exists because **YouTube's answer to "which client may
    download this" changes on YouTube's schedule**, and the symptom when it
    changes is a 403 on the media URL *after* a metadata pass that worked
    perfectly — see `kb.fetch.extractor_args` in `config.yaml` for how to find
    the value that works today.

    Not hardcoded to a client that looked right at the time: that is the
    mistake the yt-dlp pin already made once in this phase.
    """
    value = (extractor_args or "").strip()
    return ["--extractor-args", value] if value else []


def probe_url(
    url: str, *, timeout_s: float = DEFAULT_PROBE_TIMEOUT_S,
    binary: str | None = None, extractor_args: str = "",
) -> UrlInfo:
    """Read a URL's metadata without downloading the media.

    One `-J` call rather than several `--print` passes: it costs the same
    request, and it is the only way to learn what caption tracks exist, which
    step 4 needs before it can ask for the right one.
    """
    argv = [
        binary or _binary(),
        "--no-playlist",       # a link with `&list=` must fetch one video, not 200
        "--no-warnings",
        "--skip-download",
        "-J",
        *_extractor_argv(extractor_args),
        "--",
        url,
    ]
    result = _run(argv, timeout=timeout_s)
    if result.returncode != 0:
        raise FetchRefusedError(
            friendly_reason(result.stderr),
            client_related=is_client_failure(result.stderr),
        )

    try:
        meta = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as e:
        raise FetchFailedError(f"the downloader returned unreadable metadata: {e}") from e

    # A playlist URL that got past `--no-playlist` still answers with entries.
    # Take the first, which is what the user pointed at.
    if meta.get("_type") == "playlist":
        entries = [e for e in (meta.get("entries") or []) if isinstance(e, dict)]
        if not entries:
            raise FetchRefusedError("that link has no video on it")
        meta = entries[0]

    return UrlInfo(
        title=str(meta.get("title") or "").strip(),
        ext=str(meta.get("ext") or "").strip(),
        duration_s=float(meta.get("duration") or 0.0),
        filesize_approx=int(meta.get("filesize_approx") or meta.get("filesize") or 0),
        is_live=bool(meta.get("is_live") or meta.get("live_status") == "is_live"),
        subtitle_langs=tuple(sorted((meta.get("subtitles") or {}).keys())),
        auto_caption_langs=tuple(sorted((meta.get("automatic_captions") or {}).keys())),
        chapters=tuple(c for c in (meta.get("chapters") or []) if isinstance(c, dict)),
    )


def check_limits(
    info: UrlInfo, *, max_duration_s: float = 0.0, max_bytes: int = 0,
) -> None:
    """Raise `FetchRefusedError` if this video is one we will not download.

    Runs against metadata, before any media is transferred. `--max-filesize`
    on the download is the backstop for a missing or lying estimate; this is
    the check that means a six-hour stream costs one request instead of an
    hour of bandwidth and a swept lease.
    """
    if info.is_live:
        raise FetchRefusedError(
            "this is a live stream — it has no end, so there is nothing to "
            "finish downloading. Try again once it has been published as a "
            "recording",
        )
    if max_duration_s and info.duration_s > max_duration_s:
        raise FetchRefusedError(
            f"the video is {info.duration_s / 60:.0f} minutes long and this "
            f"server refuses anything over {max_duration_s / 60:.0f} minutes",
        )
    if max_bytes and info.filesize_approx > max_bytes:
        raise FetchRefusedError(
            f"the video is about {info.filesize_approx // (1024 * 1024)}MB and "
            f"this server refuses anything over {max_bytes // (1024 * 1024)}MB",
        )


def download(
    url: str,
    stage_dir: Path,
    file_id: str,
    *,
    timeout_s: float,
    max_bytes: int = 0,
    caption_source: str = "",
    sub_langs: str = DEFAULT_SUB_LANGS,
    fmt: str = DEFAULT_FORMAT,
    binary: str | None = None,
    extractor_args: str = "",
    on_progress: Callable[[int, int | None], None] | None = None,
) -> Downloaded:
    """Download into the staging directory. Returns where it landed.

    **Never writes to the final path.** The output template is
    `<stage_dir>/<file_id>.%(ext)s`, and the caller renames into place only
    once the file is complete — a partial download sitting at the path the row
    implies is a file the media worker will claim and cheerfully transcribe.

    The extension is yt-dlp's to choose, which is why the caller is handed a
    path rather than told one: the container depends on what the site serves
    and what the format selector merged into.

    `on_progress(downloaded_bytes, total_or_None)` is called as the bytes
    arrive, several times a second. **Throttling is the caller's job** — this
    end reports what yt-dlp reports, and how often that is worth forwarding
    over HTTP is a question about the caller's transport, not about the
    download.
    """
    stage_dir.mkdir(parents=True, exist_ok=True)
    argv = [
        binary or _binary(),
        "--no-playlist",
        "--no-warnings",
        "--no-continue",       # a stale .part from a swept attempt must not be resumed
        "-f", fmt,
        "--merge-output-format", OUTPUT_CONTAINER,
        "--remux-video", OUTPUT_CONTAINER,
        *_extractor_argv(extractor_args),
        "-o", str(stage_dir / f"{file_id}.%(ext)s"),
        # The one line of stdout we read: where the finished file actually is,
        # after any merge and rename. Deriving it from the metadata `ext`
        # instead would be a guess that is wrong exactly when the format
        # selector fell through to a different container.
        "--print", "after_move:filepath",
    ]
    if on_progress is None:
        argv += ["--no-progress"]
    else:
        # `--newline` matters as much as the template: yt-dlp redraws progress
        # with a carriage return by default, so without it the whole download
        # is one line that never ends and `for line in stdout` yields nothing
        # until the process exits.
        argv += ["--newline", "--progress-template", _PROGRESS_TEMPLATE]
    if max_bytes:
        argv += ["--max-filesize", str(max_bytes)]
    if caption_source == SOURCE_SUBTITLES:
        argv += ["--write-subs", "--sub-langs", sub_langs, "--sub-format", "vtt/best"]
    elif caption_source == SOURCE_AUTO_CAPTIONS:
        argv += ["--write-auto-subs", "--sub-langs", sub_langs, "--sub-format", "vtt/best"]
    argv += ["--", url]

    returncode, printed, stderr = _stream(
        argv, timeout=timeout_s, on_progress=on_progress,
    )
    if returncode != 0:
        raise FetchRefusedError(
            friendly_reason(stderr), client_related=is_client_failure(stderr),
        )

    path = Path(printed[-1]) if printed else None
    if path is None or not path.exists():
        # `--max-filesize` aborts with returncode 0 and prints nothing, which
        # is the one success-shaped failure this call has.
        if max_bytes and "larger than max-filesize" in (stderr or "").lower():
            raise FetchRefusedError(
                f"the video is larger than this server's "
                f"{max_bytes // (1024 * 1024)}MB limit",
            )
        raise FetchFailedError(
            "the downloader reported success but wrote no file: "
            f"{friendly_reason(stderr)}",
        )

    segments: list[dict] = []
    source = ""
    if caption_source:
        segments, source = _read_captions(stage_dir, file_id, caption_source)

    return Downloaded(path=path, segments=segments, transcript_source=source)


def _read_captions(
    stage_dir: Path, file_id: str, caption_source: str,
) -> tuple[list[dict], str]:
    """Parse whatever caption file yt-dlp wrote, or give up quietly.

    Giving up quietly is the point: a missing or unparseable caption track
    means whisper does the job, which is what would have happened anyway. It
    is not worth failing a download that succeeded.
    """
    tracks = sorted(stage_dir.glob(f"{file_id}*.vtt"))
    if not tracks:
        log.info("fetch: no caption file was written for %s — whisper will run", file_id)
        return [], ""
    try:
        raw = tracks[0].read_text("utf-8", errors="replace")
    except OSError as e:
        log.warning("fetch: caption file unreadable for %s: %s", file_id, e)
        return [], ""

    segments = parse_vtt(raw)
    if not segments:
        log.info("fetch: %s parsed to nothing usable — whisper will run", tracks[0].name)
        return [], ""
    log.info(
        "fetch: %d caption segments from %s (%s)",
        len(segments), tracks[0].name, caption_source,
    )
    return segments, caption_source


_TIMING = re.compile(
    r"(\d{1,2}:)?(\d{2}):(\d{2})[.,](\d{1,3})\s*-->\s*"
    r"(\d{1,2}:)?(\d{2}):(\d{2})[.,](\d{1,3})",
)
#: `<00:00:01.234>` position markers and `<c>`/`</c>` styling, which YouTube's
#: auto-captions carry inline on every word.
_TAGS = re.compile(r"<[^>]*>")


def _clock(hours: str | None, minutes: str, seconds: str, millis: str) -> float:
    h = int((hours or "0").rstrip(":") or 0)
    return h * 3600 + int(minutes) * 60 + int(seconds) + int(millis.ljust(3, "0")) / 1000


def parse_vtt(text: str, *, min_segment_s: float = 5.0) -> list[dict]:
    """WebVTT to `{t_start, t_end, text}` segments, deduplicated and merged.

    Two things make this more than a format parse, and both come from
    auto-captions specifically:

    **Rolling repetition.** YouTube's auto-captions "paint on": each cue
    repeats the tail of the one before it so the words appear to accumulate
    on screen. Parsed literally, a ten-minute video produces a transcript
    that says everything two or three times — which is not merely ugly, it
    poisons retrieval, because a chunk of triplicated text matches a query
    about that phrasing far more strongly than the sentence deserves.
    Deduplication is per *line* against the previous cue, which is the level
    the repetition actually happens at.

    **Granularity.** Cues arrive roughly per phrase, sometimes per second.
    Whisper's segments are sentence-shaped, and everything downstream —
    chunking, the `[HH:MM:SS]` sidecar, the frame-description context — was
    built against that shape. Adjacent cues are merged up to
    `min_segment_s` so a caption track and a whisper transcript are the same
    kind of object by the time anything else sees them.
    """
    cues: list[tuple[float, float, list[str]]] = []
    block: list[str] = []

    def flush(block: list[str]) -> None:
        timing = None
        body: list[str] = []
        for line in block:
            match = _TIMING.search(line)
            if match and timing is None:
                groups = match.groups()
                timing = (
                    _clock(groups[0], groups[1], groups[2], groups[3]),
                    _clock(groups[4], groups[5], groups[6], groups[7]),
                )
                continue
            if timing is not None:
                cleaned = html.unescape(_TAGS.sub("", line)).strip()
                if cleaned:
                    body.append(cleaned)
        if timing is not None and body:
            cues.append((timing[0], timing[1], body))

    for raw_line in (text or "").splitlines():
        if raw_line.strip():
            block.append(raw_line)
            continue
        if block:
            flush(block)
            block = []
    if block:
        flush(block)

    # Drop lines carried over from the previous cue. Comparing against the
    # whole previous cue rather than only its last line matters because the
    # repetition is a two-line window, not a one-line one.
    deduped: list[tuple[float, float, str]] = []
    previous: list[str] = []
    for start, end, body in cues:
        fresh = [line for line in body if line not in previous]
        previous = body
        if fresh:
            deduped.append((start, end, " ".join(fresh)))

    return _merge(deduped, min_segment_s=min_segment_s)


def _merge(
    cues: list[tuple[float, float, str]], *, min_segment_s: float,
) -> list[dict]:
    """Glue short adjacent cues into whisper-shaped segments."""
    merged: list[dict] = []
    for start, end, text in cues:
        if merged and (end - merged[-1]["t_start"]) < min_segment_s:
            merged[-1]["t_end"] = end
            merged[-1]["text"] = f"{merged[-1]['text']} {text}".strip()
            continue
        merged.append({"t_start": start, "t_end": end, "text": text})
    return merged


def _lang_matchers(langs: str) -> tuple[str, ...]:
    return tuple(part.strip().lower() for part in (langs or "").split(",") if part.strip())


def _lang_matches(tag: str, wanted: tuple[str, ...]) -> bool:
    """Does a caption language tag satisfy one of the requested patterns?

    A trailing `.*` means "this language, any region": `en.*` matches `en`,
    `en-US`, `en-GB` and `en-orig`, and stops at the region separator.

    **Deliberately stricter than yt-dlp's own matching**, which treats the
    pattern as a regex and would therefore accept `engineering` for `en.*`.
    That is a silly-looking case with a serious version: this decides which
    track gets ingested and how it is attributed, and the permissive failure
    is claiming a video has English subtitles when it has none, then ingesting
    something else as though it were what was asked for. Being narrower than
    the flag we pass can only mean falling back to whisper, which is what
    would have happened anyway.
    """
    tag = (tag or "").lower()
    for pattern in wanted:
        if pattern.endswith(".*"):
            base = pattern[:-2]
            if tag == base or tag.startswith((f"{base}-", f"{base}_")):
                return True
        elif tag == pattern:
            return True
    return False
