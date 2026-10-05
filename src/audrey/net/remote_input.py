"""Parse request-local remote evidence without storing it in a user library."""
from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
import tempfile
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from audrey.kb import remote_input_worker
from audrey.net.public_fetch import PublicAsset, _error

PARSE_DEADLINE_SECONDS = 15
log = logging.getLogger(__name__)


async def parse_remote_asset(asset: PublicAsset, *, kind: str, max_chars: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="audrey-remote-input-") as directory:
        path = Path(directory) / "input.bin"
        await asyncio.to_thread(path.write_bytes, asset.data)
        worker = await asyncio.to_thread(lambda: str(Path(remote_input_worker.__file__).resolve()))
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, worker,
                str(path), kind, asset.media_type, str(max_chars),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            )
        except OSError as exc:
            raise _error(503, "responses_remote_parser_unavailable", "Remote input parser is unavailable.") from exc
        try:
            try:
                async with asyncio.timeout(PARSE_DEADLINE_SECONDS):
                    stdout, _ = await process.communicate()
            except TimeoutError as exc:
                raise _error(504, "responses_remote_input_timeout", "Remote input extraction timed out.") from exc
            if process.returncode != 0:
                log.warning("Remote input parser exited without evidence: returncode=%s", process.returncode)
                interrupted = process.returncode in {-signal.SIGKILL, -signal.SIGXCPU}
                raise _error(
                    413 if interrupted else 502,
                    "responses_remote_input_too_large" if interrupted else "responses_remote_parser_failed",
                    "Remote input parser stopped before producing usable evidence.",
                )
            result = json.loads(stdout)
            if "error" in result:
                raise _error(result["status"], "responses_remote_input_invalid", result["error"])
            return result
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="Remote input could not be extracted.") from exc
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
