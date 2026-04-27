"""GPU concurrency gate.

Local Ollama models share two RTX 3090 Tis with a tight PSU budget — running
more than one local generation at a time risks brownouts and VRAM thrash. So
we serialize local workers behind an `asyncio.Semaphore` (default value 1).

Cloud models bypass the gate entirely: they run on Ollama Pro's hosted
infrastructure, so concurrency there is bounded by the cloud-side cap (3).

Usage:
    gate = GpuGate(concurrency=cfg.gpu_concurrency)
    async with gate.acquire(model_name, location="local"):
        await ollama.chat(...)
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

from audrey.metrics import gpu_gate_wait_seconds


class GpuGate:
    """Local-only async semaphore. Cloud calls pass through."""

    def __init__(self, *, concurrency: int = 1) -> None:
        if concurrency < 1:
            concurrency = 1
        self._sem = asyncio.Semaphore(concurrency)
        self._concurrency = concurrency

    @property
    def concurrency(self) -> int:
        return self._concurrency

    @asynccontextmanager
    async def acquire(self, model: str, *, location: str):
        """Acquire the gate iff `location == 'local'`. Cloud is a no-op.

        Cloud calls don't pay any gate cost so we don't observe — only local
        contention belongs in the histogram.
        """
        if location == "local":
            t0 = time.perf_counter()
            async with self._sem:
                gpu_gate_wait_seconds.observe(time.perf_counter() - t0)
                yield
        else:
            yield


__all__ = ["GpuGate"]
