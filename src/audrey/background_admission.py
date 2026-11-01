"""Admit one background model call, yielding queued resources to chat."""

from __future__ import annotations

import asyncio
import sys
import time
from contextlib import asynccontextmanager

from audrey.scheduling import local_workload


class BackgroundGenerationGate:
    """Keep per-user then GPU ordering without trapping chat behind queued work.

    A background call may hold a per-user permit while waiting for the GPU.
    A later same-owner chat can request that permit back until generation
    starts. Retrying retains the original age so repeated handoffs cannot
    starve background work. Running generations remain uninterrupted.
    """

    def __init__(self, gate, inflight):
        self.gate, self.inflight = gate, inflight

    @asynccontextmanager
    async def acquire(self, model: str, *, location: str, user_id: str | None = None):
        queued_at = time.monotonic()
        with local_workload("background", queued_at=queued_at):
            while True:
                async with self.inflight.slot(
                    user_id, yield_to_interactive=True, queued_at=queued_at,
                ) as permit:
                    gpu = self.gate.acquire(model, location=location, user_id=user_id)
                    # Only __aenter__ runs in a child: the provider and its
                    # authority/usage contexts remain in the caller's task.
                    entering = asyncio.create_task(gpu.__aenter__())
                    give_way = asyncio.create_task(permit.give_way.wait())
                    try:
                        await asyncio.wait((entering, give_way), return_when=asyncio.FIRST_COMPLETED)
                        if entering.done():
                            await entering
                            # Both events can complete together. The permit
                            # arbitrates synchronously before the provider runs.
                            if permit.mark_running():
                                yield
                                return
                    finally:
                        error = sys.exc_info()
                        # Runtime shutdown/lease cancellation can cancel this
                        # caller twice. Shield the resource cleanup, then keep
                        # awaiting it so no child or GPU reservation is orphaned.
                        cleanup = asyncio.create_task(self._close_gpu(gpu, entering, give_way, error))
                        cancelled = False
                        while not cleanup.done():
                            try:
                                await asyncio.shield(cleanup)
                            except asyncio.CancelledError:
                                cancelled = True
                        cleanup.result()
                        if cancelled:
                            raise asyncio.CancelledError

    @staticmethod
    async def _close_gpu(gpu, entering, give_way, error):
        give_way.cancel()
        if not entering.done():
            entering.cancel()
        await asyncio.gather(entering, give_way, return_exceptions=True)
        if not entering.cancelled() and entering.exception() is None:
            # FairLocalGate does not install task-local state on entry;
            # closing it here only releases its shared GPU reservation.
            await gpu.__aexit__(*error)
