"""Short installed-backend admission smoke; no network, models or account writes.

Run locally with the checkout's Python or stream through the Tower wrapper.
Cases use isolated production queues, an explicit cap-one contention fixture,
long-aging unaged-priority fixtures and a backdated GPU-aging timestamp. They do not measure
live GPU latency or inspect the running application's queues.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path
from types import SimpleNamespace


class SmokeSetupError(Exception):
    pass


class SmokeFailureError(Exception):
    pass


def require(condition, explanation):
    if not condition:
        raise SmokeFailureError(explanation)


def load_runtime(config: Path):
    try:
        import yaml

        from audrey.background_admission import BackgroundGenerationGate
        from audrey.pipeline.fair_gate import FairLocalGate
        from audrey.routes.inflight import UserInflightRegistry
        from audrey.scheduling import (
            LocalSchedulingSettings,
            current_background_queued_at,
            current_local_workload,
            local_workload,
        )
    except ImportError as error:
        raise SmokeSetupError("Installed backend lacks admission support; rebuild audrey.") from error
    try:
        if config.stat().st_size > 1024 * 1024:
            raise ValueError
        raw = yaml.safe_load(config.read_text(encoding="utf-8"))
        fairness = raw.get("fairness", {}) or {}
        settings = LocalSchedulingSettings.model_validate(fairness)
        capacity = raw.get("gpu", {}).get("concurrency", 1)
        per_user = fairness.get("max_inflight_per_user", 3)
        if any(type(value) is not int or not 1 <= value <= 32 for value in (capacity, per_user)):
            raise ValueError
    except (OSError, UnicodeError, ValueError, TypeError, AttributeError, yaml.YAMLError) as error:
        raise SmokeSetupError("Configuration is unreadable or invalid for this bounded smoke (capacities 1–32).") from error
    return SimpleNamespace(
        gate=FairLocalGate, registry=UserInflightRegistry, adapter=BackgroundGenerationGate,
        workload=local_workload, current=current_local_workload, queued_at=current_background_queued_at,
        aging=settings.background_aging_seconds, capacity=capacity, per_user=per_user,
    )


class Scenario:
    def __init__(self, runtime, *, configured=False, unaged_fixture=False):
        self.runtime = runtime
        aging = max(60.0, runtime.aging) if unaged_fixture else runtime.aging
        self.gate = runtime.gate(concurrency=runtime.capacity if configured else 1, background_aging_seconds=aging)
        self.registry = runtime.registry(max_inflight_per_user=runtime.per_user if configured else 1, background_aging_seconds=aging)
        self.adapter = runtime.adapter(self.gate, self.registry)
        self.tasks = []

    async def __aenter__(self):
        return self

    def start(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.tasks.append(task)
        return task

    async def __aexit__(self, *_error):
        for task in self.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.idle()

    def idle(self):
        for resource in (self.gate, self.registry):
            pressure = resource.pressure_snapshot()
            require(pressure["in_use"] == pressure["waiting"] == 0, "Admission resources were not fully released.")
        require(self.runtime.current() == "interactive" and self.runtime.queued_at() is None, "Workload scope leaked.")


async def until(predicate):
    while not predicate():  # noqa: ASYNC110 — production pressure snapshots expose no event.
        await asyncio.sleep(0.001)


async def chat(scenario, order):
    async with scenario.registry.slot("smoke-owner"):
        async with scenario.gate.acquire("smoke", location="local", user_id="smoke-owner"):
            require(scenario.runtime.current() == "interactive", "Chat workload was misclassified.")
            order.append("chat")


async def background(scenario, order, *, started=None, release=None):
    caller = asyncio.current_task()
    async with scenario.adapter.acquire("smoke", location="local", user_id="smoke-owner"):
        require(asyncio.current_task() is caller, "Provider work moved to a child task.")
        require(scenario.runtime.current() == "background", "Background workload was misclassified.")
        order.append("background")
        if started:
            started.set()
        if release:
            await release.wait()


async def configured_capacity(runtime):
    async with Scenario(runtime, configured=True) as scenario:
        async with AsyncExitStack() as held:
            for _ in range(runtime.capacity):
                await held.enter_async_context(scenario.gate.acquire("smoke", location="local", user_id="smoke-blocker"))
            for _ in range(runtime.per_user):
                await held.enter_async_context(scenario.registry.slot("smoke-owner"))
            order = []
            queued = scenario.start(background(scenario, order))
            await until(lambda: scenario.registry.pressure_snapshot()["waiting"] == 1)
            require(scenario.gate.pressure_snapshot()["in_use"] == runtime.capacity, "GPU capacity was not enforced.")
            require(scenario.registry.snapshot()["smoke-owner"] == runtime.per_user, "User capacity was not enforced.")
            queued.cancel()
            result = (await asyncio.gather(queued, return_exceptions=True))[0]
            require(isinstance(result, asyncio.CancelledError) and not order, "A queued call ran after cancellation.")


async def chat_priority(runtime):
    async with Scenario(runtime, unaged_fixture=True) as scenario:
        order = []
        async with scenario.gate.acquire("smoke", location="local", user_id="smoke-blocker"):
            agent = scenario.start(background(scenario, order))
            await until(lambda: scenario.gate.pressure_snapshot()["waiting_background"] == 1)
            foreground = scenario.start(chat(scenario, order))
            await until(lambda: scenario.gate.pressure_snapshot()["waiting_interactive"] == 1)
            require(not order, "The blocking holder did not serialize work.")
        await asyncio.gather(agent, foreground)
        require(order == ["chat", "background"], "Chat did not pass the queued same-owner background call.")


async def background_aging(runtime):
    async with Scenario(runtime) as scenario:
        order = []

        async def aged():
            # Synthetic age exercises the configured threshold without sleeping
            # for it or changing the live configuration or system clock.
            with runtime.workload("background", queued_at=time.monotonic() - runtime.aging - 1):
                async with scenario.gate.acquire("smoke", location="local", user_id="smoke-owner"):
                    order.append("background")

        async with scenario.gate.acquire("smoke", location="local", user_id="smoke-blocker"):
            agent = scenario.start(aged())
            await until(lambda: scenario.gate.pressure_snapshot()["aged_background"] == 1)
            foreground = scenario.start(chat(scenario, order))
            await until(lambda: scenario.gate.pressure_snapshot()["waiting_interactive"] == 1)
        await asyncio.gather(agent, foreground)
        require(order == ["background", "chat"], "Aged background work lost its place to later chat.")


async def running_call(runtime):
    async with Scenario(runtime) as scenario:
        order, started, release = [], asyncio.Event(), asyncio.Event()
        agent = scenario.start(background(scenario, order, started=started, release=release))
        await started.wait()
        foreground = scenario.start(chat(scenario, order))
        await until(lambda: scenario.registry.pressure_snapshot()["waiting"] == 1)
        require(order == ["background"] and not agent.done(), "Chat interrupted a running call.")
        release.set()
        await asyncio.gather(agent, foreground)
        require(order == ["background", "chat"], "A running call failed to finish before chat.")


async def cancel_handoff(runtime):
    async with Scenario(runtime, unaged_fixture=True) as scenario:
        order = []
        async with scenario.gate.acquire("smoke", location="local", user_id="smoke-blocker"):
            agent = scenario.start(background(scenario, order))
            await until(lambda: scenario.gate.pressure_snapshot()["waiting_background"] == 1)
            foreground = scenario.start(chat(scenario, order))
            await until(lambda: scenario.gate.pressure_snapshot()["waiting_interactive"] == 1)
            agent.cancel()
            result = (await asyncio.gather(agent, return_exceptions=True))[0]
            require(isinstance(result, asyncio.CancelledError), "Background cancellation did not settle.")
        await foreground
        require(order == ["chat"], "Cancelled background work still ran.")


CASES = (
    ("configured_capacity", configured_capacity), ("chat_priority_cap_one", chat_priority),
    ("gpu_background_aging_synthetic_timestamp", background_aging),
    ("running_call_not_preempted", running_call), ("cancel_handoff_releases_resources", cancel_handoff),
)


async def run(runtime, deadline_seconds):
    checks = []
    for name, case in CASES:
        try:
            async with asyncio.timeout(deadline_seconds):
                await case(runtime)
            checks.append({"check": name, "passed": True})
        except (SmokeFailureError, TimeoutError) as error:
            checks.append({"check": name, "passed": False, "reason": str(error) if isinstance(error, SmokeFailureError) else "Timed out."})
            break
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--timeout-seconds", type=float, default=10)
    parser.add_argument("--json", action="store_true", help="Print one compact JSON receipt.")
    args = parser.parse_args(argv)
    if not 0 < args.timeout_seconds <= 10:
        parser.error("--timeout-seconds must be positive and at most 10 (five checks, at most 50 seconds).")
    receipt = {"scope": "isolated installed scheduler; no models/network/accounts", "checks": []}
    try:
        runtime = load_runtime(args.config)
        receipt["configured"] = {"gpu_concurrency": runtime.capacity, "max_inflight_per_user": runtime.per_user, "background_aging_seconds": runtime.aging}
        receipt["fixtures"] = {"contention_gpu_capacity": 1, "contention_user_capacity": 1,
                               "unaged_priority_aging_seconds": max(60.0, runtime.aging),
                               "gpu_aging": "backdated timestamp with configured threshold"}
        receipt["checks"] = asyncio.run(run(runtime, args.timeout_seconds))
        status = 0 if len(receipt["checks"]) == len(CASES) and all(row["passed"] for row in receipt["checks"]) else 1
    except SmokeSetupError as error:
        receipt["error"], status = str(error), 2
    except KeyboardInterrupt:
        receipt["error"], status = "Interrupted.", 130
    except Exception as error:  # noqa: BLE001 — CLI boundary returns a sanitized crash receipt.
        receipt["error"], status = f"Smoke crashed ({type(error).__name__}).", 3
    receipt["exit_code"] = status
    if args.json:
        print(json.dumps(receipt, separators=(",", ":")))
    else:
        for row in receipt["checks"]:
            print(f"{'PASS' if row['passed'] else 'FAIL'} {row['check']}" + (f": {row['reason']}" if "reason" in row else ""))
        if "error" in receipt:
            print(receipt["error"])
        print(f"{'PASS' if status == 0 else 'FAIL'} admission smoke (exit {status}); {receipt['scope']}")
        if "configured" in receipt:
            print("Configured: " + json.dumps(receipt["configured"], separators=(",", ":")))
            print("Fixtures: " + json.dumps(receipt["fixtures"], separators=(",", ":")))
    return status


if __name__ == "__main__":
    sys.exit(main())
