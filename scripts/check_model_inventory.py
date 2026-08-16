#!/usr/bin/env python3
"""Cross-check every model named in `config.yaml` against what Ollama actually has.

WHY

`_validate_deep_panel_pools` rejects a model name that is missing from
`model_registry` — but `model_registry` is part of the same config file. It
validates the config against ITSELF. Nothing has ever checked it against the
box, so a model that was never pulled (or was deleted) passes boot cleanly and
fails at request time, where the damage is quiet:

    unknown model -> HealthTracker.is_healthy returns True (unknown -> True)
                  -> ModelRegistry.location_of defaults to "local"
                  -> the Ollama call fails, AFTER taking a GPU-gate slot
                  -> the pool degrades to the next member

The 2026-08-15 audit found FIVE phantoms sitting in production config, one of
them (`deepseek-r1:32b`) the first-listed worker of two `audrey_local` pools —
and first-listed is the de facto primary. None of it was visible anywhere.

⚠️ It also reports the other direction: weights on disk that nothing references.
That is the reclaim list, and it is worth real space (148 GB after the qwen3.8
swap). ⚠️ But read the caveat on `kb.text_embedder` below before deleting
anything.

HOW IT FINDS MODEL NAMES

⚠️ **It does NOT hardcode a list of role keys.** Writing that list by hand
missed `deep_panel_research.*.factchecker` and `kb.video.summarise_model` on the
first attempt — both real roles that dispatch a model. Instead it descends
generically through the SECTIONS known to carry models, so a role added later is
covered without touching this file. The section list is short and stable; the
roles inside it are not.

USAGE

⚠️ The Unraid box has no `python3`, and the repo is not bind-mounted into
`audrey-ai` — but that container has Python and can reach Ollama over the
compose network. So:

  docker cp config.yaml audrey-ai:/tmp/cfg.yaml
  docker cp scripts/check_model_inventory.py audrey-ai:/tmp/cmi.py
  docker exec audrey-ai python3 /tmp/cmi.py --config /tmp/cfg.yaml \\
      --tags-url http://ollama:11434

Or, with no network access at all, paste `ollama list` output in:

  docker exec ollama ollama list > /tmp/tags.txt
  python3 scripts/check_model_inventory.py --ollama-list /tmp/tags.txt

Exit status is 1 when config names something Ollama does not have, so this can
gate a deploy. Reclaimable-but-present is NOT an error.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Sections of `config.yaml` that can hold a model name. Everything inside these
# is descended generically — see the module docstring for why the role keys are
# deliberately NOT enumerated.
#
# ⚠️ `kb.text_embedder` is here so the embedder is counted as REFERENCED. It is
# reached outside `model_registry`, so without this entry `nomic-embed-text`
# shows up on the reclaim list and someone deletes the KB's embedder.
_MODEL_SECTIONS: tuple[tuple[str, ...], ...] = (
    ("model_registry",),
    ("deep_panel",),
    ("deep_panel_cloud",),
    ("deep_panel_local",),
    ("deep_panel_research",),
    ("router", "model"),
    ("fast_path", "tool_capable_models"),
    ("passthrough", "allowed_models"),
    ("kb", "text_embedder"),
    ("kb", "video", "summarise_model"),
    ("vision", "model"),
    ("vision", "also_capable"),
)

# How much it matters that a given reference is dead, worst first. Matched as a
# substring against the dotted path, so `workers[0]` beats the generic
# `workers`. Anything unmatched lands at the bottom.
_SEVERITY: tuple[tuple[str, str], ...] = (
    ("router.model", "CRITICAL"),          # every request classifies through it
    ("workers[0]", "HIGH"),                # first-listed is the de facto primary
    ("researchers[0]", "HIGH"),
    ("tool_capable_models", "HIGH"),       # dead here = silent ungrounded answers
    (".writer", "HIGH"),
    (".verifier", "HIGH"),
    (".factchecker", "HIGH"),
    (".synthesizer", "HIGH"),
    ("text_embedder", "HIGH"),
    ("summarise_model", "HIGH"),
    ("workers[", "MEDIUM"),
    ("researchers[", "MEDIUM"),
    (".fallback_synth", "MEDIUM"),
    ("model_registry", "MEDIUM"),          # a pool member, reachable by failover
    ("allowed_models", "LOW"),             # passthrough clients only
    ("also_capable", "LOW"),
)
_SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "UNKNOWN": 4}


@dataclass(slots=True)
class Inventory:
    """What Ollama reports it has, plus sizes when the source carried them."""

    names: set[str] = field(default_factory=set)
    sizes: dict[str, int] = field(default_factory=dict)  # bytes; absent for cloud


def normalise(name: str) -> str:
    """`nomic-embed-text` and `nomic-embed-text:latest` are the SAME model.

    ⚠️ Ollama implies `:latest` when a tag is omitted, and `config.yaml` uses
    the bare form for `kb.text_embedder` while `ollama list` always prints the
    tag. Without this the checker reports the KB embedder as a phantom AND as
    reclaimable in the same run — two confident, opposite, wrong answers about
    one model. Found by running the tool against the real box on its first pass.
    """
    return name if ":" in name else f"{name}:latest"


def _severity(path: str) -> str:
    for needle, level in _SEVERITY:
        if needle in path:
            return level
    return "UNKNOWN"


def _dig(cfg: dict[str, Any], path: tuple[str, ...]) -> Any:
    node: Any = cfg
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _descend(node: Any, path: str, out: dict[str, list[str]]) -> None:
    """Collect every string leaf under `node`, keyed by model name.

    A dict entry with a `name` key is a registry spec — record `name` and stop,
    rather than also collecting `location: local` as if it were a model.
    """
    if isinstance(node, dict):
        if isinstance(node.get("name"), str):
            out.setdefault(node["name"], []).append(f"{path}.name")
            return
        for key, value in node.items():
            _descend(value, f"{path}.{key}", out)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            _descend(value, f"{path}[{i}]", out)
    elif isinstance(node, str) and node.strip():
        out.setdefault(node, []).append(path)


def config_models(cfg: dict[str, Any]) -> dict[str, list[str]]:
    """`{model_name: [dotted path, ...]}` for every model the config names."""
    found: dict[str, list[str]] = {}
    for section in _MODEL_SECTIONS:
        node = _dig(cfg, section)
        if node is not None:
            _descend(node, ".".join(section), found)
    return found


def parse_ollama_list(text: str) -> Inventory:
    """Parse `ollama list` table output.

    ⚠️ Bound on the COLUMN COUNT, not on a size regex: a cloud model's SIZE is
    a bare `-`, and a matcher that insisted on `17 GB` would silently drop every
    cloud entry and report them all as phantoms.
    """
    inv = Inventory()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2 or parts[0].upper() == "NAME":
            continue
        name = parts[0]
        if ":" not in name:
            continue
        inv.names.add(name)
        # NAME ID SIZE UNIT MODIFIED… — size is parts[2:4] when present.
        if len(parts) >= 4 and parts[2] not in ("-", ""):
            try:
                inv.sizes[name] = _to_bytes(float(parts[2]), parts[3])
            except (ValueError, KeyError):
                pass
    return inv


_UNITS = {"B": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12}


def _to_bytes(value: float, unit: str) -> int:
    return int(value * _UNITS[unit.upper()])


def fetch_tags(base_url: str, timeout: float = 10.0) -> Inventory:
    """`GET /api/tags`. Same shape as `ollama list`, machine-readable."""
    url = base_url.rstrip("/") + "/api/tags"
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
        payload = json.loads(resp.read().decode("utf-8"))
    inv = Inventory()
    for entry in payload.get("models") or []:
        name = entry.get("name") or entry.get("model")
        if not name:
            continue
        inv.names.add(str(name))
        if isinstance(entry.get("size"), int):
            inv.sizes[str(name)] = entry["size"]
    return inv


def build_report(
    referenced: dict[str, list[str]], inv: Inventory
) -> dict[str, Any]:
    # Compare on the normalised tag in BOTH directions — see `normalise`.
    on_box = {normalise(n) for n in inv.names}
    missing = []
    for name, paths in referenced.items():
        if normalise(name) in on_box:
            continue
        level = min((_severity(p) for p in paths), key=lambda s: _SEVERITY_ORDER[s])
        missing.append({"model": name, "severity": level, "paths": sorted(paths)})
    missing.sort(key=lambda m: (_SEVERITY_ORDER[m["severity"]], m["model"]))

    wanted = {normalise(n) for n in referenced}
    unreferenced = []
    for name in sorted(inv.names):
        if normalise(name) in wanted:
            continue
        unreferenced.append({"model": name, "bytes": inv.sizes.get(name, 0)})
    return {
        "referenced": len(referenced),
        "on_box": len(inv.names),
        "missing": missing,
        "unreferenced": unreferenced,
        "reclaimable_bytes": sum(u["bytes"] for u in unreferenced),
    }


def render(rep: dict[str, Any]) -> str:
    out: list[str] = []
    w = out.append
    w("═══ model inventory ═══")
    w(f"config references {rep['referenced']} model(s); Ollama has {rep['on_box']}")
    w("")
    if rep["missing"]:
        w("⛔ NAMED IN CONFIG, NOT ON THE BOX — these fail at REQUEST time, after")
        w("   taking a GPU-gate slot. Boot validation cannot see them.")
        for m in rep["missing"]:
            w(f"  [{m['severity']:<8}] {m['model']}")
            for p in m["paths"]:
                w(f"             {p}")
    else:
        w("✅ every model named in config is present on the box.")
    w("")
    if rep["unreferenced"]:
        gb = rep["reclaimable_bytes"] / 10**9
        w(f"◦ ON THE BOX, UNREFERENCED — {gb:.0f} GB reclaimable:")
        for u in sorted(rep["unreferenced"], key=lambda x: -x["bytes"]):
            size = f"{u['bytes'] / 10**9:.0f} GB" if u["bytes"] else "—"
            w(f"    {u['model']:<34} {size}")
        w("  ⚠️ Keep the predecessors of any recent swap until it is verified in")
        w("     production — rollback is a re-pull of everything deleted here.")
    else:
        w("◦ nothing unreferenced on the box.")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Cross-check config.yaml model names against Ollama's inventory.",
    )
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--tags-url", default=None,
                    help="Ollama base URL, e.g. http://ollama:11434 (uses /api/tags)")
    ap.add_argument("--ollama-list", default=None,
                    help="file holding `ollama list` output, or '-' for stdin")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if not args.tags_url and not args.ollama_list:
        ap.error("give one of --tags-url or --ollama-list")

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"no such config: {cfg_path}", file=sys.stderr)
        return 2
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    try:
        if args.tags_url:
            inv = fetch_tags(args.tags_url)
        elif args.ollama_list == "-":
            inv = parse_ollama_list(sys.stdin.read())
        else:
            inv = parse_ollama_list(Path(args.ollama_list).read_text(encoding="utf-8"))
    except OSError as e:
        print(f"could not read Ollama inventory: {e}", file=sys.stderr)
        return 2

    if not inv.names:
        # An empty inventory would report EVERY model as missing — a confident,
        # completely wrong table. Refuse rather than emit it.
        print("Ollama inventory came back empty — refusing to report every model "
              "as missing. Check the URL or the pasted list.", file=sys.stderr)
        return 2

    report = build_report(config_models(cfg), inv)
    print(json.dumps(report, indent=2) if args.json else render(report))
    return 1 if report["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
