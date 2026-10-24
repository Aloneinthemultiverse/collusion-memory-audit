"""Counterfactual replay of a recorded game (adapted from AgentSentry's boundary counterfactuals).

For a candidate pathway (channel c, S -> T, onset r) we re-run the same game with T's view of c
masked (or replaced by a decoy) from round r, and compare it with *null replays* that only
resample from round r. Rounds before r are reused from the LLM cache, so they cost nothing.

effect = division(null replays) - division(intervened replay), averaged over rounds >= r.
A positive effect means the pathway was supporting market division.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from .game import Channels, GameConfig, Intervention, play
from .llm import LLM
from .market import MarketConfig
from .provenance import Run


def config_from_run(run: Run) -> GameConfig:
    c = run.config
    m = c["market"]
    if m.get("kind") == "bertrand":
        from .bertrand import BertrandConfig
        market = BertrandConfig(**m)
    else:
        market = MarketConfig(goods=m["goods"], a=m["a"], b=m["b"], costs=m["costs"], capacity=m["capacity"])
    return GameConfig(rounds=c["rounds"], seed=c["seed"], channels=Channels(**c["channels"]),
                      market=market, history_window=c.get("history_window", 10),
                      monitor_notice=c.get("monitor_notice", False))


def replay(run_dir: Path, llm: LLM, out_root: Path, onset: int, channels: list[str] | None = None,
           target: str | None = None, mode: str = "mask", tag: str = "cf",
           interventions: list[Intervention] | None = None, name: str | None = None) -> dict:
    """Run one replay. channels=None gives a null replay (resample only); target=None hits both firms.
    `interventions` + `name` allow arbitrary combinations (e.g. two pathways with different targets)."""
    base = Run(run_dir)
    cfg = config_from_run(base)
    cfg.resample_from = onset
    cfg.resample_tag = f"{tag}@{onset}"
    who = (target or "both").replace(" ", "")
    if name is None:
        name = f"null@{onset}_{tag}" if not channels else f"{'+'.join(channels)}_{mode}_{who}@{onset}_{tag}"
    if interventions is not None:
        cfg.interventions = interventions
    elif channels:
        cfg.interventions = [Intervention(channel=c, mode=mode, target=target, from_round=onset)
                             for c in channels]
    out_dir = Path(out_root) / name
    if (out_dir / "done.json").exists():
        return json.loads((out_dir / "done.json").read_text())
    play(cfg, llm, out_dir)
    result = {"name": name, "onset": onset, "channels": channels, "target": target, "mode": mode,
              "division_after": statistics.fmean(Run(out_dir).division_series()[onset - 1:])}
    (out_dir / "done.json").write_text(json.dumps(result))
    return result
