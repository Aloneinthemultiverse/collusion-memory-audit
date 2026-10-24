"""Backtrack a decision through the provenance graph, with hidden reasoning at every step.

  python scripts/backtrack.py runs/main/sym/openai/gpt-oss-20b/C1M_monitored/seed15 --round 12 --firm "Firm B" --depth 2

Shows the firm's action and hidden reasoning in that round, then what it observed (rival's past
actions, board notes, its own memory), and for each of those the action + reasoning that produced it,
recursively up to --depth steps back. Writes the same trace as JSON next to the game.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from collusion.provenance import Run, build_graph  # noqa: E402


def describe(e: dict) -> str:
    d = e["data"]
    if e["kind"] == "action":
        return f"action {d['quantities']}"
    if e["kind"] == "note":
        return f"board note: {d['text']!r}"
    if e["kind"] == "memory_write":
        return f"memory: {(d['plans'] + ' | ' + d['insights'])[:200]!r}"
    if e["kind"] == "market_clear":
        return f"market cleared, prices {[round(p, 1) for p in d['prices']]}"
    return e["kind"]


def producing_action(g, by_id: dict, eid: int) -> dict | None:
    """The action that produced a note / memory entry (its parent), or the event itself if an action."""
    e = by_id[eid]
    if e["kind"] == "action":
        return e
    for p in g.predecessors(eid):
        if by_id[p]["kind"] == "action":
            return by_id[p]
    return None


def trace(g, by_id, action: dict, depth: int, indent: int = 0, seen=None) -> dict:
    seen = seen if seen is not None else set()
    pad = "  " * indent
    print(f"{pad}[{action['agent']} r{action['round']}] {describe(action)}")
    reasoning = action["data"].get("reasoning", "")
    print(f"{pad}  reasoning: {reasoning[:500] or '(not recorded)'}")
    node = {"event": action["id"], "agent": action["agent"], "round": action["round"],
            "action": action["data"]["quantities"], "reasoning": reasoning, "sources": []}
    obs = [by_id[p] for p in g.predecessors(action["id"]) if by_id[p]["kind"] == "observation"]
    if not obs or depth == 0:
        return node
    for kind, ids in obs[0]["data"]["sources"].items():
        if kind == "own_history" or not ids:
            continue
        latest = max(ids)  # most recent item of this source
        src = by_id[latest]
        print(f"{pad}  <- via {kind}: {describe(src)}")
        prod = producing_action(g, by_id, latest)
        entry = {"via": kind, "event": latest, "what": describe(src)}
        if prod and prod["id"] not in seen:
            seen.add(prod["id"])
            entry["produced_by"] = trace(g, by_id, prod, depth - 1, indent + 2, seen)
        node["sources"].append(entry)
    return node


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--firm", required=True)
    ap.add_argument("--depth", type=int, default=2)
    args = ap.parse_args()
    run = Run(Path(args.run_dir))
    g = build_graph(run.events)
    by_id = {e["id"]: e for e in run.events}
    act = next(e for e in run.events if e["kind"] == "action" and e["agent"] == args.firm and e["round"] == args.round)
    tree = trace(g, by_id, act, args.depth)
    out = Path(args.run_dir) / f"backtrack_{args.firm.replace(' ', '')}_r{args.round}.json"
    out.write_text(json.dumps(tree, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
