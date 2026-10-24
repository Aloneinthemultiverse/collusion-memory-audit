"""Pilot: does market division appear with this model?

Runs C1 (all channels on) and C0 (no cross-agent channels) for one or more seeds.

  python scripts/run_pilot.py --provider mock --rounds 10
  python scripts/run_pilot.py --provider gemini --model gemini-2.5-flash-lite --rounds 30 --seeds 0 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collusion.game import Channels, GameConfig, play  # noqa: E402
from collusion.llm import LLM  # noqa: E402
from collusion.market import MarketConfig  # noqa: E402
from collusion.bertrand import BertrandConfig  # noqa: E402
from collusion.metrics import summarize, summarize_bertrand  # noqa: E402

# "sym": identical costs, so Nash has no specialisation and any market split needs coordination.
# "asym": each firm has a cheap good (Lin et al.); splitting is partly individually rational.
COSTS = {"sym": [[25.0, 25.0], [25.0, 25.0]], "asym": [[10.0, 40.0], [40.0, 10.0]]}

CONDITIONS = {
    "C1_all_channels": Channels(rival_history=True, board=True, memory=True),
    "C0_no_cross_agent": Channels(rival_history=False, board=False, memory=True),
    "C1M_monitored": Channels(rival_history=True, board=True, memory=True),  # + monitor notice
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="mock", choices=["mock", "gemini", "groq", "nvidia", "openrouter", "ollama"])
    ap.add_argument("--model", default="mock")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--reasoning-effort", default=None, choices=["low", "medium", "high"])
    ap.add_argument("--costs", default="sym", choices=list(COSTS))
    ap.add_argument("--market", default="cournot", choices=["cournot", "bertrand"])
    ap.add_argument("--key-env", default=None, help="env var holding the API key, e.g. NVIDIA_API_KEY_2")
    ap.add_argument("--out", default="runs/pilot")
    args = ap.parse_args()

    llm = LLM(args.provider, args.model, temperature=args.temperature,
              reasoning_effort=args.reasoning_effort, key_env=args.key_env)
    results = []
    for cond in args.conditions:
        for seed in args.seeds:
            tag = "bertrand" if args.market == "bertrand" else args.costs
            out_dir = Path(args.out) / tag / args.model / cond / f"seed{seed}"
            print(f"== {cond} seed={seed} -> {out_dir}")
            market = BertrandConfig() if args.market == "bertrand" else MarketConfig(costs=COSTS[args.costs])
            cfg = GameConfig(rounds=args.rounds, seed=seed, channels=CONDITIONS[cond], market=market,
                             monitor_notice=cond.startswith("C1M"))
            res = play(cfg, llm, out_dir)
            if args.market == "bertrand":
                s = summarize_bertrand(market, res["rounds"], last=min(10, args.rounds))
            else:
                s = summarize(cfg.market, res["rounds"], res["nash"], last=min(10, args.rounds))
            s.update({"condition": cond, "seed": seed})
            (out_dir / "summary.json").write_text(json.dumps(s, indent=2))
            results.append(s)
            if args.market == "bertrand":
                print(f"   collusion index {s['collusion_index']:+.2f}  mean price {s['mean_price']:.2f}  "
                      f"profit excess {s['profit_excess']:+.1%}")
            else:
                print(f"   HHI excess {s['hhi_excess']:+.3f}  CV excess {s['cv_excess']:+.3f}  "
                      f"profit excess {s['profit_excess']:+.1%}")

    print("\n== Pilot summary (last-10-round averages vs Cournot-Nash)")
    for s in results:
        if "collusion_index" in s:
            print(f"{s['condition']:20s} seed {s['seed']}: collusion index {s['collusion_index']:+.2f}  "
                  f"profit {s['profit_excess']:+.1%}")
        else:
            print(f"{s['condition']:20s} seed {s['seed']}: HHI {s['hhi_excess']:+.3f}  "
                  f"CV {s['cv_excess']:+.3f}  profit {s['profit_excess']:+.1%}")
    print(f"LLM calls: {llm.calls}, cache hits: {llm.cache_hits}")


if __name__ == "__main__":
    main()
