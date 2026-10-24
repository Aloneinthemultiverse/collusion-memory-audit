"""Summarise the memory-transplant games: split rates, who specialises in what, and when.

  python scripts/analyze_transplant.py
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from scipy.stats import fisher_exact, mannwhitneyu

ROOT = Path("runs/transplant")


def markets(d: Path) -> list[list[list[float]]]:
    ev = [json.loads(line) for line in (d / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    return [e["data"]["Q"] for e in ev if e["kind"] == "market_clear"]


def main() -> None:
    res = json.loads(Path("results/transplant.json").read_text())
    rows = []
    for g in res["games"]:
        Q = markets(ROOT / g["arm"] / f"seed{g['seed']}")
        last = Q[-10:]
        a = [sum(q[0][k] for q in last) for k in (0, 1)]
        b = [sum(q[1][k] for q in last) for k in (0, 1)]
        a_main = 0 if a[0] >= a[1] else 1
        # share of B's output in the good A does NOT focus on (1 = B fully takes the other good)
        b_other = b[1 - a_main] / max(sum(b), 1e-9)
        a_focus = a[a_main] / max(sum(a), 1e-9)
        a_r1 = Q[0][0]
        a_focus_r1 = max(a_r1) / max(sum(a_r1), 1e-9)
        div = [abs(q[0][0] / max(q[0][0] + q[1][0], 1e-9) - q[0][1] / max(q[0][1] + q[1][1], 1e-9)) for q in Q]
        onset = next((t + 1 for t in range(len(div)) if all(x >= 0.8 for x in div[t:t + 5])), None)
        rows.append({**{k: g[k] for k in ("arm", "seed", "division", "donor")}, "a_focus": a_focus,
                     "b_other_good": b_other, "a_focus_round1": a_focus_r1, "split_onset": onset})
    for r in rows:
        print(f"{r['arm']:10s} {r['seed']} div {r['division']:.2f}  A focus {r['a_focus']:.2f} (r1 {r['a_focus_round1']:.2f})"
              f"  B on other good {r['b_other_good']:.2f}  onset {r['split_onset']}  donor {r['donor']}")
    arms = {a: [r for r in rows if r["arm"] == a] for a in ("transplant", "control", "empty")}
    summ = {}
    for a, rs in arms.items():
        summ[a] = {"split": sum(r["division"] >= 0.8 for r in rs), "n": len(rs),
                   "mean_division": statistics.fmean(r["division"] for r in rs),
                   "mean_b_other": statistics.fmean(r["b_other_good"] for r in rs),
                   "mean_a_focus_r1": statistics.fmean(r["a_focus_round1"] for r in rs),
                   "onsets": [r["split_onset"] for r in rs]}
        print(a, summ[a])
    tests = {}
    for other in ("control", "empty"):
        t, o = summ["transplant"], summ[other]
        p = fisher_exact([[t["split"], t["n"] - t["split"]], [o["split"], o["n"] - o["split"]]])[1]
        u = mannwhitneyu([r["division"] for r in arms["transplant"]], [r["division"] for r in arms[other]],
                         alternative="two-sided").pvalue
        tests[other] = {"fisher_p": p, "mannwhitney_p": u}
        print(f"transplant vs {other}: Fisher p={p:.3f}, Mann-Whitney p={u:.3f}")
    Path("results/transplant_analysis.json").write_text(json.dumps({"games": rows, "arms": summ, "tests": tests}, indent=2))


if __name__ == "__main__":
    main()
