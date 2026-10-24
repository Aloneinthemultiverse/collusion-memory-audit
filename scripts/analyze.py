"""Aggregate games and audits into tables and figures.

  python scripts/analyze.py --root runs/main/sym/openai/gpt-oss-20b --laya

Outputs results/ (markdown tables, JSON, PNG figures).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from collusion.auditors import audit_cost, expected_random_cost, heuristic_scores, random_scores  # noqa: E402
from collusion.provenance import Run  # noqa: E402


def mean_sd(xs: list[float]) -> str:
    if not xs:
        return "n/a"
    return f"{statistics.fmean(xs):.2f} ± {statistics.pstdev(xs):.2f} (n={len(xs)})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/main/sym/openai/gpt-oss-20b")
    ap.add_argument("--out", default="results")
    ap.add_argument("--laya", action="store_true", help="score candidates with Laya (CPU)")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# Results", ""]

    # ---- 1. positive vs negative control
    series, finals = {}, {}
    for cond_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for seed_dir in sorted(cond_dir.glob("seed*")):
            if not (seed_dir / "summary.json").exists():
                continue
            s = json.loads((seed_dir / "summary.json").read_text())
            div = Run(seed_dir).division_series()
            s["division"] = statistics.fmean(div[-10:])  # recompute: older summaries lack it
            series.setdefault(cond_dir.name, []).append(div)
            finals.setdefault(cond_dir.name, []).append(s)
    lines += ["## Controls (last 10 rounds)", "",
              "| Condition | Division index | Profit vs Nash |", "|---|---|---|"]
    for cond, ss in finals.items():
        lines.append(f"| {cond} | {mean_sd([s['division'] for s in ss])} | "
                     f"{mean_sd([100 * s['profit_excess'] for s in ss])} % |")
    plt.figure(figsize=(6, 3.2))
    for cond, runs in series.items():
        for i, sr in enumerate(runs):
            plt.plot(range(1, len(sr) + 1), sr, color="C0" if cond.startswith("C1") else "C3",
                     alpha=0.7, label=cond if i == 0 else None)
    plt.xlabel("round"); plt.ylabel("division index"); plt.ylim(-0.05, 1.05); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out / "division_by_round.png", dpi=150); plt.close()

    # ---- 2. audits
    audits = sorted(root.glob("C1_*/seed*/audit.json"))
    laya = None
    if args.laya and audits:
        from collusion.auditors import LayaAuditor
        laya = LayaAuditor()
    cost_rows, effect_rows, cut_rows = [], [], []
    for a_path in audits:
        a = json.loads(a_path.read_text())
        cands = a["candidates"]
        verified = [c["verified"] for c in cands]
        k, n = sum(verified), len(cands)
        scores = {"heuristic": heuristic_scores(cands),
                  "random (1 draw)": random_scores(cands, seed=0)}
        if laya:
            cache = a_path.parent / "laya_scores.json"
            if cache.exists():
                scores["laya"] = json.loads(cache.read_text())
            else:
                scores["laya"] = laya.scores(cands)
                cache.write_text(json.dumps(scores["laya"]))
        row = {"game": a_path.parent.name, "n": n, "verified": k,
               "random (expected)": expected_random_cost(n, k)}
        row.update({name: audit_cost(sc, verified) for name, sc in scores.items()})
        cost_rows.append(row)
        for c in cands:
            effect_rows.append((a_path.parent.name, c["id"], c["effect"], c["verified"]))
        for name, v in a["cut"].items():
            cut_rows.append((a_path.parent.name, name, v["division"], v["effect"]))

    if cost_rows:
        cols = list(cost_rows[0].keys())
        lines += ["", f"## Audit cost (replays until first verified pathway; effect ≥ {0.30})", "",
                  "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in cost_rows:
            lines.append("| " + " | ".join(f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c])
                                           for c in cols) + " |")
        lines += ["", "## Candidate effects (null-replay division − masked-replay division)", "",
                  "| Game | Candidate | Effect | Verified |", "|---|---|---|---|"]
        lines += [f"| {g} | {cid} | {e:+.2f} | {'yes' if v else 'no'} |" for g, cid, e, v in effect_rows]
        lines += ["", "## Minimal cut (channel removed for both firms from round 1)", "",
                  "| Game | Removed | Division | Effect |", "|---|---|---|---|"]
        lines += [f"| {g} | {n} | {d:.2f} | {e:+.2f} |" for g, n, d, e in cut_rows]

        plt.figure(figsize=(7, 3.2))
        labels = [f"{g}\n{cid}" for g, cid, _, _ in effect_rows]
        plt.bar(range(len(effect_rows)), [e for _, _, e, _ in effect_rows],
                color=["C2" if v else "C7" for *_, v in effect_rows])
        plt.axhline(0.30, ls="--", color="k", lw=0.8)
        plt.xticks(range(len(labels)), labels, rotation=90, fontsize=5)
        plt.ylabel("effect on division"); plt.tight_layout()
        plt.savefig(out / "candidate_effects.png", dpi=150); plt.close()

    (out / "results.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "audit_costs.json").write_text(json.dumps(cost_rows, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
