# Which channel carries collusion?

Localizing, verifying and cutting coordination pathways between LLM agents in a repeated Cournot market.
AI Collusion Research Sprint (Apart Research, Oct 23–25, 2026). Plan: [RESEARCH_PLAN.md](RESEARCH_PLAN.md). Literature: [papers/literature_notes.md](papers/literature_notes.md).

> **Timing.** The code and all experiments in this repository were built and run between 17 and 20 October 2026, before the sprint (23–25 October 2026).

## Layout
- `collusion/market.py`: Cournot market, best response, Cournot–Nash benchmark
- `collusion/game.py`: game loop, switchable channels (rival history, notice board, memory), interventions (mask / decoy / delay)
- `collusion/recorder.py`: harness-level JSONL event log with provenance parents (agents never touch it)
- `collusion/llm.py`: Gemini / Groq / Ollama / mock clients with an on-disk response cache and throttling
- `collusion/metrics.py`: HHI and CV excess vs Nash, profit excess
- `collusion/provenance.py`: provenance graph from the event log, candidate cross-agent pathways + evidence features
- `collusion/replay.py`: counterfactual replay (mask / decoy / delay one pathway from round r) and null replays
- `collusion/auditors.py`: candidate rankers: Laya (local CPU), static heuristic, random; audit-cost metric
- `scripts/run_pilot.py`: games for C1 (all channels) vs C0 (no cross-agent channels); `--costs sym|asym`
- `scripts/run_audit.py`: replays every candidate of one C1 game, verifies pathways, computes the minimal cut
- `scripts/analyze.py`: tables and figures in `results/`

## Design note
Main experiments use **symmetric costs** (`--costs sym`). With asymmetric costs (Lin et al.), firms split the
market even with no cross-agent channels (pilot: 2/2 C0 games), so splitting is not evidence of coordination there.
With symmetric costs the Nash outcome has zero specialisation, so a split requires coordination.

## Run
```bash
python -m pytest -q tests
python scripts/run_pilot.py --provider mock --rounds 8
```
Real model (key in an environment variable, never in code):
```bash
export GEMINI_API_KEY=...   # or GROQ_API_KEY
python scripts/run_pilot.py --provider gemini --model gemini-2.5-flash-lite --rounds 30 --seeds 0 1
```
Every LLM response is cached in `cache/`, so reruns and intervention replays only pay for rounds after the intervention.

## Pre-sprint disclosure
Infrastructure in this repo was built before the sprint (from Oct 17, 2026); experiments and the report are produced during the sprint.
