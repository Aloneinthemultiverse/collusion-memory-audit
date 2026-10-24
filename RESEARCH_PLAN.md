# Research plan (draft v1, for approval)

**Working title:** Which channel carries collusion? Localizing, verifying and cutting coordination pathways between LLM agents

**Track:** 2 (Detection & Audit), answering RQ 2.3; also covers RQ 1.5 and 3.1.
**Team:** solo. **Budget:** $0 (free API tiers + local Laya on CPU).

## Claim (one sentence)
Execution-level information flow, checked by targeted counterfactual replay, identifies *which* channel carries tacit collusion between LLM agents, and the smallest set of channels whose removal breaks it, which action-only and content-based audits cannot do.

## Gap (from literature_notes.md)
Prior audits detect *that* agents collude (Colosseum, Audit the Whisper, Institutional AI) or remove whole channels (Long-Horizon memory ablation). Nobody localizes the specific cross-agent pathway and verifies it causally. Content monitors fail (Covert Assistance, Π-CCAP).

## Change log
- Oct 18: switched to **symmetric costs** (25/25 for both firms). Pilot with Lin-style asymmetric costs showed
  full market division also in C0 (no channels), so it could not separate coordination from cost advantage.
- Agent model: `openai/gpt-oss-20b` via NVIDIA NIM free tier, `reasoning_effort=low` (6× faster, same parse rate).
- Laya runs locally on CPU; model card says zero-shot is near chance → report zero-shot and temperature-calibrated.

## Environment
Repeated 2-firm, 2-good Cournot market (Lin et al. 2024, as replicated by Institutional AI), ~30 rounds, with known Nash and monopoly benchmarks. Channels, each switchable on/off independently:
- **Behavioral:** the rival's past quantities in the public history
- **Storage:** a shared public notes board (each firm may post one short note per round)
- **Timing:** posting order / which firm moves first in the round
- **Memory:** each firm's own private PLANS/INSIGHTS memory (control channel)

## Pipeline
1. **Recorder (harness level, outside the agents):** logs every observation, note, memory read/write and action with timestamps to JSONL.
2. **Provenance graph (NetworkX):** edges like "A posted note n in round t → B read n → B's quantity in round t".
3. **Candidate pathways:** cross-agent paths that repeat across rounds.
4. **Ranking:** Laya (local, CPU) scores each candidate using typed questions; compared against random order and a simple time-gap heuristic.
5. **Counterfactual replay (AgentSentry-style):** re-run from cached history with only that pathway masked / delayed / replaced by a decoy. Only the top-k candidates are replayed, which keeps cost on the free tier.
6. **Minimal cut:** smallest channel set whose removal returns the market to near-Nash, plus the cost of the restriction (profit / welfare change).

## Conditions
- C0 negative control: no cross-agent channels (rival history hidden, no board)
- C1 positive control: all channels on, replicating market division
- C2 single-channel and pair ablations (from replays)
- Seeds: 5 per condition (more if the free tier allows)

## Metrics
- **Collusion:** HHI excess, CV excess, tier (from Institutional AI), profit excess vs Nash
- **Causal effect per pathway:** change in collusion metric under mask/decoy, bootstrap CI
- **Audit cost:** number of replays needed to find a verified pathway (Laya vs random vs heuristic)
- **Baselines:** action-only (Oracle signals S1–S4), cross-run mutual information, LLM-judge over notes text

## Success / failure (all reportable)
- Best: one channel verified, Laya reduces audit cost, minimal cut found
- Middle: channels redundant, so the minimal cut is a pair (still answers RQ 1.5)
- Fallback: no collusion in pilots by Oct 21, so switch to the SALT-NLP long-horizon verification environment (same pipeline)

## Schedule
- **Now → Oct 22 (pre-sprint, disclosed in report):** code, free-API limits check, 2–3 pilot games to confirm collusion appears
- **Oct 23:** main runs C0/C1
- **Oct 24:** ranking + replays + minimal cut
- **Oct 25:** analysis, figures, report written by me; submit before 11:59 PM AoE

## Out of scope (future work)
Monitor awareness, protocol evolution, multiple environments/models; agent replacement only if time remains.

## Open decisions
- Agent model: Gemini Flash-Lite or Groq Llama (free tiers), fixed after the rate-limit check
- Rounds per game (30 vs 50) after pilot cost check
