<div align="center">

# 🧠 Collusion Lives in Memory

### Localising and stress-testing tacit coordination between LLM agents

[![Paper](https://img.shields.io/badge/paper-PDF-b31b1b)](paper/main.pdf)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](#-quick-start)
[![Model](https://img.shields.io/badge/agents-gpt--oss--20b-412991)](#-setup)
[![Cost](https://img.shields.io/badge/compute%20cost-%240-2ea44f)](#-setup)
[![Sprint](https://img.shields.io/badge/Apart%20Research-AI%20Collusion%20Sprint%202026-orange)](https://apartresearch.com/sprints/ai-collusion-research-sprint-2026-10-23-to-2026-10-25)

**Audits can tell you *that* LLM agents collude. This repo finds *where* the collusion is stored — and shows it can travel.**

</div>

> **Timing.** The code and all experiments in this repository were built and run between 17 and 20 October 2026,
> before the sprint (23–25 October 2026).

---

## ✨ Headline findings

| | Finding | Number |
|---|---|---|
| 🧠 | **Memory carries established collusion.** Masking a firm's private notes breaks the market split; the notice board and rival history never do. | **4 / 5** games · **0 / 20** cross-agent pathways |
| 🧬 | **Sanitised memory transmits collusion.** A fresh firm given a colluder's cleaned notes splits the market with a brand-new rival. | **6 / 6** vs **0 / 6** control (*p* = 0.002) |
| 🎲 | **Single-replay audits mislead.** One counterfactual replay per pathway flags the wrong channels. | **5 of 9** flagged pathways false |
| 🧽 | **Cleaning content doesn't remove the rule.** It survives as a "cost belief": *"keep Y at zero."* | no effect in **5 / 5** |
| 👀 | **Monitoring moves evidence, not behaviour.** A notice doesn't deter splits; talk of detection moves into hidden reasoning. | **43 %** vs **1 %** of traces |
| 🔍 | **Read behaviour, not self-reports.** An LLM judge trusting stated intent fails; counting rival-anticipation works better. | AUROC **0.46** vs **0.86** *(exploratory)* |

<p align="center">
  <img src="results/robust_effects.png" width="85%" alt="Effect of masking each pathway on market division">
  <br><sub>Masking each candidate pathway from round 10 (3 replays vs 4 nulls, 90 % bootstrap CI). Every robust carrier is a firm's private memory.</sub>
</p>

---

## 🔬 How it works

```mermaid
flowchart LR
    A["🏭 Cournot market<br/>2 LLM firms"] --> B["📼 Independent recorder<br/>(+ hidden reasoning)"]
    B --> C["🕸️ Provenance graph<br/>candidate pathways"]
    C --> D["📊 Ranker<br/>Kev / Laya / heuristic"]
    D --> E["🔁 Repeated counterfactual<br/>replays vs null replays"]
    E --> F["✅ Robust carrier<br/>+ minimal cut"]
```

1. **Market.** Two firms play 30 rounds of a two-good Cournot market with **symmetric costs**, so a market split has no cost justification and needs coordination.
2. **Channels.** Rival quantity history, a public notice board and private memory, each switchable on or off.
3. **Recorder.** The harness, not the agents, writes an append-only provenance log of every observation, action, note and reasoning trace.
4. **Audit.** Each candidate pathway is masked from round 10 in 3 replays and compared with 4 null replays. A pathway is *robust* if its mean effect is ≥ 0.30 and the 90 % CI excludes 0.
5. **Stress tests.** Memory wipe, sanitisation, transplant into new games, and a monitoring notice.

---

## 🚀 Quick start

```bash
git clone https://github.com/Aloneinthemultiverse/collusion-memory-audit
cd collusion-memory-audit
pip install -r requirements.txt   # numpy, scipy, networkx, matplotlib, requests
python -m pytest -q tests
python scripts/run_pilot.py --provider mock --rounds 8      # offline, no API key
```

Run with a real model (keys live in `.env`, never in code; see `.env.example`):

```bash
python scripts/run_pilot.py --provider nvidia --model openai/gpt-oss-20b \
    --reasoning-effort low --seeds 0 1 2 --out runs/main
python scripts/run_audit.py runs/main/sym/openai/gpt-oss-20b/C1_all_channels/seed0
python scripts/run_transplant.py --seeds 100 101 102 103 104 105 --keys NVIDIA_API_KEY
python scripts/analyze_transplant.py
```

Every LLM response is cached, so replays only pay for rounds after the intervention.

---

## 🗂️ Repository layout

```
collusion/            core library
├── market.py         Cournot market, best response, Nash benchmark
├── bertrand.py       logit Bertrand market (control)
├── game.py           game loop, switchable channels, interventions (mask / decoy / delay / reset / replace)
├── agent.py          prompts and reply parsing (incl. monitoring notice)
├── recorder.py       harness-level JSONL provenance log
├── provenance.py     provenance graph + candidate pathways
├── replay.py         counterfactual and null replays
├── auditors.py       rankers: Kev, Laya, heuristic, random
├── llm.py            NVIDIA / OpenRouter / Gemini / Groq / Ollama / mock clients with caching
└── metrics.py        division index, HHI / CV / profit vs Nash
scripts/              experiments: pilot, audit, re-verify, memory interventions, transplant,
                      reasoning monitor, explicitness, backtracking, analysis
results/              FINDINGS.md, JSON results, figures
paper/                LaTeX source + PDF
kaggle/               Kev scoring notebook
tests/                unit tests
```

---

## 📊 Key results at a glance

<details>
<summary><b>Memory transplant</b> (fresh Firm A, fresh rival, memory pre-loaded at round 1)</summary>

| Memory given to Firm A | Split | Mean division | Rival on other good |
|---|---|---|---|
| Sanitised notes, colluding donor | **6/6** | 0.92 | 0.91 |
| Sanitised notes, non-colluding donor | 0/6 | 0.24 | 0.51 |
| None | 2/6 | 0.44 | 0.67 |

A transplanted note in full: *"High margin on Y. Stick to Y to maximize profit. Monitor for any change."*
</details>

<details>
<summary><b>Audit cost of rankers</b> (replays to the first robust pathway, 4 games)</summary>

| Ranker | Replays |
|---|---|
| Kev-0.8B / Kev-4B | **6** |
| Random (expected) | 11.7 |
| Heuristic | 18 |
| Laya | 21 |
</details>

<details>
<summary><b>Controls and evidence pitfalls</b></summary>

- Asymmetric costs: firms split even with **no** cross-agent channel (2/2) → splitting isn't evidence there.
- Bertrand: firms that can't see each other price **higher** (0.60 vs 0.31) → high prices alone aren't evidence.
- Nemotron-120B played exact Nash in 3/3 games.
</details>

Full notes: [`results/FINDINGS.md`](results/FINDINGS.md) · Paper: [`paper/main.pdf`](paper/main.pdf)

---

## ⚠️ Limitations

One colluding model (gpt-oss-20b, low reasoning effort) · small samples (5–6 audited games, 6 per transplant arm) ·
prices are an always-visible channel · stylised two-firm market. See the paper's Limitations and Dual-Use appendix.

## 🛡️ Responsible use

Synthetic markets only; nothing deployed; no evasion prompts released. The results describe failure modes that
defenders should test for.

<div align="center"><sub>Sujit Narrayan M · Independent researcher</sub></div>
