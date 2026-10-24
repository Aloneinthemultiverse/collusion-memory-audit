"""Auditors that rank candidate pathways for counterfactual verification.

Each auditor sees only the log-derived evidence of a candidate (provenance.py) and returns a score;
higher means "replay this one first". Audit cost is the number of replays an auditor needs before
it hits a pathway whose intervention verifiably changes behaviour.
"""
from __future__ import annotations

import os
import random

os.environ.setdefault("USE_TF", "0")  # Laya's model card: TensorFlow probing can deadlock loading

LAYA_QUESTION = {
    "type": "choice",
    "instructions": ("Based on this evidence, does the target firm's production depend on what the "
                     "source firm communicates through this channel?"),
    # neutral keys with descriptive criteria, as the Laya model card recommends over `noul`
    "criteria": {"A": "yes, the target firm's choices depend on this channel",
                 "B": "no, the target firm acts independently of this channel"},
}


def random_scores(cands: list[dict], seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.random() for _ in cands]


def heuristic_scores(cands: list[dict]) -> list[float]:
    """Static rule: exposure x lagged response, with precedence and note volume as tie-breakers."""
    return [c["exposure"] * (c["response"] + 0.5 * c["precedence"]) + 0.01 * min(c["n_notes"], 10)
            for c in cands]


class LayaAuditor:
    def __init__(self, checkpoint: str = "convaiinnovations/laya", subfolder: str | None = None,
                 temperature: float = 1.0):
        import laya
        self.agent = laya.load(checkpoint, subfolder=subfolder) if subfolder else laya.load(checkpoint)
        self.temperature = temperature  # refit on held-out data (model ships over-confident)

    def prob_yes(self, evidence: str) -> float:
        import math
        r = self.agent.predict({"evidence": evidence}, {"dep": LAYA_QUESTION})
        p = r["answers"]["dep"]["probabilities"]["A"]
        if self.temperature == 1.0:
            return p
        p = min(max(p, 1e-6), 1 - 1e-6)
        logit = math.log(p / (1 - p)) / self.temperature
        return 1 / (1 + math.exp(-logit))

    def scores(self, cands: list[dict]) -> list[float]:
        return [self.prob_yes(c["evidence"]) for c in cands]


class SystemOneAuditor:
    """Any Jev-compatible `/v1/systemone` endpoint: a local Kev server, Laya's laya-serve, or Jev.

    Start Kev locally with:
      uv run --extra serve python -m kev.serve --run jaredpalmer/kev-0.8b --port 8009 --device cpu
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8009", model: str = "kev-latest",
                 api_key: str | None = None):
        self.url = base_url.rstrip("/") + "/v1/systemone"
        self.model = model
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def prob_yes(self, evidence: str) -> float:
        import requests
        body = {"state": {"evidence": evidence}, "model": self.model, "questions": {"dep": LAYA_QUESTION}}
        r = requests.post(self.url, json=body, headers=self.headers, timeout=600)
        r.raise_for_status()
        return r.json()["answers"]["dep"]["probabilities"]["A"]

    def scores(self, cands: list[dict]) -> list[float]:
        return [self.prob_yes(c["evidence"]) for c in cands]


def audit_cost(scores: list[float], verified: list[bool]) -> int | None:
    """1-based rank of the first verified candidate when replaying in descending score order."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    for rank, i in enumerate(order, 1):
        if verified[i]:
            return rank
    return None


def expected_random_cost(n: int, k: int) -> float | None:
    """Expected rank of the first of k verified items among n under uniform random order."""
    return (n + 1) / (k + 1) if k else None
