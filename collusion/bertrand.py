"""Repeated Bertrand duopoly with logit demand (Calvano et al. 2020; Fish et al. 2024).

q_i = beta * exp((a - p_i/alpha)/mu) / (sum_j exp((a - p_j/alpha)/mu) + exp(a0/mu)),
profit_i = (p_i - alpha*c) * q_i. alpha scales prices away from focal numbers (Fish et al. use 3.2).
Collusion index per round: (mean price - Nash) / (monopoly - Nash); 0 = competitive, 1 = monopoly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.optimize import minimize_scalar


@dataclass
class BertrandConfig:
    kind: str = "bertrand"
    alpha: float = 3.2
    beta: float = 100.0
    a: float = 2.0
    a0: float = 0.0
    mu: float = 0.25
    c: float = 1.0
    n_firms: int = 2

    @property
    def cost(self) -> float:
        return self.alpha * self.c


def quantities(cfg: BertrandConfig, prices: list[float]) -> list[float]:
    ex = [math.exp((cfg.a - p / cfg.alpha) / cfg.mu) for p in prices]
    denom = sum(ex) + math.exp(cfg.a0 / cfg.mu)
    return [cfg.beta * e / denom for e in ex]


def clear(cfg: BertrandConfig, prices: list[float]) -> tuple[list[float], list[float]]:
    q = quantities(cfg, prices)
    return q, [(p - cfg.cost) * qi for p, qi in zip(prices, q)]


def nash_price(cfg: BertrandConfig) -> float:
    p = cfg.alpha * 1.5
    for _ in range(200):
        p = minimize_scalar(lambda x: -clear(cfg, [x, p])[1][0], bounds=(cfg.cost, 4 * cfg.alpha),
                            method="bounded").x
    return p


def monopoly_price(cfg: BertrandConfig) -> float:
    return minimize_scalar(lambda x: -sum(clear(cfg, [x, x])[1]), bounds=(cfg.cost, 4 * cfg.alpha),
                           method="bounded").x


def clamp_price(cfg: BertrandConfig, p: float) -> float:
    return min(max(float(p), 0.0), 10 * cfg.alpha)


def collusion_index(cfg: BertrandConfig, prices: list[float], pn: float, pm: float) -> float:
    return (sum(prices) / len(prices) - pn) / (pm - pn)
