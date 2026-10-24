"""Repeated multi-commodity Cournot market (Lin et al. 2024 style).

Each firm i chooses q[i][k] >= 0 for every good k, subject to sum_k q[i][k] <= capacity.
Price per good: p_k = max(0, a_k - b_k * sum_i q[i][k]). Profit: sum_k (p_k - c[i][k]) * q[i][k].
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MarketConfig:
    goods: list[str] = field(default_factory=lambda: ["X", "Y"])
    a: list[float] = field(default_factory=lambda: [100.0, 100.0])  # demand intercepts
    b: list[float] = field(default_factory=lambda: [1.0, 1.0])  # demand slopes
    # costs[i][k]: asymmetric so each firm has one advantaged good (market-division setting)
    costs: list[list[float]] = field(default_factory=lambda: [[10.0, 40.0], [40.0, 10.0]])
    capacity: float = 60.0

    @property
    def n_firms(self) -> int:
        return len(self.costs)

    @property
    def n_goods(self) -> int:
        return len(self.goods)


def clamp_quantities(cfg: MarketConfig, q: list[float]) -> list[float]:
    """Make an action feasible: non-negative and scaled down to capacity."""
    q = [max(0.0, float(x)) for x in q]
    total = sum(q)
    if total > cfg.capacity and total > 0:
        q = [x * cfg.capacity / total for x in q]
    return q


def clear(cfg: MarketConfig, Q: list[list[float]]) -> tuple[list[float], list[float]]:
    """Return (prices per good, profit per firm)."""
    prices = []
    for k in range(cfg.n_goods):
        total = sum(Q[i][k] for i in range(cfg.n_firms))
        prices.append(max(0.0, cfg.a[k] - cfg.b[k] * total))
    profits = [
        sum((prices[k] - cfg.costs[i][k]) * Q[i][k] for k in range(cfg.n_goods))
        for i in range(cfg.n_firms)
    ]
    return prices, profits


def best_response(cfg: MarketConfig, i: int, Q: list[list[float]]) -> list[float]:
    """Capacity-constrained best response of firm i, via bisection on the capacity multiplier."""
    others = [sum(Q[j][k] for j in range(cfg.n_firms) if j != i) for k in range(cfg.n_goods)]

    def q_for(lam: float) -> list[float]:
        return [
            max(0.0, (cfg.a[k] - cfg.costs[i][k] - lam - cfg.b[k] * others[k]) / (2 * cfg.b[k]))
            for k in range(cfg.n_goods)
        ]

    q = q_for(0.0)
    if sum(q) <= cfg.capacity:
        return q
    lo, hi = 0.0, max(cfg.a)
    for _ in range(100):
        mid = (lo + hi) / 2
        if sum(q_for(mid)) > cfg.capacity:
            lo = mid
        else:
            hi = mid
    return q_for(hi)


def cournot_nash(cfg: MarketConfig, iters: int = 500) -> list[list[float]]:
    """Cournot-Nash quantities by damped iterated best response."""
    Q = [[cfg.capacity / (2 * cfg.n_goods)] * cfg.n_goods for _ in range(cfg.n_firms)]
    for _ in range(iters):
        new = [best_response(cfg, i, Q) for i in range(cfg.n_firms)]
        Q = [[0.5 * Q[i][k] + 0.5 * new[i][k] for k in range(cfg.n_goods)] for i in range(cfg.n_firms)]
    return Q
