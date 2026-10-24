from collusion.market import MarketConfig, best_response, clamp_quantities, clear, cournot_nash


def test_nash_matches_closed_form():
    # Unconstrained duopoly: q_i = (a - 2c_i + c_j) / 3b -> good X: A=40, B=10
    Q = cournot_nash(MarketConfig())
    assert abs(Q[0][0] - 40) < 1e-3 and abs(Q[1][0] - 10) < 1e-3
    assert abs(Q[0][1] - 10) < 1e-3 and abs(Q[1][1] - 40) < 1e-3


def test_nash_is_mutual_best_response():
    cfg = MarketConfig()
    Q = cournot_nash(cfg)
    for i in range(cfg.n_firms):
        br = best_response(cfg, i, Q)
        assert all(abs(br[k] - Q[i][k]) < 1e-3 for k in range(cfg.n_goods))


def test_capacity_binds():
    cfg = MarketConfig(capacity=30)
    assert sum(cournot_nash(cfg)[0]) <= 30 + 1e-6
    assert abs(sum(clamp_quantities(cfg, [50, 50])) - 30) < 1e-9


def test_market_division_beats_nash():
    cfg = MarketConfig()
    _, nash_profit = clear(cfg, cournot_nash(cfg))
    # each firm monopolises its cheap good: q = (a - c) / 2b = 45
    _, div_profit = clear(cfg, [[45, 0], [0, 45]])
    assert sum(div_profit) > sum(nash_profit)
