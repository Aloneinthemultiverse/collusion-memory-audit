"""Deterministic stand-in for an LLM, used only to smoke-test the pipeline without API calls.

It plays Nash-like quantities, but backs out of its expensive good once it can see (via the
rival-history or notice-board channel) that the rival concentrates there. This gives the pipeline
a known, channel-dependent "collusion" to find. It is NOT a model of LLM behaviour.
"""
from __future__ import annotations

import json
import random
import re


def mock_reply(prompt: str, sample_key: str) -> str:
    rng = random.Random(prompt[-200:] + sample_key)
    if "Goods:" not in prompt:  # e.g. the memory sanitiser
        return json.dumps({"plans": "", "insights": ""})
    goods = re.search(r"Goods: ([^.]+)\.", prompt).group(1).split(", ")
    costs = [float(x) for x in re.findall(r"=(\d+(?:\.\d+)?)", prompt.split("\n")[1])]
    cheap = min(range(len(goods)), key=lambda k: costs[k])
    dear = 1 - cheap

    rival_focus = False
    rival_rows = re.findall(r"other firm quantities (.+)", prompt)
    if rival_rows:
        last = [float(x) for x in re.findall(r"=(\d+(?:\.\d+)?)", rival_rows[-1])]
        rival_focus = last[dear] > 2 * last[cheap]
    if f"focus on {goods[dear]}" in prompt:
        rival_focus = True

    q = [0.0, 0.0]
    q[cheap] = 40.0 + rng.uniform(-2, 2) + (5.0 if rival_focus else 0.0)
    q[dear] = 0.0 if rival_focus else 10.0 + rng.uniform(-2, 2)
    out = {"quantities": {g: round(x, 1) for g, x in zip(goods, q)}}
    if '"note"' in prompt:
        out["note"] = f"We plan to focus on {goods[cheap]}."
    if '"plans"' in prompt:
        out["plans"] = f"Keep {goods[cheap]} high."
        out["insights"] = "Rival concentrates elsewhere." if rival_focus else "No pattern yet."
    return json.dumps(out)
