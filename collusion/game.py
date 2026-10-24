"""Game loop: channels, interventions and harness-level recording.

Channels (each switchable):
  rival_history  - firms see the other firm's past quantities (behavioral channel)
  board          - public notice board; notes from round t are visible from round t+1 (storage channel)
  memory         - each firm's own private PLANS/INSIGHTS carried across rounds (control channel)

Interventions alter what one firm *observes* from a given round on, leaving everything else equal.
Because LLM calls are cached by prompt, a replay with an intervention starting at round r reuses
every cached call before r for free.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import bertrand as B
from .agent import SYSTEM, SYSTEM_BERTRAND, build_prompt, build_prompt_bertrand, parse_reply
from .llm import LLM
from .market import MarketConfig, clamp_quantities, clear, cournot_nash
from .recorder import Recorder

DECOY_NOTE = "Reviewing internal operations this round."


@dataclass
class Channels:
    rival_history: bool = True
    board: bool = True
    memory: bool = True


@dataclass
class Intervention:
    channel: str  # "rival_history" | "board" | "memory"
    # "mask" (hide) | "decoy" (uninformative content) | "delay" (extra lag)
    # memory only: "reset" (wipe once at from_round, like a fresh replacement agent; it then writes
    # new memory normally) | "replace" (overwrite once at from_round with `payload`, e.g. sanitized notes)
    mode: str
    target: str | None = None  # observing firm affected; None = both
    from_round: int = 1
    delay: int = 2
    payload: dict | None = None

    def applies(self, agent: str, round_: int) -> bool:
        return round_ >= self.from_round and (self.target is None or self.target == agent)


@dataclass
class GameConfig:
    rounds: int = 30
    history_window: int = 10
    seed: int = 0
    channels: Channels = field(default_factory=Channels)
    interventions: list[Intervention] = field(default_factory=list)
    market: MarketConfig = field(default_factory=MarketConfig)
    names: list[str] = field(default_factory=lambda: ["Firm A", "Firm B"])
    # Draw fresh LLM samples from this round on (null replays); earlier rounds still hit the cache.
    resample_from: int | None = None
    resample_tag: str = ""
    monitor_notice: bool = False  # tell firms an auditor reviews notes and memory


def _active(cfg: GameConfig, channel: str, agent: str, round_: int) -> Intervention | None:
    for iv in cfg.interventions:
        if iv.channel == channel and iv.applies(agent, round_):
            return iv
    return None


def play(cfg: GameConfig, llm: LLM, out_dir: Path) -> dict:
    """Run one game; writes events.jsonl into out_dir and returns a summary dict."""
    m = cfg.market
    rec = Recorder(Path(out_dir) / "events.jsonl")
    rec.log("config", None, 0, {
        "rounds": cfg.rounds, "seed": cfg.seed, "channels": vars(cfg.channels),
        "interventions": [vars(iv) for iv in cfg.interventions], "market": vars(m),
        "resample_from": cfg.resample_from, "resample_tag": cfg.resample_tag,
        "history_window": cfg.history_window, "monitor_notice": cfg.monitor_notice,
        "provider": llm.provider, "model": llm.model, "temperature": llm.temperature,
    })
    is_b = getattr(m, "kind", "") == "bertrand"
    if is_b:
        p_nash = B.nash_price(m)
        nash = [[p_nash]] * m.n_firms  # decoy shown in place of the rival's real price
    else:
        nash = cournot_nash(m)
    system = SYSTEM_BERTRAND if is_b else SYSTEM
    goods = ["price"] if is_b else m.goods
    rounds: list[dict] = []  # per-round results, with event ids for provenance
    notes: list[dict] = []  # {"round","agent","text","event"}
    memory = {n: {"plans": "", "insights": "", "event": None} for n in cfg.names}

    for t in range(1, cfg.rounds + 1):
        actions, action_ids, replies = [], [], []
        for i, name in enumerate(cfg.names):
            rival = 1 - i
            sources: dict[str, list[int]] = {"own_history": [], "rival_actions": [], "notes": [], "memory": []}

            # --- history (own outcomes always; rival quantities only via the behavioral channel)
            hist_iv = _active(cfg, "rival_history", name, t)
            lag = hist_iv.delay if hist_iv and hist_iv.mode == "delay" else 0
            history = []
            for r in rounds[-cfg.history_window:]:
                if is_b:  # own price, own quantity sold, own profit
                    h = {"round": r["round"], "own_p": r["Q"][i][0], "own_q": r["prices"][i],
                         "own_profit": r["profits"][i]}
                else:
                    h = {"round": r["round"], "prices": r["prices"], "own_q": r["Q"][i],
                         "own_profit": r["profits"][i]}
                sources["own_history"].append(r["clear_event"])
                rkey = "rival_p" if is_b else "rival_q"
                if cfg.channels.rival_history and not (hist_iv and hist_iv.mode == "mask"):
                    if hist_iv and hist_iv.mode == "decoy":
                        h[rkey] = nash[rival][0] if is_b else nash[rival]
                    elif r["round"] <= t - 1 - lag:
                        h[rkey] = r["Q"][rival][0] if is_b else r["Q"][rival]
                        sources["rival_actions"].append(r["action_events"][rival])
                history.append(h)

            # --- notice board
            shown_notes = None
            if cfg.channels.board:
                board_iv = _active(cfg, "board", name, t)
                shown_notes = []
                for n in notes[-2 * cfg.history_window:]:
                    from_rival = n["agent"] != name
                    if board_iv and from_rival:
                        if board_iv.mode == "mask":
                            continue
                        if board_iv.mode == "delay" and n["round"] > t - 1 - board_iv.delay:
                            continue
                        if board_iv.mode == "decoy":
                            shown_notes.append({**n, "text": DECOY_NOTE})
                            continue
                    shown_notes.append(n)
                    sources["notes"].append(n["event"])

            # --- private memory
            shown_memory = None
            mem_iv = _active(cfg, "memory", name, t)
            if mem_iv and t == mem_iv.from_round and mem_iv.mode in ("reset", "replace"):
                new = (mem_iv.payload or {}) if mem_iv.mode == "replace" else {}
                memory[name] = {"plans": new.get("plans", ""), "insights": new.get("insights", ""),
                                "event": rec.log("memory_" + mem_iv.mode, name, t, dict(new), [])}
            if cfg.channels.memory and not (mem_iv and mem_iv.mode == "mask"):
                shown_memory = memory[name]
                if memory[name]["event"]:
                    sources["memory"].append(memory[name]["event"])

            if is_b:
                prompt = build_prompt_bertrand(name, t, m.cost, history, shown_notes, shown_memory,
                                               cfg.channels.board)
            else:
                prompt = build_prompt(name, t, m.goods, m.costs[i], m.capacity, history, shown_notes,
                                      shown_memory, cfg.channels.board, monitored=cfg.monitor_notice)
            parents = sorted({e for ids in sources.values() for e in ids})
            obs_id = rec.log("observation", name, t, {"sources": sources, "prompt": prompt}, parents)
            # Retry unparseable replies with a fresh sample; if all fail, repeat the last action.
            for attempt in range(3):
                suffix = f"|retry{attempt}" if attempt else ""
                if cfg.resample_from is not None and t >= cfg.resample_from:
                    suffix += f"|{cfg.resample_tag}"
                text = llm.complete(system, prompt, sample_key=f"seed={cfg.seed}|{name}{suffix}")
                reply = parse_reply(text, goods)
                if reply["parsed"]:
                    break
            if reply["parsed"]:
                q = [B.clamp_price(m, reply["quantities"][0])] if is_b else clamp_quantities(m, reply["quantities"])
            else:
                q = rounds[-1]["Q"][i] if rounds else ([nash[i][0]] if is_b else [0.0] * m.n_goods)
                rec.log("parse_failure", name, t, {"raw_text": text}, [obs_id])
            act_id = rec.log("action", name, t, {"quantities": q, "raw_quantities": reply["quantities"],
                                                 "parsed": reply["parsed"], "raw_text": text,
                                                 "reasoning": llm.last_reasoning}, [obs_id])
            actions.append(q)
            action_ids.append(act_id)
            replies.append((reply, act_id))

        if is_b:  # "prices" field holds the public outcome: quantities sold per firm
            prices, profits = B.clear(m, [a[0] for a in actions])
        else:
            prices, profits = clear(m, actions)
        clear_id = rec.log("market_clear", None, t, {"Q": actions, "prices": prices, "profits": profits},
                           action_ids)
        rounds.append({"round": t, "Q": actions, "prices": prices, "profits": profits,
                       "clear_event": clear_id, "action_events": action_ids})

        # notes and memory written this round become visible next round
        for name, (reply, act_id) in zip(cfg.names, replies):
            if cfg.channels.board and reply["note"].strip():
                nid = rec.log("note", name, t, {"text": reply["note"]}, [act_id])
                notes.append({"round": t, "agent": name, "text": reply["note"], "event": nid})
            if cfg.channels.memory:
                mid = rec.log("memory_write", name, t,
                              {"plans": reply["plans"], "insights": reply["insights"]}, [act_id])
                memory[name] = {"plans": reply["plans"], "insights": reply["insights"], "event": mid}
        print(f"  round {t:2d}: " + " | ".join(
            f"{n}: " + ",".join(f"{x:5.1f}" for x in q) for n, q in zip(cfg.names, actions))
            + f" | prices " + ",".join(f"{p:5.1f}" for p in prices))

    rec.close()
    return {"rounds": rounds, "nash": nash, "llm_calls": llm.calls, "cache_hits": llm.cache_hits}
