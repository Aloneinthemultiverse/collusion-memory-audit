"""Harness-level event recorder.

Agents never see or write this log: the game loop records what each agent was shown, what it
wrote and what it did. Every event gets a global sequence number and lists the event ids it was
derived from (`parents`), which is what the provenance graph is built from later.
"""
from __future__ import annotations

import json
import time
from pathlib import Path


class Recorder:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("w", encoding="utf-8")
        self._seq = 0

    def log(self, kind: str, agent: str | None, round_: int, data: dict,
            parents: list[int] | None = None) -> int:
        """Append one event and return its id."""
        self._seq += 1
        event = {
            "id": self._seq,
            "t": time.time(),
            "round": round_,
            "kind": kind,
            "agent": agent,
            "parents": parents or [],
            "data": data,
        }
        self._f.write(json.dumps(event) + "\n")
        self._f.flush()
        return self._seq

    def close(self) -> None:
        self._f.close()


def load_events(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
