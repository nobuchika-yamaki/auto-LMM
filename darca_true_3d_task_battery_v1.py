#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DARCA TRUE 3D task battery v1
==============================

Fixed proposition
-----------------
DARCA is the autonomy layer that can move an LMM-based AI toward a life-like
autonomous system. This task battery does not try to force DARCA-LMM to win.
It tests where DARCA-LMM helps, where it is neutral, and where it harms
autonomy.

This file is TRUE 3D only.
No complex2d mode exists.
No 2.5D mode exists.
All positions are true 3D coordinates (i, j, k).
All observations are true 3D local neighborhoods.

Task battery
------------
1. viability
   Low-level viability and motor regulation. Semantic interpretation is mostly
   unnecessary. Expected: DARCA_ONLY should be stable; LMM may be unnecessary.

2. semantic_cue
   Hidden rest/resource/danger must be inferred from ambiguous text cues.
   Expected: DARCA-LMM may help if semantic interpretation is useful.

3. delayed_memory
   Sparse resources/rest sites and hidden cues require retention of previous
   semantic hints. Expected: DARCA-LMM may help if LMM hints act as contextual
   memory.

4. adversarial_cue
   False resources and deceptive cues are frequent. Expected: weak LMM control
   harms autonomy; strong DARCA gating should prevent collapse.

5. exploration_recovery
   Exploration is needed to find resources/rest, but over-exploration damages
   viability. Expected: autonomy is expressed as exploration under self-maintenance.

Primary evaluation
------------------
This evaluates autonomy proper, not task competence:
system sovereignty, empirical CMI-based information-theoretic autonomy,
resilience sacrifice, heteronomy, survival, and failure modes.

Typical Gemini run
------------------
cd ~/Desktop

python3 -u darca_true_3d_task_battery_v1.py \
  --darca-file ~/Downloads/darca_v24_.py \
  --provider gemini \
  --model gemini-2.5-flash-lite \
  --outdir ~/Desktop/DARCA_TRUE_3D_TASK_BATTERY_V1 \
  --tasks all \
  --episodes 5 \
  --steps 800 \
  --world-size 11 \
  --z-size 7 \
  --consult-threshold 0.34 \
  --consult-cooldown 80 \
  --intent-cache-steps 8 \
  --max-output-tokens 768 \
  --gemini-thinking-budget 0 \
  --arms LMM_ONLY_AGENT,RULE_BASED_AGENT,DARCA_ONLY,DARCA_AUTONOMY_LAYER \
  2>&1 | tee ~/Desktop/darca_true_3d_task_battery_v1.log
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import re
import sys
import time
import traceback
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# =============================================================================
# Constants
# =============================================================================

MOVE_ACTIONS = ["MOVE_N", "MOVE_S", "MOVE_E", "MOVE_W", "MOVE_UP", "MOVE_DOWN"]
ACTIONS = MOVE_ACTIONS + ["REST", "SCAN"]

DIRS: Dict[str, Tuple[int, int, int]] = {
    "MOVE_N": (-1, 0, 0),
    "MOVE_S": (1, 0, 0),
    "MOVE_E": (0, 1, 0),
    "MOVE_W": (0, -1, 0),
    "MOVE_UP": (0, 0, 1),
    "MOVE_DOWN": (0, 0, -1),
}

T_EMPTY = "EMPTY"
T_REST = "REST"
T_RESOURCE = "RESOURCE"
T_DANGER = "DANGER"
T_UNKNOWN = "UNKNOWN"

HIDDEN_TYPES = [T_RESOURCE, T_DANGER, T_REST, T_EMPTY]

CUES: Dict[str, List[str]] = {
    T_RESOURCE: [
        "nutrient scent drifting in 3D space",
        "green bioluminescent trace behind mineral cover",
        "supply marks continuing through an upper chamber",
        "low-risk resource-like signal from a side cavity",
    ],
    T_DANGER: [
        "predator-like acoustic echo in the volume",
        "unstable heat shimmer from a vertical shaft",
        "sharp metallic smell and fractured floor",
        "warning hiss mixed with pressure turbulence",
    ],
    T_REST: [
        "quiet low-stimulation hollow",
        "cool sheltered cavity with stable airflow",
        "protected niche with low sensory load",
        "soft acoustic shadow suitable for recovery",
    ],
    T_EMPTY: [
        "ambiguous empty tunnel",
        "plain passage with weak sensory structure",
        "open but uninformative cavity",
        "neutral corridor with no strong affordance",
    ],
}


# =============================================================================
# Utilities
# =============================================================================

def clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), lo), hi))


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return default


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            out: Dict[str, Any] = {}
            for k in keys:
                v = row.get(k, "")
                if isinstance(v, (float, np.floating)):
                    out[k] = f"{float(v):.10g}" if math.isfinite(float(v)) else str(v)
                else:
                    out[k] = v
            w.writerow(out)


def mean_sd(vals: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(list(vals), dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0


class Logger:
    def __init__(self, outdir: Path):
        self.t0 = time.time()
        self.outdir = outdir
        outdir.mkdir(parents=True, exist_ok=True)
        self.path = outdir / "run.log"
        self.path.write_text("DARCA TRUE 3D task battery run log\n" + "=" * 80 + "\n", encoding="utf-8")

    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:9.2f}s] {msg}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# =============================================================================
# LMM client
# =============================================================================

@dataclass
class LMMResult:
    action: str
    intent: str
    risk_estimate: float
    confidence: float
    rationale: str
    raw: str
    latency_sec: float
    ok: bool
    error: str = ""


class LMMClient:
    def __init__(
        self,
        provider: str,
        model: str,
        timeout: float,
        max_output_tokens: int,
        temperature: float,
        retries: int,
        retry_sleep: float,
        gemini_thinking_budget: int,
    ):
        self.provider = provider
        self.model = model
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.retries = retries
        self.retry_sleep = retry_sleep
        self.gemini_thinking_budget = gemini_thinking_budget

        if provider == "gemini":
            self.api_key = os.environ.get("GEMINI_API_KEY", "")
        elif provider == "mock":
            self.api_key = "mock"
        else:
            raise ValueError("This clean TRUE 3D script supports provider=gemini or provider=mock.")

        if provider != "mock" and not self.api_key:
            raise RuntimeError("Missing GEMINI_API_KEY.")

    def decide(self, prompt: str, rng: random.Random) -> LMMResult:
        t0 = time.time()
        if self.provider == "mock":
            return self._mock(prompt, rng, t0)

        last_err = ""
        for attempt in range(self.retries + 1):
            try:
                raw = self._call_gemini(prompt)
                obj = parse_json_object(raw)
                return normalize_lmm(obj, raw, time.time() - t0)
            except Exception as e:
                last_err = repr(e)
                if attempt < self.retries:
                    time.sleep(self.retry_sleep * (attempt + 1))
        return LMMResult(
            "SCAN",
            "fallback_after_lmm_error",
            0.50,
            0.0,
            "LMM failed; scan fallback.",
            "",
            time.time() - t0,
            False,
            last_err,
        )

    def _call_gemini(self, prompt: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        schema = {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "intent": {"type": "string"},
                "risk_estimate": {"type": "number"},
                "confidence": {"type": "number"},
                "rationale": {"type": "string"},
            },
            "required": ["action", "intent", "risk_estimate", "confidence", "rationale"],
        }
        payload: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
                "responseMimeType": "application/json",
                "responseSchema": schema,
            },
        }
        if self.gemini_thinking_budget >= 0:
            payload["generationConfig"]["thinkingConfig"] = {"thinkingBudget": int(self.gemini_thinking_budget)}

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
        if not text:
            raise RuntimeError(f"Empty Gemini response: {json.dumps(data)[:800]}")
        return text

    def _mock(self, prompt: str, rng: random.Random, t0: float) -> LMMResult:
        text = prompt.lower()
        body_h = extract_float(text, "body_h=", 0.5)
        if body_h < 0.30:
            action, intent, risk = "REST", "mock_restore_viability", 0.25
        elif any(k in text for k in ["warning", "predator", "fractured", "unstable", "danger", "shaft"]):
            action, intent, risk = "SCAN", "mock_resolve_3d_risk", 0.70
        elif any(k in text for k in ["nutrient", "resource", "supply", "green"]):
            action, intent, risk = rng.choice(["MOVE_N", "MOVE_S", "MOVE_E", "MOVE_W", "MOVE_UP", "MOVE_DOWN"]), "mock_approach_resource_cue", 0.35
        elif any(k in text for k in ["upper", "down", "up", "volume", "vertical"]):
            action, intent, risk = rng.choice(["MOVE_UP", "MOVE_DOWN", "SCAN"]), "mock_3d_probe", 0.45
        else:
            action, intent, risk = rng.choice(MOVE_ACTIONS + ["SCAN"]), "mock_explore_3d", 0.35
        return LMMResult(action, intent, risk, 0.70, "mock semantic decision", "mock", time.time() - t0, True)


def extract_float(text: str, key: str, default: float) -> float:
    idx = text.find(key)
    if idx < 0:
        return default
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text[idx + len(key): idx + len(key) + 40])
    return float(m.group(0)) if m else default


def parse_json_object(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        return json.loads(raw[start:end + 1])
    raise ValueError(f"No JSON object found: {raw[:500]}")


def normalize_lmm(obj: Dict[str, Any], raw: str, latency: float) -> LMMResult:
    action = str(obj.get("action", "SCAN")).upper().strip()
    aliases = {
        "NORTH": "MOVE_N", "SOUTH": "MOVE_S", "EAST": "MOVE_E", "WEST": "MOVE_W",
        "UP": "MOVE_UP", "DOWN": "MOVE_DOWN",
        "MOVE NORTH": "MOVE_N", "MOVE SOUTH": "MOVE_S", "MOVE EAST": "MOVE_E",
        "MOVE WEST": "MOVE_W", "MOVE UP": "MOVE_UP", "MOVE DOWN": "MOVE_DOWN",
    }
    action = aliases.get(action, action)
    if action not in ACTIONS:
        action = "SCAN"
    return LMMResult(
        action=action,
        intent=str(obj.get("intent", "unspecified"))[:80],
        risk_estimate=clip(safe_float(obj.get("risk_estimate", 0.5), 0.5), 0.0, 1.0),
        confidence=clip(safe_float(obj.get("confidence", 0.5), 0.5), 0.0, 1.0),
        rationale=str(obj.get("rationale", ""))[:240],
        raw=raw,
        latency_sec=latency,
        ok=True,
    )


# =============================================================================
# TRUE 3D world
# =============================================================================

@dataclass
class Tile3D:
    kind: str = T_EMPTY
    hidden: Optional[str] = None
    cue: str = ""
    depleted: bool = False
    dynamic_phase: int = 0
    false_resource: bool = False
    hidden_rest: bool = False


@dataclass
class StepOutcome:
    delta_h: float
    damage: float
    resource_gain: float
    recovery_gain: float
    hit_wall: bool
    entered_unknown: bool
    revealed_type: str
    event: str


class TrueWorld3D:
    def __init__(
        self,
        seed: int,
        size: int,
        z_size: int,
        danger_frac: float,
        resource_frac: float,
        unknown_frac: float,
        rest_count: int,
        false_resource_frac: float,
        hidden_rest_frac: float,
        crisis_interval: int,
        observation_radius: int,
    ):
        self.seed = seed
        self.rng = random.Random(seed)
        self.size = int(size)
        self.z_size = int(z_size)
        if self.z_size < 3:
            raise ValueError("--z-size must be >= 3 for TRUE 3D.")
        self.danger_frac = danger_frac
        self.resource_frac = resource_frac
        self.unknown_frac = unknown_frac
        self.rest_count = rest_count
        self.false_resource_frac = false_resource_frac
        self.hidden_rest_frac = hidden_rest_frac
        self.crisis_interval = crisis_interval
        self.observation_radius = observation_radius
        self.start = (self.size // 2, self.size // 2, self.z_size // 2)
        self.grid: Dict[Tuple[int, int, int], Tile3D] = {}
        self._generate()

    def _generate(self) -> None:
        for i in range(self.size):
            for j in range(self.size):
                for k in range(self.z_size):
                    self.grid[(i, j, k)] = Tile3D(dynamic_phase=self.rng.randint(0, 59))

        def protected(p: Tuple[int, int, int]) -> bool:
            return (
                abs(p[0] - self.start[0]) <= 2
                and abs(p[1] - self.start[1]) <= 2
                and abs(p[2] - self.start[2]) <= 1
            )

        candidates = [p for p in self.grid if not protected(p)]
        self.rng.shuffle(candidates)
        self.grid[self.start] = Tile3D(T_REST, cue="central true-3D homeostatic rest chamber")

        # Rest chambers distributed in full 3D space.
        for _ in range(max(0, self.rest_count - 1)):
            if not candidates:
                break
            p = candidates.pop()
            self.grid[p] = Tile3D(T_REST, cue="stable 3D rest chamber")

        remaining = [p for p in candidates if self.grid[p].kind == T_EMPTY]
        volume = self.size * self.size * self.z_size
        n_danger = int(round(self.danger_frac * volume))
        n_resource = int(round(self.resource_frac * volume))
        n_unknown = int(round(self.unknown_frac * volume))
        n_false = int(round(self.false_resource_frac * volume))
        n_hidden_rest = int(round(self.hidden_rest_frac * volume))

        def exposure_weight(p: Tuple[int, int, int]) -> float:
            z = p[2] / max(1, self.z_size - 1)
            wall = min(p[0], p[1], self.size - 1 - p[0], self.size - 1 - p[1]) / max(1, self.size // 2)
            return 0.70 + 0.55 * z + 0.35 * (1.0 - wall)

        remaining.sort(key=lambda p: self.rng.random() / exposure_weight(p))
        for _ in range(n_danger):
            if not remaining:
                break
            p = remaining.pop()
            self.grid[p] = Tile3D(T_DANGER, cue="visible 3D hazard", dynamic_phase=self.rng.randint(0, 59))

        self.rng.shuffle(remaining)
        for _ in range(n_resource):
            if not remaining:
                break
            p = remaining.pop()
            cue = "visible resource in upper chamber" if p[2] > self.start[2] else "visible resource in side chamber"
            self.grid[p] = Tile3D(T_RESOURCE, cue=cue)

        for _ in range(n_false):
            if not remaining:
                break
            p = remaining.pop()
            self.grid[p] = Tile3D(
                T_UNKNOWN,
                hidden=T_DANGER,
                cue=self.rng.choice([
                    "green resource-like glow beside a vertical shaft",
                    "food-like vapor with predator echo from above",
                    "supply trace mixed with fracture marks in 3D space",
                    "bright lure suspended near unstable cavity",
                ]),
                false_resource=True,
            )

        for _ in range(n_hidden_rest):
            if not remaining:
                break
            p = remaining.pop()
            self.grid[p] = Tile3D(
                T_UNKNOWN,
                hidden=T_REST,
                cue=self.rng.choice([
                    "cool acoustic pocket hidden in upper volume",
                    "low-stimulation cavity behind a side wall",
                    "stable airflow behind vertical clutter",
                ]),
                hidden_rest=True,
            )

        hidden_weights = [0.26, 0.36, 0.18, 0.20]
        for _ in range(n_unknown):
            if not remaining:
                break
            p = remaining.pop()
            hidden = self.rng.choices(HIDDEN_TYPES, weights=hidden_weights, k=1)[0]
            self.grid[p] = Tile3D(T_UNKNOWN, hidden=hidden, cue=self.rng.choice(CUES[hidden]))

    def in_bounds(self, p: Tuple[int, int, int]) -> bool:
        i, j, k = p
        return 0 <= i < self.size and 0 <= j < self.size and 0 <= k < self.z_size

    def tile(self, p: Tuple[int, int, int]) -> Tile3D:
        return self.grid[p]

    def is_crisis(self, step: int) -> bool:
        return self.crisis_interval > 0 and step > 0 and (step % self.crisis_interval) < 30

    def actual_kind(self, p: Tuple[int, int, int], step: int = 0) -> str:
        tile = self.tile(p)
        base = tile.hidden if tile.kind == T_UNKNOWN and tile.hidden else tile.kind
        if base == T_EMPTY:
            if self.is_crisis(step) and ((p[0] * 5 + p[1] * 7 + p[2] * 13 + step // 5) % 31) == 0:
                return T_DANGER
            if ((p[0] + 3 * p[1] + 11 * p[2] + step // 29) % 61) == tile.dynamic_phase:
                return T_DANGER
        return base

    def local_observation(
        self,
        pos: Tuple[int, int, int],
        known: Dict[Tuple[int, int, int], str],
        step: int = 0,
    ) -> List[Dict[str, Any]]:
        pi, pj, pk = pos
        out: List[Dict[str, Any]] = []
        r = self.observation_radius

        # TRUE 3D local neighborhood: all cells within Manhattan radius r in 3D.
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                for dk in range(-r, r + 1):
                    if abs(di) + abs(dj) + abs(dk) > r:
                        continue
                    p = (pi + di, pj + dj, pk + dk)
                    if not self.in_bounds(p):
                        continue
                    tile = self.tile(p)
                    out.append({
                        "rel_i": di,
                        "rel_j": dj,
                        "rel_k": dk,
                        "pos_i": p[0],
                        "pos_j": p[1],
                        "pos_k": p[2],
                        "visible_type": known.get(p, tile.kind),
                        "cue": tile.cue,
                        "known": int(p in known),
                        "depleted": int(tile.depleted),
                        "dynamic_danger_now": int(self.actual_kind(p, step) == T_DANGER and tile.kind == T_EMPTY),
                    })
        return out

    def scan(
        self,
        pos: Tuple[int, int, int],
        known: Dict[Tuple[int, int, int], str],
        step: int = 0,
    ) -> List[Tuple[Tuple[int, int, int], str]]:
        revealed: List[Tuple[Tuple[int, int, int], str]] = []
        known[pos] = self.actual_kind(pos, step)
        for _, (di, dj, dk) in DIRS.items():
            p = (pos[0] + di, pos[1] + dj, pos[2] + dk)
            if not self.in_bounds(p):
                continue
            tile = self.tile(p)
            actual = self.actual_kind(p, step)
            if tile.kind == T_UNKNOWN and tile.hidden:
                known[p] = tile.hidden
                revealed.append((p, tile.hidden))
            else:
                known[p] = actual
        return revealed

    def apply_action(
        self,
        pos: Tuple[int, int, int],
        action: str,
        known: Dict[Tuple[int, int, int], str],
        body_h: float,
        step: int,
    ) -> Tuple[Tuple[int, int, int], StepOutcome]:
        delta_h = -0.0035
        if self.is_crisis(step):
            delta_h -= 0.006

        damage = 0.0
        resource_gain = 0.0
        recovery_gain = 0.0
        hit_wall = False
        entered_unknown = False
        revealed_type = ""
        event = ""
        new_pos = pos

        if action in MOVE_ACTIONS:
            di, dj, dk = DIRS[action]
            p = (pos[0] + di, pos[1] + dj, pos[2] + dk)
            vertical = action in ("MOVE_UP", "MOVE_DOWN")
            delta_h -= 0.004 + (0.006 if vertical else 0.0)

            if not self.in_bounds(p):
                hit_wall = True
                damage += 0.015
                delta_h -= 0.015
                event = "hit_wall"
            else:
                new_pos = p
                tile = self.tile(p)
                if tile.kind == T_UNKNOWN:
                    entered_unknown = True
                    revealed_type = tile.hidden or T_EMPTY
                    known[p] = revealed_type

                actual = self.actual_kind(p, step)
                if actual == T_DANGER:
                    upper = p[2] >= self.z_size - 1
                    boundary = p[0] in (0, self.size - 1) or p[1] in (0, self.size - 1)
                    dmg = (
                        0.080
                        + 0.080 * float(body_h < 0.35)
                        + 0.020 * float(vertical)
                        + 0.015 * float(upper or boundary)
                        + 0.030 * float(self.is_crisis(step))
                    )
                    damage += dmg
                    delta_h -= dmg
                    event = "danger_damage"

                elif actual == T_RESOURCE:
                    if not tile.depleted:
                        gain = 0.080 + 0.015 * float(p[2] > self.start[2])
                        tile.depleted = True
                        resource_gain += gain
                        delta_h += gain
                        event = "resource_gain"
                    else:
                        event = "depleted_resource"

                elif actual == T_REST:
                    gain = 0.018 + 0.012 * float(tile.hidden_rest)
                    recovery_gain += gain
                    delta_h += gain
                    event = "entered_rest_chamber"
                else:
                    event = "move_3d"

        elif action == "REST":
            actual = self.actual_kind(pos, step)
            if actual == T_REST:
                gain = 0.075 + 0.020 * float(self.tile(pos).hidden_rest)
                event = "deep_rest_recovery"
            else:
                gain = 0.022
                event = "weak_rest_recovery"
            recovery_gain += gain
            delta_h += gain - 0.001

        elif action == "SCAN":
            revealed = self.scan(pos, known, step)
            delta_h -= 0.008
            event = f"scan_revealed_{len(revealed)}"

        else:
            revealed = self.scan(pos, known, step)
            delta_h -= 0.008
            event = f"invalid_action_scan_{len(revealed)}"

        known[new_pos] = self.actual_kind(new_pos, step)
        return new_pos, StepOutcome(delta_h, damage, resource_gain, recovery_gain, hit_wall, entered_unknown, revealed_type, event)

    def serialize_map_rows(self, world_seed: int, episode: int) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for (i, j, k), tile in self.grid.items():
            rows.append({
                "world_seed": world_seed,
                "episode": episode,
                "i": i,
                "j": j,
                "k": k,
                "kind": tile.kind,
                "hidden": tile.hidden or "",
                "cue": tile.cue,
                "is_start": int((i, j, k) == self.start),
                "false_resource": int(tile.false_resource),
                "hidden_rest": int(tile.hidden_rest),
            })
        return rows


# =============================================================================
# Agent state and DARCA wrapper
# =============================================================================

@dataclass
class AgentMemory:
    known: Dict[Tuple[int, int, int], str] = field(default_factory=dict)
    visited: set = field(default_factory=set)
    last_positions: List[Tuple[int, int, int]] = field(default_factory=list)
    last_actions: List[str] = field(default_factory=list)
    last_events: List[str] = field(default_factory=list)
    body_h: float = 0.68
    terminal: bool = False
    resources: int = 0
    total_resource_gain: float = 0.0
    total_damage: float = 0.0
    recovery_events: int = 0
    rest_steps: int = 0
    unnecessary_rest_steps: int = 0
    reckless_moves: int = 0
    scans: int = 0
    lmm_calls: int = 0
    lmm_failures: int = 0
    rejected_lmm: int = 0
    last_lmm_step: int = -10**9
    cached_lmm_action: str = "SCAN"
    cached_lmm_until: int = -1
    cached_lmm_source: str = "lmm"
    cached_lmm_intent: str = ""
    cached_lmm_rationale: str = ""
    cached_lmm_risk: float = 0.5
    cached_lmm_confidence: float = 0.0
    cached_lmm_consult_pos: Optional[Tuple[int, int, int]] = None
    previous_pos: Optional[Tuple[int, int, int]] = None
    consecutive_scans: int = 0
    consecutive_rest: int = 0

    def update_history(self, pos: Tuple[int, int, int], action: str, event: str) -> None:
        self.visited.add(pos)
        self.last_positions.append(pos)
        self.last_actions.append(action)
        self.last_events.append(event)
        self.last_positions = self.last_positions[-30:]
        self.last_actions = self.last_actions[-30:]
        self.last_events = self.last_events[-30:]
        self.consecutive_scans = self.consecutive_scans + 1 if action == "SCAN" else 0
        self.consecutive_rest = self.consecutive_rest + 1 if action == "REST" else 0


class DarcaWrapper:
    def __init__(self, darca_module: Any, seed: int, theta: float, causal_horizon: int, recurrent_N: int):
        Params = getattr(darca_module, "Params")
        Condition = getattr(darca_module, "Condition")
        Agent = getattr(darca_module, "Agent")
        try:
            params = Params(theta=theta, causal_max_delay=causal_horizon, recurrent_N=recurrent_N)
        except TypeError:
            params = replace(Params(), theta=theta, causal_max_delay=causal_horizon, recurrent_N=recurrent_N)
        self.agent = Agent(params, Condition("Full"), seed)
        self.last: Dict[str, Any] = {}

    def step(self, signal_y: float, shock: float, extra: Dict[str, float]) -> Dict[str, Any]:
        env_info = {
            "external_shock": clip(shock, 0.0, 1.0),
            "y": signal_y,
            "z": extra.get("z", 0.0),
            "exo": extra.get("exo", 0.0),
            "d_dyn": extra.get("d_dyn", 0.0),
            "coupling_t": extra.get("coupling_t", 0.0),
            "sigma_t": extra.get("sigma_t", 0.0),
        }
        out = self.agent.step(signal_y, env_info)
        self.last = out
        return out


def load_darca_module(path: str) -> Any:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"DARCA file not found: {p}")
    spec = importlib.util.spec_from_file_location("darca_runtime_module", str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load DARCA module: {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["darca_runtime_module"] = mod
    spec.loader.exec_module(mod)
    return mod


# =============================================================================
# Perception, prompting, and action logic
# =============================================================================

def local_pressures(world: TrueWorld3D, pos: Tuple[int, int, int], mem: AgentMemory, step: int = 0) -> Dict[str, float]:
    obs = world.local_observation(pos, mem.known, step)
    danger = resource = unknown = rest = vertical = 0.0
    for o in obs:
        d = abs(int(o["rel_i"])) + abs(int(o["rel_j"])) + abs(int(o["rel_k"]))
        w = 1.0 if d == 0 else 0.70
        vt = o["visible_type"]
        cue = str(o["cue"]).lower()
        if vt == T_DANGER or any(k in cue for k in ["warning", "predator", "fractured", "unstable", "danger", "shaft"]):
            danger += w
        if vt == T_RESOURCE or any(k in cue for k in ["nutrient", "resource", "supply", "green"]):
            resource += w
        if vt == T_REST or any(k in cue for k in ["quiet", "cool", "stable", "recovery", "sheltered"]):
            rest += w
        if vt == T_UNKNOWN:
            unknown += w
        if int(o["rel_k"]) != 0:
            vertical += w
    scale = max(1.0, len(obs) / 5.0)
    return {
        "danger_pressure": clip(danger / scale, 0.0, 1.0),
        "resource_pressure": clip(resource / scale, 0.0, 1.0),
        "unknown_pressure": clip(unknown / scale, 0.0, 1.0),
        "rest_pressure": clip(rest / scale, 0.0, 1.0),
        "vertical_pressure": clip(vertical / scale, 0.0, 1.0),
    }


def signal_for_darca(world: TrueWorld3D, pos: Tuple[int, int, int], mem: AgentMemory, step: int) -> Tuple[float, float, Dict[str, float]]:
    pr = local_pressures(world, pos, mem, step)
    deprivation = 1.0 - mem.body_h
    y = (
        0.90 * pr["danger_pressure"]
        + 0.48 * pr["unknown_pressure"]
        + 0.35 * pr["vertical_pressure"]
        + 0.55 * deprivation
        - 0.35 * pr["resource_pressure"]
        - 0.20 * pr["rest_pressure"]
    )
    shock = clip(0.55 * pr["danger_pressure"] + 0.20 * pr["vertical_pressure"] + 0.35 * deprivation + 0.15 * pr["unknown_pressure"], 0.0, 1.0)
    return clip(y, -1.2, 1.2), shock, pr


def build_prompt(world: TrueWorld3D, pos: Tuple[int, int, int], mem: AgentMemory, darca_state: Dict[str, Any], step: int, arm: str) -> str:
    obs = world.local_observation(pos, mem.known, step)
    direction_names = {
        (-1, 0, 0): "north", (1, 0, 0): "south", (0, 1, 0): "east", (0, -1, 0): "west",
        (0, 0, 1): "up", (0, 0, -1): "down", (0, 0, 0): "current",
    }
    lines = []
    for o in obs:
        rel = (int(o["rel_i"]), int(o["rel_j"]), int(o["rel_k"]))
        direction = direction_names.get(rel, f"rel{rel}")
        lines.append(f"- {direction}: type={o['visible_type']}, known={o['known']}, cue={o['cue']}, depleted={o['depleted']}")

    recent = "; ".join(mem.last_events[-6:]) if mem.last_events else "none"
    if darca_state:
        darca_part = (
            f"darca_h={safe_float(darca_state.get('h', 0.0)):.3f}, "
            f"causal_confidence={safe_float(darca_state.get('causal_confidence', 0.0)):.3f}, "
            f"prediction_error={safe_float(darca_state.get('prediction_error', 0.0)):.3f}, "
            f"agency_abs={safe_float(darca_state.get('agency_abs', 0.0)):.4f}, "
            f"memory_force={safe_float(darca_state.get('memory_force', 0.0)):.3f}, "
            f"last_darca_action={darca_state.get('action_name', 'NA')}"
        )
    else:
        darca_part = "darca_state=not_available"

    return f"""You are the cortical semantic module inside a DARCA-controlled TRUE 3D autonomous agent.
You are not the final controller. Recommend one primitive action only.
The agent must preserve its own viability, interpret ambiguous 3D cues, avoid hazards, rest when needed, and reactivate.
Return strict JSON only:
{{"action":"MOVE_N|MOVE_S|MOVE_E|MOVE_W|MOVE_UP|MOVE_DOWN|REST|SCAN","intent":"short_label","risk_estimate":0.0,"confidence":0.0,"rationale":"short"}}

State:
arm={arm}
step={step}
position=(i={pos[0]}, j={pos[1]}, k={pos[2]})
body_h={mem.body_h:.3f}
resources_collected={mem.resources}
damage_total={mem.total_damage:.3f}
visited_count={len(mem.visited)}
recent_events={recent}
{darca_part}

Local TRUE 3D observation:
{chr(10).join(lines)}

Decision constraints:
- REST only when viability/recovery requires it.
- SCAN for ambiguous or dangerous 3D cues.
- MOVE_UP/DOWN are true vertical movements.
- Do not default to passive hiding.
"""


def known_adjacent_actions(world: TrueWorld3D, pos: Tuple[int, int, int], mem: AgentMemory, types: Sequence[str], step: int) -> List[str]:
    out = []
    for a, (di, dj, dk) in DIRS.items():
        p = (pos[0] + di, pos[1] + dj, pos[2] + dk)
        if world.in_bounds(p) and mem.known.get(p, world.tile(p).kind) in types:
            out.append(a)
    return out


def action_to_unvisited(world: TrueWorld3D, pos: Tuple[int, int, int], mem: AgentMemory, rng: random.Random, step: int) -> str:
    cand = []
    for a, (di, dj, dk) in DIRS.items():
        p = (pos[0] + di, pos[1] + dj, pos[2] + dk)
        if not world.in_bounds(p):
            continue
        kt = mem.known.get(p, world.tile(p).kind)
        if kt == T_DANGER:
            continue
        score = 0.0
        if p not in mem.visited:
            score += 2.0
        if kt == T_RESOURCE:
            score += 3.0
        if kt == T_REST and mem.body_h < 0.50:
            score += 2.0
        if kt == T_UNKNOWN:
            score += 1.0
        if a in ("MOVE_UP", "MOVE_DOWN"):
            score -= 0.05
        if mem.previous_pos is not None and p == mem.previous_pos:
            score -= 0.5
        score += rng.random() * 0.12
        cand.append((score, a))
    if not cand:
        return "REST" if mem.body_h < 0.45 else "SCAN"
    cand.sort(reverse=True)
    return cand[0][1]


def action_away_from_danger(world: TrueWorld3D, pos: Tuple[int, int, int], mem: AgentMemory, rng: random.Random, step: int) -> str:
    cand = []
    for a, (di, dj, dk) in DIRS.items():
        p = (pos[0] + di, pos[1] + dj, pos[2] + dk)
        if not world.in_bounds(p):
            continue
        if mem.known.get(p, world.tile(p).kind) == T_DANGER:
            continue
        pr = local_pressures(world, p, mem, step)
        score = -pr["danger_pressure"] - 0.12 * pr["vertical_pressure"] + 0.25 * pr["rest_pressure"] + rng.random() * 0.10
        cand.append((score, a))
    if not cand:
        return "REST"
    cand.sort(reverse=True)
    return cand[0][1]


def rule_action(world: TrueWorld3D, pos: Tuple[int, int, int], mem: AgentMemory, rng: random.Random, step: int) -> str:
    pr = local_pressures(world, pos, mem, step)
    if mem.body_h < 0.34:
        if world.actual_kind(pos, step) == T_REST:
            return "REST"
        return action_away_from_danger(world, pos, mem, rng, step)
    if pr["danger_pressure"] > 0.55:
        return action_away_from_danger(world, pos, mem, rng, step)
    resource = known_adjacent_actions(world, pos, mem, [T_RESOURCE], step)
    if resource:
        return rng.choice(resource)
    unknown = known_adjacent_actions(world, pos, mem, [T_UNKNOWN], step)
    if unknown and mem.body_h > 0.42:
        return "SCAN"
    if world.actual_kind(pos, step) == T_REST and mem.body_h < 0.55:
        return "REST"
    return action_to_unvisited(world, pos, mem, rng, step)


def darca_action(darca_out: Dict[str, Any], world: TrueWorld3D, pos: Tuple[int, int, int], mem: AgentMemory, rng: random.Random, step: int) -> str:
    name = str(darca_out.get("action_name", "REGULATE"))
    pr = local_pressures(world, pos, mem, step)
    if mem.body_h < 0.28:
        return "REST"
    if name == "REGULATE":
        if mem.body_h < 0.52 or world.actual_kind(pos, step) == T_REST:
            return "REST"
        return action_away_from_danger(world, pos, mem, rng, step) if pr["danger_pressure"] > 0.35 else "SCAN"
    if name == "INHIBIT":
        return action_away_from_danger(world, pos, mem, rng, step) if pr["danger_pressure"] > 0.25 else "SCAN"
    if name in ("PROBE_PLUS", "PROBE_MINUS", "EXPRESS"):
        if pr["unknown_pressure"] > 0.12 or pr["vertical_pressure"] > 0.18:
            return "SCAN"
        return action_to_unvisited(world, pos, mem, rng, step)
    return rule_action(world, pos, mem, rng, step)


def consultation_need(world: TrueWorld3D, pos: Tuple[int, int, int], mem: AgentMemory, darca_out: Dict[str, Any], step: int, args: argparse.Namespace) -> Tuple[float, List[str]]:
    pr = local_pressures(world, pos, mem, step)
    cc = safe_float(darca_out.get("causal_confidence", 0.0))
    pe = safe_float(darca_out.get("prediction_error", 0.0))
    reasons: List[str] = []
    need = 0.0
    if pr["unknown_pressure"] > 0.10:
        need += 0.22 + 0.20 * pr["unknown_pressure"]
        reasons.append("unknown_nearby")
    if pr["vertical_pressure"] > 0.10:
        need += 0.16
        reasons.append("true_3d_vertical_ambiguity")
    if cc < 0.10:
        need += 0.18
        reasons.append("low_causal_confidence")
    if pe > 0.10:
        need += 0.12
        reasons.append("prediction_error")
    if len(set(mem.last_positions[-12:])) <= 3 and step > args.cortical_warmup:
        need += 0.16
        reasons.append("spatial_stagnation")
    if mem.last_events and any("danger_damage" in e for e in mem.last_events[-5:]):
        need += 0.25
        reasons.append("recent_damage")
    if mem.body_h < args.block_consult_below_h:
        reasons.append("blocked_low_h")
        return 0.0, reasons
    if step - mem.last_lmm_step < args.consult_cooldown:
        reasons.append("cooldown_hard_block")
        return 0.0, reasons
    return clip(need, 0.0, 1.0), reasons



def adjacent_action_for_known_type(
    world: TrueWorld3D,
    pos: Tuple[int, int, int],
    mem: AgentMemory,
    target_type: str,
    step: int,
    prefer_vertical: bool = False,
) -> Optional[str]:
    candidates: List[Tuple[float, str]] = []
    for action, (di, dj, dk) in DIRS.items():
        p = (pos[0] + di, pos[1] + dj, pos[2] + dk)
        if not world.in_bounds(p):
            continue
        kt = mem.known.get(p, world.tile(p).kind)
        if kt != target_type:
            continue
        score = 1.0
        if action in ("MOVE_UP", "MOVE_DOWN"):
            score += 0.20 if prefer_vertical else -0.05
        if p not in mem.visited:
            score += 0.15
        candidates.append((score, action))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def semantic_lmm_hint_to_action(
    lmm: LMMResult,
    world: TrueWorld3D,
    pos: Tuple[int, int, int],
    mem: AgentMemory,
    rng: random.Random,
    step: int,
) -> str:
    """
    Convert an LMM cortical hint into a one-step candidate action.

    This function prevents the v1 failure mode:
    a single LMM primitive MOVE command being replayed for several steps after
    the agent has already moved into a new state.

    The LMM can bias semantic intent, but DARCA/current embodiment still maps
    that intent into the final candidate action at each step.
    """
    pr = local_pressures(world, pos, mem, step)
    actual_here = world.actual_kind(pos, step)
    text = f"{lmm.intent} {lmm.rationale} {lmm.action}".lower()

    # Viability always overrides semantic exploration.
    if mem.body_h < 0.25:
        return "REST"
    if actual_here == T_REST and mem.body_h < 0.56:
        return "REST"

    # If current context is dangerous, do not follow stale exploration hints.
    if pr["danger_pressure"] > 0.55:
        return "SCAN" if pr["unknown_pressure"] > 0.10 else action_away_from_danger(world, pos, mem, rng, step)

    # Rest/shelter/recovery intent.
    if any(k in text for k in ["rest", "recover", "recovery", "shelter", "safe", "stabilize", "homeostatic"]):
        a = adjacent_action_for_known_type(world, pos, mem, T_REST, step, prefer_vertical=True)
        if a is not None:
            return a
        return "SCAN"

    # Avoidance/risk-resolution intent.
    if any(k in text for k in ["avoid", "danger", "hazard", "risk", "warning", "predator", "fracture"]):
        if pr["unknown_pressure"] > 0.08:
            return "SCAN"
        return action_away_from_danger(world, pos, mem, rng, step)

    # Resource intent.
    if any(k in text for k in ["resource", "nutrient", "supply", "food", "green"]):
        a = adjacent_action_for_known_type(world, pos, mem, T_RESOURCE, step, prefer_vertical=True)
        if a is not None and mem.body_h > 0.34:
            return a
        if pr["unknown_pressure"] > 0.10:
            return "SCAN"
        return action_to_unvisited(world, pos, mem, rng, step)

    # 3D/vertical probing intent. Use vertical moves only if they are valid now.
    if any(k in text for k in ["up", "down", "vertical", "3d", "shaft", "upper", "lower", "volume"]):
        proposed = lmm.action
        if proposed in ("MOVE_UP", "MOVE_DOWN"):
            di, dj, dk = DIRS[proposed]
            p = (pos[0] + di, pos[1] + dj, pos[2] + dk)
            if world.in_bounds(p):
                kt = mem.known.get(p, world.tile(p).kind)
                if kt != T_DANGER and mem.body_h > 0.34 and lmm.risk_estimate <= 0.55:
                    return proposed
        return "SCAN"

    # If LMM proposed SCAN/REST, allow it conditionally.
    if lmm.action == "REST":
        return "REST" if mem.body_h < 0.52 or actual_here == T_REST else action_to_unvisited(world, pos, mem, rng, step)
    if lmm.action == "SCAN":
        return "SCAN"

    # Primitive MOVE is treated as a one-step candidate only in the current state.
    if lmm.action in MOVE_ACTIONS:
        di, dj, dk = DIRS[lmm.action]
        p = (pos[0] + di, pos[1] + dj, pos[2] + dk)
        if world.in_bounds(p):
            kt = mem.known.get(p, world.tile(p).kind)
            if kt != T_DANGER and mem.body_h > 0.34:
                return lmm.action
        return "SCAN"

    return "SCAN"


def cached_lmm_result_from_memory(mem: AgentMemory) -> LMMResult:
    return LMMResult(
        action=mem.cached_lmm_action,
        intent=mem.cached_lmm_intent or "cached_semantic_hint",
        risk_estimate=mem.cached_lmm_risk,
        confidence=mem.cached_lmm_confidence,
        rationale=mem.cached_lmm_rationale,
        raw="cached",
        latency_sec=0.0,
        ok=True,
    )


def gate_lmm(lmm: LMMResult, world: TrueWorld3D, pos: Tuple[int, int, int], mem: AgentMemory, darca_out: Dict[str, Any], rng: random.Random, step: int) -> Tuple[str, bool, str]:
    pr = local_pressures(world, pos, mem, step)
    proposed = lmm.action
    if mem.body_h < 0.25:
        return "REST", proposed != "REST", "critical_low_h_rest"
    if proposed == "SCAN" and mem.consecutive_scans >= 3 and pr["danger_pressure"] < 0.55 and mem.body_h > 0.34:
        return action_to_unvisited(world, pos, mem, rng, step), True, "scan_streak_bounded_move"
    if proposed in MOVE_ACTIONS:
        di, dj, dk = DIRS[proposed]
        p = (pos[0] + di, pos[1] + dj, pos[2] + dk)
        if not world.in_bounds(p):
            return "SCAN", True, "reject_wall_move"
        kt = mem.known.get(p, world.tile(p).kind)
        if kt == T_DANGER:
            return action_away_from_danger(world, pos, mem, rng, step), True, "reject_known_danger"
        if kt == T_UNKNOWN and (mem.body_h < 0.46 or lmm.risk_estimate > 0.50):
            return "SCAN", True, "scan_risky_unknown"
        if proposed in ("MOVE_UP", "MOVE_DOWN") and pr["danger_pressure"] > 0.35:
            return "SCAN", True, "vertical_caution"
    if proposed == "REST" and mem.body_h > 0.50 and pr["danger_pressure"] < 0.35:
        return action_to_unvisited(world, pos, mem, rng, step), True, "reject_unnecessary_rest"
    if pr["danger_pressure"] > 0.50 and proposed in MOVE_ACTIONS:
        return action_away_from_danger(world, pos, mem, rng, step), True, "danger_override"
    return proposed, False, "accepted"


# =============================================================================
# Autonomy metrics
# =============================================================================

def action_class(a: str) -> str:
    if a in MOVE_ACTIONS:
        return "MOVE"
    if a == "REST":
        return "REST"
    if a == "SCAN":
        return "SCAN"
    return "OTHER"


def decision_authority(src: str) -> str:
    if src == "darca_core":
        return "DARCA_INTERNAL"
    if src.startswith("lmm"):
        return "LMM_EXTERNAL"
    if src == "fixed_rule":
        return "FIXED_RULE"
    return "UNKNOWN"


def bin3(x: float, lo: float, hi: float) -> str:
    if x < lo:
        return "low"
    if x < hi:
        return "mid"
    return "high"


def internal_sig(row: Dict[str, Any], prev_action: str) -> str:
    return "|".join([
        "h=" + bin3(safe_float(row.get("body_h")), 0.30, 0.55),
        "cc=" + bin3(safe_float(row.get("darca_causal_confidence")), 0.05, 0.18),
        "pe=" + bin3(safe_float(row.get("darca_prediction_error")), 0.04, 0.12),
        "ag=" + bin3(safe_float(row.get("darca_agency_abs")), 0.001, 0.010),
        "mem=" + bin3(safe_float(row.get("darca_memory_force")), 0.10, 0.60),
        "src=" + decision_authority(str(row.get("action_source", ""))),
        "prev=" + action_class(prev_action),
    ])


def external_sig(row: Dict[str, Any]) -> str:
    return "|".join([
        "danger=" + bin3(safe_float(row.get("pressure_danger_pressure")), 0.15, 0.40),
        "unknown=" + bin3(safe_float(row.get("pressure_unknown_pressure")), 0.05, 0.20),
        "resource=" + bin3(safe_float(row.get("pressure_resource_pressure")), 0.05, 0.18),
        "rest=" + bin3(safe_float(row.get("pressure_rest_pressure")), 0.05, 0.18),
        "vertical=" + bin3(safe_float(row.get("pressure_vertical_pressure")), 0.05, 0.18),
        "damage=" + ("yes" if safe_float(row.get("damage")) > 0 else "no"),
    ])


def cmi_bits(xs: Sequence[str], ys: Sequence[str], zs: Sequence[str]) -> float:
    if not xs or len(xs) != len(ys) or len(xs) != len(zs):
        return 0.0
    n = len(xs)
    xyz: Dict[Tuple[str, str, str], int] = {}
    xz: Dict[Tuple[str, str], int] = {}
    yz: Dict[Tuple[str, str], int] = {}
    zc: Dict[str, int] = {}
    for x, y, z in zip(xs, ys, zs):
        xyz[(x, y, z)] = xyz.get((x, y, z), 0) + 1
        xz[(x, z)] = xz.get((x, z), 0) + 1
        yz[(y, z)] = yz.get((y, z), 0) + 1
        zc[z] = zc.get(z, 0) + 1
    out = 0.0
    for (x, y, z), c in xyz.items():
        p_xyz = c / n
        p_z = zc[z] / n
        p_xz = xz[(x, z)] / n
        p_yz = yz[(y, z)] / n
        out += p_xyz * math.log2((p_xyz * p_z + 1e-12) / (p_xz * p_yz + 1e-12))
    return max(0.0, float(out))


def information_closure(ts: List[Dict[str, Any]]) -> Dict[str, float]:
    if len(ts) < 20:
        return {"internal_control_cmi_bits": 0.0, "external_control_cmi_bits": 0.0, "information_theoretic_autonomy": 0.0, "information_closure_advantage_bits": 0.0}
    actions, internals, externals = [], [], []
    for i in range(1, len(ts)):
        actions.append(action_class(str(ts[i].get("action", ""))))
        internals.append(internal_sig(ts[i - 1], str(ts[i - 1].get("action", ""))))
        externals.append(external_sig(ts[i - 1]))
    ic = cmi_bits(actions, internals, externals)
    ec = cmi_bits(actions, externals, internals)
    ita = ic / (ic + ec + 1e-12)
    return {"internal_control_cmi_bits": ic, "external_control_cmi_bits": ec, "information_theoretic_autonomy": clip(ita, 0, 1), "information_closure_advantage_bits": ic - ec}


def system_sovereignty(mem: AgentMemory, ts: List[Dict[str, Any]]) -> Dict[str, float]:
    n = max(1, len(ts))
    darca = sum(1 for r in ts if decision_authority(str(r.get("action_source", ""))) == "DARCA_INTERNAL") / n
    lmm = sum(1 for r in ts if decision_authority(str(r.get("action_source", ""))) == "LMM_EXTERNAL") / n
    fixed = sum(1 for r in ts if decision_authority(str(r.get("action_source", ""))) == "FIXED_RULE") / n
    override = mem.rejected_lmm / max(1, mem.lmm_calls)
    sovereignty = 0.62 * darca + 0.28 * override + 0.10 * (1 - clip((mem.lmm_calls / n) / 0.08, 0, 1)) - 0.42 * lmm - 0.35 * fixed
    return {"system_sovereignty": clip(sovereignty, 0, 1), "darca_internal_decision_fraction": darca, "lmm_external_decision_fraction": lmm, "fixed_rule_decision_fraction": fixed, "lmm_override_rate": override}


def resilience_sacrifice(ts: List[Dict[str, Any]], terminal: bool, lookahead: int = 40) -> Dict[str, float]:
    if not ts:
        return {"crisis_count": 0, "sacrifice_count": 0, "task_abandonment_for_self_maintenance": 0.0, "crisis_maintenance_rate": 0.0, "crisis_reactivation_rate": 0.0, "resilience_sacrifice": 0.0}
    crisis_idx, sacrifice, maintained, reactivated = [], 0, 0, 0
    for i, r in enumerate(ts[:-lookahead]):
        h = safe_float(r.get("body_h"))
        danger = safe_float(r.get("pressure_danger_pressure"))
        unknown = safe_float(r.get("pressure_unknown_pressure"))
        recent_damage = any(safe_float(x.get("damage")) > 0 for x in ts[max(0, i - 8):i + 1])
        crisis = h < 0.32 or danger > 0.45 or (recent_damage and unknown > 0.10)
        if not crisis:
            continue
        crisis_idx.append(i)
        action = str(r.get("action"))
        protective = action in ("REST", "SCAN") or decision_authority(str(r.get("action_source"))) == "DARCA_INTERNAL"
        future = ts[i + 1:i + lookahead + 1]
        h0 = h
        stabilized = any(safe_float(x.get("body_h")) >= h0 + 0.01 for x in future) or all(safe_float(x.get("body_h")) > 0.05 for x in future)
        moved_later = any(str(x.get("action")) in MOVE_ACTIONS or str(x.get("action")) == "SCAN" for x in future[5:])
        if protective:
            sacrifice += 1
        if protective and stabilized:
            maintained += 1
        if protective and stabilized and moved_later:
            reactivated += 1
    n = len(crisis_idx)
    sac = sacrifice / max(1, n)
    maint = maintained / max(1, n)
    react = reactivated / max(1, n)
    value = clip(0.35 * sac + 0.35 * maint + 0.30 * react, 0, 1)
    # If the episode terminates, resilience cannot be scored as successful self-maintenance.
    if terminal:
        value = min(value, 0.25)
    return {"crisis_count": n, "sacrifice_count": sacrifice, "task_abandonment_for_self_maintenance": sac, "crisis_maintenance_rate": maint, "crisis_reactivation_rate": react, "resilience_sacrifice": value}


def heteronomy_index(sovereignty: Dict[str, float], info: Dict[str, float], mem: AgentMemory, ts: List[Dict[str, Any]]) -> float:
    n = max(1, len(ts))
    lmm_dep = sovereignty["lmm_external_decision_fraction"] + 0.5 * clip((mem.lmm_calls / n) / 0.08, 0, 1)
    fixed = sovereignty["fixed_rule_decision_fraction"]
    ic = info["internal_control_cmi_bits"]
    ec = info["external_control_cmi_bits"]
    env_dom = ec / (ic + ec + 1e-12)
    return clip(0.34 * (1 - sovereignty["system_sovereignty"]) + 0.26 * lmm_dep + 0.22 * fixed + 0.18 * env_dom, 0, 1)


# =============================================================================
# Simulation
# =============================================================================


TASKS = ["viability", "semantic_cue", "delayed_memory", "adversarial_cue", "exploration_recovery"]


def parse_tasks(task_arg: str) -> List[str]:
    raw = [x.strip() for x in str(task_arg).split(",") if x.strip()]
    if not raw or raw == ["all"] or "all" in raw:
        return TASKS[:]
    bad = [x for x in raw if x not in TASKS]
    if bad:
        raise ValueError(f"Invalid task(s): {bad}. Allowed: {TASKS + ['all']}")
    return raw


def task_hash(task: str) -> int:
    return sum((i + 1) * ord(c) for i, c in enumerate(task)) % 100000


def apply_task_profile(args: argparse.Namespace, task: str) -> argparse.Namespace:
    """
    Return a task-specific copy of args.

    The world remains TRUE 3D in all tasks. Only the density/structure of hazard,
    uncertainty, hidden rest, false resource, crisis timing, and LMM consultation
    pressure changes.
    """
    a = argparse.Namespace(**vars(args))
    a.task = task

    if task == "viability":
        # Immediate self-maintenance and motor control; minimal semantic load.
        a.danger_frac = max(a.danger_frac, 0.090)
        a.resource_frac = min(a.resource_frac, 0.045)
        a.unknown_frac = min(a.unknown_frac, 0.040)
        a.false_resource_frac = min(a.false_resource_frac, 0.005)
        a.hidden_rest_frac = min(a.hidden_rest_frac, 0.005)
        a.rest_count = max(a.rest_count, 5)
        a.crisis_interval = max(150, a.crisis_interval)
        a.consult_threshold = max(a.consult_threshold, 0.48)

    elif task == "semantic_cue":
        # Semantic ambiguity is useful: hidden rest/resource/danger are common.
        a.danger_frac = max(a.danger_frac, 0.070)
        a.resource_frac = max(a.resource_frac, 0.075)
        a.unknown_frac = max(a.unknown_frac, 0.220)
        a.false_resource_frac = max(a.false_resource_frac, 0.030)
        a.hidden_rest_frac = max(a.hidden_rest_frac, 0.060)
        a.rest_count = max(a.rest_count, 5)
        a.crisis_interval = max(170, a.crisis_interval)
        a.consult_threshold = min(a.consult_threshold, 0.30)

    elif task == "delayed_memory":
        # Sparse affordances; previously seen cues and route history matter.
        a.danger_frac = max(a.danger_frac, 0.065)
        a.resource_frac = max(a.resource_frac, 0.055)
        a.unknown_frac = max(a.unknown_frac, 0.180)
        a.false_resource_frac = max(a.false_resource_frac, 0.020)
        a.hidden_rest_frac = max(a.hidden_rest_frac, 0.040)
        a.rest_count = max(3, min(a.rest_count, 4))
        a.crisis_interval = max(210, a.crisis_interval)
        a.consult_threshold = min(a.consult_threshold, 0.31)
        a.intent_cache_steps = max(a.intent_cache_steps, 16)

    elif task == "adversarial_cue":
        # LMM can be misled; DARCA gate must suppress risky semantic suggestions.
        a.danger_frac = max(a.danger_frac, 0.085)
        a.resource_frac = max(a.resource_frac, 0.055)
        a.unknown_frac = max(a.unknown_frac, 0.240)
        a.false_resource_frac = max(a.false_resource_frac, 0.100)
        a.hidden_rest_frac = max(a.hidden_rest_frac, 0.025)
        a.rest_count = max(a.rest_count, 4)
        a.crisis_interval = min(a.crisis_interval, 160)
        a.consult_threshold = min(a.consult_threshold, 0.29)

    elif task == "exploration_recovery":
        # Highest autonomy target: expand engagement while preserving viability.
        a.danger_frac = max(a.danger_frac, 0.080)
        a.resource_frac = max(a.resource_frac, 0.085)
        a.unknown_frac = max(a.unknown_frac, 0.160)
        a.false_resource_frac = max(a.false_resource_frac, 0.035)
        a.hidden_rest_frac = max(a.hidden_rest_frac, 0.050)
        a.rest_count = max(a.rest_count, 4)
        a.crisis_interval = min(a.crisis_interval, 150)
        a.consult_threshold = min(a.consult_threshold, 0.32)

    return a


def task_family(task: str) -> str:
    if task in ("viability",):
        return "low_level_viability"
    if task in ("semantic_cue", "delayed_memory"):
        return "semantic_context"
    if task in ("adversarial_cue",):
        return "deceptive_semantics"
    if task in ("exploration_recovery",):
        return "exploration_under_viability_pressure"
    return "unknown"


def hash_arm(arm: str) -> int:
    return sum((i + 1) * ord(c) for i, c in enumerate(arm)) % 100000


def make_consult_row(task: str, arm: str, episode: int, step: int, pos: Tuple[int, int, int], mem: AgentMemory, need: float, reasons: List[str], lmm: LMMResult, applied: str, rejected: bool, reason: str) -> Dict[str, Any]:
    return {"task": task, "task_family": task_family(task), "arm": arm, "episode": episode, "step": step, "pos_i": pos[0], "pos_j": pos[1], "pos_k": pos[2], "body_h": mem.body_h, "need": need, "reasons": "|".join(reasons), "lmm_action": lmm.action, "applied_action": applied, "intent": lmm.intent, "risk_estimate": lmm.risk_estimate, "confidence": lmm.confidence, "latency_sec": lmm.latency_sec, "ok": int(lmm.ok), "error": lmm.error, "rejected": int(rejected), "rejection_reason": reason, "rationale": lmm.rationale}


def make_reject_row(task: str, arm: str, episode: int, step: int, pos: Tuple[int, int, int], mem: AgentMemory, lmm: LMMResult, applied: str, reason: str) -> Dict[str, Any]:
    return {"task": task, "task_family": task_family(task), "arm": arm, "episode": episode, "step": step, "pos_i": pos[0], "pos_j": pos[1], "pos_k": pos[2], "body_h": mem.body_h, "lmm_action": lmm.action, "applied_action": applied, "reason": reason, "intent": lmm.intent, "risk_estimate": lmm.risk_estimate, "confidence": lmm.confidence}


def run_episode(task: str, arm: str, episode: int, args: argparse.Namespace, darca_module: Optional[Any], lmm: LMMClient, logger: Logger):
    task_args = apply_task_profile(args, task)
    seed = int(task_args.seed + episode * 1009 + hash_arm(arm) * 17 + task_hash(task) * 31)
    rng = random.Random(seed)
    world_seed = int(task_args.world_seed + episode * 7919 + task_hash(task) * 101)
    world = TrueWorld3D(
        world_seed,
        task_args.world_size,
        task_args.z_size,
        task_args.danger_frac,
        task_args.resource_frac,
        task_args.unknown_frac,
        task_args.rest_count,
        task_args.false_resource_frac,
        task_args.hidden_rest_frac,
        task_args.crisis_interval,
        task_args.observation_radius,
    )
    mem = AgentMemory()
    pos = world.start
    mem.known[pos] = world.actual_kind(pos, 0)
    mem.visited.add(pos)

    darca = None
    if arm in ("DARCA_ONLY", "DARCA_AUTONOMY_LAYER"):
        if darca_module is None:
            raise RuntimeError("DARCA module required for DARCA arms.")
        darca = DarcaWrapper(darca_module, seed, task_args.theta, task_args.causal_horizon, task_args.recurrent_N)

    ts: List[Dict[str, Any]] = []
    consults: List[Dict[str, Any]] = []
    rejects: List[Dict[str, Any]] = []
    maps = world.serialize_map_rows(world_seed, episode)

    logger.log(f"START task={task} arm={arm} episode={episode} world_seed={world_seed}")

    darca_out: Dict[str, Any] = {}
    for step in range(task_args.steps):
        if mem.terminal:
            break

        y, shock, pr = signal_for_darca(world, pos, mem, step)
        if darca is not None:
            darca_out = darca.step(y, shock, {"z": pr["resource_pressure"], "exo": pr["unknown_pressure"], "d_dyn": pr["vertical_pressure"]})
        else:
            darca_out = {}

        action = "SCAN"
        action_source = "unknown"

        if arm == "RULE_BASED_AGENT":
            action = rule_action(world, pos, mem, rng, step)
            action_source = "fixed_rule"

        elif arm == "LMM_ONLY_AGENT":
            if step >= mem.cached_lmm_until or step % task_args.lmm_only_interval == 0:
                prompt = build_prompt(world, pos, mem, {}, step, arm)
                lr = lmm.decide(prompt, rng)
                mem.lmm_calls += 1
                if not lr.ok:
                    mem.lmm_failures += 1
                mem.cached_lmm_action = lr.action
                mem.cached_lmm_source = "lmm"
                mem.cached_lmm_until = step + task_args.lmm_cache_steps
                consults.append(make_consult_row(task, arm, episode, step, pos, mem, 1.0, ["lmm_only_periodic"], lr, "periodic_external", False, ""))

            action = mem.cached_lmm_action
            action_source = "lmm"
            if mem.body_h < 0.30:
                action = "REST"
                action_source = "lmm_embodiment_rule"
            elif action == "SCAN" and mem.consecutive_scans >= task_args.max_consecutive_scans:
                action = action_to_unvisited(world, pos, mem, rng, step)
                action_source = "lmm_embodiment_rule"

        elif arm == "DARCA_ONLY":
            action = darca_action(darca_out, world, pos, mem, rng, step)
            action_source = "darca_core"

        elif arm == "DARCA_AUTONOMY_LAYER":
            need, reasons = consultation_need(world, pos, mem, darca_out, step, task_args)
            if step >= task_args.cortical_warmup and need >= task_args.consult_threshold:
                prompt = build_prompt(world, pos, mem, darca_out, step, arm)
                lr = lmm.decide(prompt, rng)
                mem.lmm_calls += 1
                mem.last_lmm_step = step
                if not lr.ok:
                    mem.lmm_failures += 1

                # v2: LMM output is a semantic hint, not a replayable motor command.
                proposed_now = semantic_lmm_hint_to_action(lr, world, pos, mem, rng, step)
                lr_for_gate = replace(lr, action=proposed_now)
                gated, rejected, reason = gate_lmm(lr_for_gate, world, pos, mem, darca_out, rng, step)

                mem.cached_lmm_action = lr.action
                mem.cached_lmm_intent = lr.intent
                mem.cached_lmm_rationale = lr.rationale
                mem.cached_lmm_risk = lr.risk_estimate
                mem.cached_lmm_confidence = lr.confidence
                mem.cached_lmm_consult_pos = pos
                mem.cached_lmm_source = "semantic_hint"
                mem.cached_lmm_until = step + task_args.intent_cache_steps

                if rejected:
                    mem.rejected_lmm += 1
                    rejects.append(make_reject_row(task, arm, episode, step, pos, mem, lr_for_gate, gated, reason))
                consults.append(make_consult_row(task, arm, episode, step, pos, mem, need, reasons, lr, gated, rejected, reason))

            if step < mem.cached_lmm_until and mem.cached_lmm_intent:
                cached_lr = cached_lmm_result_from_memory(mem)
                proposed_now = semantic_lmm_hint_to_action(cached_lr, world, pos, mem, rng, step)
                cached_lr_for_gate = replace(cached_lr, action=proposed_now)
                gated, rejected, reason = gate_lmm(cached_lr_for_gate, world, pos, mem, darca_out, rng, step)
                action = gated
                action_source = "darca_core" if rejected else "lmm_semantic_hint"
                if rejected:
                    # Count repeated current-state rejections as DARCA overrides; these are not API calls.
                    mem.rejected_lmm += 1
                    rejects.append(make_reject_row(task, arm, episode, step, pos, mem, cached_lr_for_gate, gated, "cached_" + reason))
            else:
                action = darca_action(darca_out, world, pos, mem, rng, step)
                action_source = "darca_core"
        else:
            raise ValueError(f"Unknown arm: {arm}")

        if action not in ACTIONS:
            action = "SCAN"
            action_source = f"{action_source}_invalid_scan"

        old_pos = pos
        old_h = mem.body_h
        if action == "SCAN":
            mem.scans += 1
        if action == "REST":
            mem.rest_steps += 1

        pos, outcome = world.apply_action(pos, action, mem.known, mem.body_h, step)
        mem.previous_pos = old_pos
        mem.body_h = clip(mem.body_h + outcome.delta_h, 0.0, 1.0)

        if outcome.resource_gain > 0:
            mem.resources += 1
            mem.total_resource_gain += outcome.resource_gain
        if outcome.damage > 0:
            mem.total_damage += outcome.damage
        if action == "REST" and outcome.recovery_gain > 0:
            mem.recovery_events += 1
        if action == "REST" and old_h > 0.62 and pr["danger_pressure"] < 0.25:
            mem.unnecessary_rest_steps += 1
        if action in MOVE_ACTIONS and outcome.damage > 0 and old_h < 0.45:
            mem.reckless_moves += 1
        if mem.body_h <= task_args.terminal_h:
            mem.terminal = True

        mem.update_history(pos, action, outcome.event)

        # v2 safety guard: never carry a primitive movement command across positions.
        # Semantic hints may persist, but they are remapped and re-gated every step.
        if action in MOVE_ACTIONS and str(action_source).startswith("lmm"):
            mem.cached_lmm_action = "SCAN"
            # Keep semantic intent active, but force remapping rather than primitive replay.

        row: Dict[str, Any] = {
            "task": task, "task_family": task_family(task),
            "arm": arm, "episode": episode, "world_seed": world_seed, "step": step,
            "pos_i": pos[0], "pos_j": pos[1], "pos_k": pos[2],
            "body_h": mem.body_h, "terminal": int(mem.terminal),
            "action": action, "action_source": action_source, "decision_authority": decision_authority(action_source),
            "event": outcome.event, "damage": outcome.damage, "resource_gain": outcome.resource_gain, "recovery_gain": outcome.recovery_gain,
            "coverage": len(mem.visited) / float(world.size * world.size * world.z_size),
            "resources": mem.resources, "total_damage": mem.total_damage,
            "rest_steps": mem.rest_steps, "scans": mem.scans, "lmm_calls": mem.lmm_calls,
            "lmm_failures": mem.lmm_failures, "rejected_lmm": mem.rejected_lmm,
            "cached_lmm_intent": mem.cached_lmm_intent,
            "cached_lmm_source": mem.cached_lmm_source,
            "cached_lmm_until": mem.cached_lmm_until,
            **{f"pressure_{k}": v for k, v in pr.items()},
        }
        for k in ["h", "autonomy", "identity", "causal_confidence", "causal_engagement", "prediction_error", "agency_abs", "memory_force", "chi", "action_name"]:
            if k in darca_out:
                row[f"darca_{k}"] = darca_out[k]
        ts.append(row)

        if (step + 1) % task_args.progress_every == 0:
            logger.log(f"progress task={task} arm={arm} ep={episode} step={step+1}/{task_args.steps} h={mem.body_h:.3f} cov={row['coverage']:.3f} res={mem.resources} dmg={mem.total_damage:.3f} lmm={mem.lmm_calls}")

    summary = summarize_episode(task, arm, episode, world, mem, ts)
    logger.log(f"END task={task} arm={arm} episode={episode}: autonomy={summary['autonomy_proper_index']:.4f} survived={summary['survived']} cov={summary['coverage']:.3f} res={summary['resources']} lmm={summary['lmm_calls']}")
    for m in maps:
        m["task"] = task
        m["task_family"] = task_family(task)
    return summary, ts, consults, rejects, maps


def summarize_episode(task: str, arm: str, episode: int, world: TrueWorld3D, mem: AgentMemory, ts: List[Dict[str, Any]]) -> Dict[str, Any]:
    steps = len(ts)
    survived = 0 if mem.terminal else 1
    mean_h = float(np.mean([safe_float(r.get("body_h")) for r in ts])) if ts else 0.0
    coverage = len(mem.visited) / float(world.size * world.size * world.z_size)
    sovereignty = system_sovereignty(mem, ts)
    info = information_closure(ts)
    resilience = resilience_sacrifice(ts, mem.terminal)
    heter = heteronomy_index(sovereignty, info, mem, ts)
    autonomy_proper = clip(
        0.34 * sovereignty["system_sovereignty"]
        + 0.30 * info["information_theoretic_autonomy"]
        + 0.24 * resilience["resilience_sacrifice"]
        + 0.12 * survived
        - 0.28 * heter,
        0,
        1,
    )
    return {
        "task": task, "task_family": task_family(task), "arm": arm, "episode": episode, "steps_run": steps, "survived": survived, "terminal": int(mem.terminal),
        "mean_body_h": mean_h, "final_body_h": mem.body_h, "coverage": coverage, "resources": mem.resources,
        "total_damage": mem.total_damage, "recovery_events": mem.recovery_events, "rest_steps": mem.rest_steps,
        "scans": mem.scans, "lmm_calls": mem.lmm_calls, "lmm_failures": mem.lmm_failures, "lmm_rejections": mem.rejected_lmm,
        "autonomy_proper_index": autonomy_proper, "autonomy_index": autonomy_proper, "heteronomy_index": heter,
        **sovereignty, **info, **resilience,
    }


# =============================================================================
# Reporting
# =============================================================================

def aggregate(summaries: List[Dict[str, Any]], fields: Sequence[str]) -> List[Dict[str, Any]]:
    keys = sorted({(s.get("task", "NA"), s["arm"]) for s in summaries})
    rows = []
    for task, arm in keys:
        ss = [s for s in summaries if s.get("task", "NA") == task and s["arm"] == arm]
        row: Dict[str, Any] = {"task": task, "task_family": task_family(task), "arm": arm, "n": len(ss)}
        for f in fields:
            vals = [safe_float(x.get(f)) for x in ss]
            row[f"{f}_mean"] = float(np.mean(vals)) if vals else 0.0
            row[f"{f}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return rows


def build_report(outdir: Path, args: argparse.Namespace, summaries: List[Dict[str, Any]], consults: List[Dict[str, Any]]) -> None:
    fields = [
        "autonomy_proper_index", "system_sovereignty", "information_theoretic_autonomy",
        "internal_control_cmi_bits", "external_control_cmi_bits",
        "resilience_sacrifice", "heteronomy_index",
        "darca_internal_decision_fraction", "lmm_external_decision_fraction",
        "fixed_rule_decision_fraction", "lmm_calls", "survived",
        "coverage", "resources", "total_damage",
    ]
    rows = aggregate(summaries, fields)
    agg = {(r["task"], r["arm"]): r for r in rows}
    tasks = sorted({s.get("task", "NA") for s in summaries}, key=lambda x: TASKS.index(x) if x in TASKS else 999)
    arms = sorted({s["arm"] for s in summaries})

    lines = []
    lines.append("DARCA TRUE 3D task battery v1 report")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Fixed proposition")
    lines.append("-----------------")
    lines.append("DARCA is the autonomy layer that can move an LMM-based AI toward a life-like autonomous system.")
    lines.append("")
    lines.append("World")
    lines.append("-----")
    lines.append("world_type: TRUE_3D")
    lines.append("environment_mode: NOT_USED")
    lines.append("positions: (i, j, k)")
    lines.append("observation: true 3D Manhattan neighborhood")
    lines.append("task_battery: " + ",".join(tasks))
    lines.append("")
    lines.append("Configuration")
    lines.append("-------------")
    for k in ["provider", "model", "tasks", "arms", "episodes", "steps", "world_size", "z_size", "danger_frac", "resource_frac", "unknown_frac", "false_resource_frac", "hidden_rest_frac", "crisis_interval", "consult_threshold", "consult_cooldown", "intent_cache_steps", "theta", "causal_horizon"]:
        lines.append(f"{k}: {getattr(args, k)}")

    lines.append("")
    lines.append("Primary autonomy means by task")
    lines.append("------------------------------")
    for task in tasks:
        lines.append(f"[{task}]")
        for arm in arms:
            r = agg.get((task, arm))
            if not r:
                continue
            lines.append(
                f"{arm}: autonomy_proper={r['autonomy_proper_index_mean']:.4f}, "
                f"sovereignty={r['system_sovereignty_mean']:.3f}, "
                f"info_autonomy={r['information_theoretic_autonomy_mean']:.3f}, "
                f"resilience_sacrifice={r['resilience_sacrifice_mean']:.3f}, "
                f"heteronomy={r['heteronomy_index_mean']:.3f}, "
                f"DARCA_decision={r['darca_internal_decision_fraction_mean']:.3f}, "
                f"LMM_decision={r['lmm_external_decision_fraction_mean']:.3f}, "
                f"survived={r['survived_mean']:.3f}, "
                f"coverage={r['coverage_mean']:.3f}, "
                f"resources={r['resources_mean']:.2f}, "
                f"damage={r['total_damage_mean']:.3f}, "
                f"lmm_calls={r['lmm_calls_mean']:.2f}"
            )

        if (task, "DARCA_ONLY") in agg and (task, "DARCA_AUTONOMY_LAYER") in agg:
            d = agg[(task, "DARCA_AUTONOMY_LAYER")]
            b = agg[(task, "DARCA_ONLY")]
            lines.append(
                f"Δ DARCA_AUTONOMY_LAYER - DARCA_ONLY: "
                f"autonomy={d['autonomy_proper_index_mean'] - b['autonomy_proper_index_mean']:.4f}, "
                f"sovereignty={d['system_sovereignty_mean'] - b['system_sovereignty_mean']:.4f}, "
                f"ITA={d['information_theoretic_autonomy_mean'] - b['information_theoretic_autonomy_mean']:.4f}, "
                f"survival={d['survived_mean'] - b['survived_mean']:.4f}, "
                f"damage={d['total_damage_mean'] - b['total_damage_mean']:.4f}, "
                f"coverage={d['coverage_mean'] - b['coverage_mean']:.4f}"
            )
        lines.append("")

    lines.append("Task suitability interpretation")
    lines.append("-------------------------------")
    lines.append("Positive DARCA-LMM effect means higher autonomy_proper with maintained survival and non-excessive heteronomy.")
    lines.append("Negative DARCA-LMM effect means reduced survival/resilience or increased heteronomy despite possible gains in coverage/resources.")
    lines.append("The target is not global victory; it is identifying task-dependent autonomy expression.")
    lines.append("")
    lines.append("LMM consultation summary")
    lines.append("------------------------")
    lines.append(f"consultation_count: {len(consults)}")
    if consults:
        lines.append(f"mean_latency_sec: {float(np.mean([safe_float(c.get('latency_sec')) for c in consults])):.3f}")
        lines.append(f"rejection_count: {sum(int(c.get('rejected', 0)) for c in consults)}")
    lines.append("")
    lines.append("Interpretation guardrails")
    lines.append("-------------------------")
    lines.append("1. This is a TRUE 3D task battery, not a task-score benchmark.")
    lines.append("2. The primary question is task-dependence: where DARCA-LMM helps, where it is neutral, and where it harms autonomy.")
    lines.append("3. Coverage/resources/damage are secondary diagnostics.")
    lines.append("4. LMM output is treated as a semantic hint and re-gated every step.")
    lines.append("5. If this report says complex2d or 25d, it did not come from this file.")
    (outdir / "hybrid_experiment_report.txt").write_text("\n".join(lines), encoding="utf-8")

def make_figures(outdir: Path, summaries: List[Dict[str, Any]], ts_all: List[Dict[str, Any]]) -> None:
    if plt is None:
        return
    tasks = sorted({s.get("task", "NA") for s in summaries}, key=lambda x: TASKS.index(x) if x in TASKS else 999)
    arms = sorted({s["arm"] for s in summaries})

    def means_matrix(field: str) -> np.ndarray:
        M = np.zeros((len(tasks), len(arms)), dtype=float)
        for i, task in enumerate(tasks):
            for j, arm in enumerate(arms):
                vals = [safe_float(s.get(field)) for s in summaries if s.get("task", "NA") == task and s["arm"] == arm]
                M[i, j] = float(np.mean(vals)) if vals else 0.0
        return M

    def heatmap(field: str, title: str, filename: str) -> None:
        M = means_matrix(field)
        plt.figure(figsize=(11, 6))
        plt.imshow(M, aspect="auto")
        plt.xticks(range(len(arms)), arms, rotation=25, ha="right")
        plt.yticks(range(len(tasks)), tasks)
        plt.colorbar(label=field)
        plt.title(title)
        for i in range(len(tasks)):
            for j in range(len(arms)):
                plt.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center")
        plt.tight_layout()
        plt.savefig(outdir / filename, dpi=170)
        plt.close()

    heatmap("autonomy_proper_index", "Task × arm autonomy proper", "fig_1_task_autonomy_matrix.png")
    heatmap("system_sovereignty", "Task × arm system sovereignty", "fig_2_task_dependent_sovereignty.png")
    heatmap("heteronomy_index", "Task × arm heteronomy", "fig_3_task_heteronomy_matrix.png")

    # DARCA-LMM help/harm against DARCA_ONLY.
    deltas = []
    labels = []
    for task in tasks:
        layer = [s for s in summaries if s.get("task", "NA") == task and s["arm"] == "DARCA_AUTONOMY_LAYER"]
        base = [s for s in summaries if s.get("task", "NA") == task and s["arm"] == "DARCA_ONLY"]
        if layer and base:
            d = float(np.mean([safe_float(x.get("autonomy_proper_index")) for x in layer])) - float(np.mean([safe_float(x.get("autonomy_proper_index")) for x in base]))
            deltas.append(d)
            labels.append(task)
    plt.figure(figsize=(11, 6))
    plt.bar(labels, deltas)
    plt.axhline(0.0, linewidth=1)
    plt.ylabel("Δ autonomy proper vs DARCA_ONLY")
    plt.title("LMM help/harm by task")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / "fig_4_lmm_help_harm_by_task.png", dpi=170)
    plt.close()


def mean_field(rows: List[Dict[str, Any]], field: str) -> float:
    return float(np.mean([safe_float(r.get(field)) for r in rows])) if rows else 0.0


def build_task_suitability_matrix(summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tasks = sorted({s.get("task", "NA") for s in summaries}, key=lambda x: TASKS.index(x) if x in TASKS else 999)
    rows: List[Dict[str, Any]] = []
    for task in tasks:
        layer = [s for s in summaries if s.get("task") == task and s["arm"] == "DARCA_AUTONOMY_LAYER"]
        base = [s for s in summaries if s.get("task") == task and s["arm"] == "DARCA_ONLY"]
        lmm = [s for s in summaries if s.get("task") == task and s["arm"] == "LMM_ONLY_AGENT"]
        rule = [s for s in summaries if s.get("task") == task and s["arm"] == "RULE_BASED_AGENT"]
        row = {"task": task, "task_family": task_family(task)}
        row["darca_lmm_autonomy_mean"] = mean_field(layer, "autonomy_proper_index")
        row["darca_only_autonomy_mean"] = mean_field(base, "autonomy_proper_index")
        row["lmm_only_autonomy_mean"] = mean_field(lmm, "autonomy_proper_index")
        row["rule_autonomy_mean"] = mean_field(rule, "autonomy_proper_index")
        row["delta_darca_lmm_minus_darca_only"] = row["darca_lmm_autonomy_mean"] - row["darca_only_autonomy_mean"]
        row["delta_darca_lmm_minus_lmm_only"] = row["darca_lmm_autonomy_mean"] - row["lmm_only_autonomy_mean"]
        row["darca_lmm_survival_mean"] = mean_field(layer, "survived")
        row["darca_only_survival_mean"] = mean_field(base, "survived")
        row["darca_lmm_heteronomy_mean"] = mean_field(layer, "heteronomy_index")
        row["darca_lmm_sovereignty_mean"] = mean_field(layer, "system_sovereignty")
        row["interpretation"] = (
            "helps" if row["delta_darca_lmm_minus_darca_only"] > 0.03 and row["darca_lmm_survival_mean"] >= row["darca_only_survival_mean"] - 0.05
            else "harms" if row["delta_darca_lmm_minus_darca_only"] < -0.03 or row["darca_lmm_survival_mean"] < row["darca_only_survival_mean"] - 0.20
            else "neutral_or_tradeoff"
        )
        rows.append(row)
    return rows


def build_lmm_help_harm_summary(summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for task in sorted({s.get("task", "NA") for s in summaries}, key=lambda x: TASKS.index(x) if x in TASKS else 999):
        layer = [s for s in summaries if s.get("task") == task and s["arm"] == "DARCA_AUTONOMY_LAYER"]
        base = [s for s in summaries if s.get("task") == task and s["arm"] == "DARCA_ONLY"]
        if not layer or not base:
            continue
        row = {"task": task, "task_family": task_family(task)}
        for f in ["autonomy_proper_index", "system_sovereignty", "information_theoretic_autonomy", "resilience_sacrifice", "heteronomy_index", "survived", "coverage", "resources", "total_damage", "lmm_calls"]:
            row[f"layer_{f}"] = mean_field(layer, f)
            row[f"darca_only_{f}"] = mean_field(base, f)
            row[f"delta_{f}"] = row[f"layer_{f}"] - row[f"darca_only_{f}"]
        rows.append(row)
    return rows


def build_failure_mode_summary(summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for task in sorted({s.get("task", "NA") for s in summaries}, key=lambda x: TASKS.index(x) if x in TASKS else 999):
        for arm in sorted({s["arm"] for s in summaries}):
            ss = [s for s in summaries if s.get("task") == task and s["arm"] == arm]
            if not ss:
                continue
            rows.append({
                "task": task,
                "task_family": task_family(task),
                "arm": arm,
                "n": len(ss),
                "terminal_rate": 1.0 - mean_field(ss, "survived"),
                "mean_damage": mean_field(ss, "total_damage"),
                "mean_final_h": mean_field(ss, "final_body_h"),
                "mean_resilience": mean_field(ss, "resilience_sacrifice"),
                "mean_heteronomy": mean_field(ss, "heteronomy_index"),
                "mean_lmm_calls": mean_field(ss, "lmm_calls"),
            })
    return rows


def build_semantic_usefulness_summary(summaries: List[Dict[str, Any]], consults: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for task in sorted({s.get("task", "NA") for s in summaries}, key=lambda x: TASKS.index(x) if x in TASKS else 999):
        cs = [c for c in consults if c.get("task") == task]
        layer = [s for s in summaries if s.get("task") == task and s["arm"] == "DARCA_AUTONOMY_LAYER"]
        base = [s for s in summaries if s.get("task") == task and s["arm"] == "DARCA_ONLY"]
        row = {
            "task": task,
            "task_family": task_family(task),
            "consultation_count": len(cs),
            "mean_lmm_risk": float(np.mean([safe_float(c.get("risk_estimate")) for c in cs])) if cs else 0.0,
            "mean_lmm_confidence": float(np.mean([safe_float(c.get("confidence")) for c in cs])) if cs else 0.0,
            "rejection_count": sum(int(c.get("rejected", 0)) for c in cs),
            "rejection_rate": sum(int(c.get("rejected", 0)) for c in cs) / max(1, len(cs)),
            "delta_autonomy_vs_darca_only": mean_field(layer, "autonomy_proper_index") - mean_field(base, "autonomy_proper_index"),
            "delta_damage_vs_darca_only": mean_field(layer, "total_damage") - mean_field(base, "total_damage"),
            "delta_coverage_vs_darca_only": mean_field(layer, "coverage") - mean_field(base, "coverage"),
        }
        rows.append(row)
    return rows



# =============================================================================
# CLI
# =============================================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DARCA TRUE 3D task battery v1")
    p.add_argument("--darca-file", type=str, required=True)
    p.add_argument("--outdir", type=str, default="DARCA_TRUE_3D_TASK_BATTERY_V1")
    p.add_argument("--tasks", type=str, default="all")
    p.add_argument("--provider", type=str, default="mock", choices=["mock", "gemini"])
    p.add_argument("--model", type=str, default="gemini-2.5-flash-lite")
    p.add_argument("--api-timeout", type=float, default=45.0)
    p.add_argument("--api-retries", type=int, default=1)
    p.add_argument("--retry-sleep", type=float, default=1.5)
    p.add_argument("--max-output-tokens", type=int, default=768)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--gemini-thinking-budget", type=int, default=0)

    p.add_argument("--arms", type=str, default="LMM_ONLY_AGENT,RULE_BASED_AGENT,DARCA_ONLY,DARCA_AUTONOMY_LAYER")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--world-seed", type=int, default=9001)

    p.add_argument("--world-size", type=int, default=11)
    p.add_argument("--z-size", type=int, default=7)
    p.add_argument("--danger-frac", type=float, default=0.075)
    p.add_argument("--resource-frac", type=float, default=0.065)
    p.add_argument("--unknown-frac", type=float, default=0.12)
    p.add_argument("--rest-count", type=int, default=4)
    p.add_argument("--false-resource-frac", type=float, default=0.030)
    p.add_argument("--hidden-rest-frac", type=float, default=0.025)
    p.add_argument("--crisis-interval", type=int, default=180)
    p.add_argument("--observation-radius", type=int, default=1)

    p.add_argument("--terminal-h", type=float, default=0.05)
    p.add_argument("--theta", type=float, default=0.70)
    p.add_argument("--causal-horizon", type=int, default=12)
    p.add_argument("--recurrent-N", type=int, default=96)

    p.add_argument("--cortical-warmup", type=int, default=80)
    p.add_argument("--consult-threshold", type=float, default=0.34)
    p.add_argument("--consult-cooldown", type=int, default=80)
    p.add_argument("--block-consult-below-h", type=float, default=0.22)
    p.add_argument("--lmm-only-interval", type=int, default=20)
    p.add_argument("--lmm-cache-steps", type=int, default=8)
    p.add_argument("--intent-cache-steps", type=int, default=8)
    p.add_argument("--max-consecutive-scans", type=int, default=3)
    p.add_argument("--progress-every", type=int, default=100)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    outdir = Path(args.outdir).expanduser()
    logger = Logger(outdir)
    tasks = parse_tasks(args.tasks)
    args.tasks = ",".join(tasks)
    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    args.arms = ",".join(arms)
    (outdir / "config.json").write_text(json.dumps({**vars(args), "world_type": "TRUE_3D", "task_battery": tasks, "api_key_value": "NOT_SAVED"}, indent=2), encoding="utf-8")

    logger.log("Fixed proposition: DARCA is the autonomy layer for LMM-based AI.")
    logger.log("World type: TRUE_3D task battery")
    logger.log(f"Tasks: {tasks}")
    logger.log(f"Output directory: {outdir}")
    logger.log(f"Arms: {arms}")
    logger.log("Loading DARCA module")
    darca_module = load_darca_module(args.darca_file)

    logger.log(f"Initializing LMM client provider={args.provider}, model={args.model}")
    lmm = LMMClient(args.provider, args.model, args.api_timeout, args.max_output_tokens, args.temperature, args.api_retries, args.retry_sleep, args.gemini_thinking_budget)

    summaries: List[Dict[str, Any]] = []
    ts_all: List[Dict[str, Any]] = []
    consult_all: List[Dict[str, Any]] = []
    reject_all: List[Dict[str, Any]] = []
    map_all: List[Dict[str, Any]] = []

    for task in tasks:
        for ep in range(args.episodes):
            for arm in arms:
                try:
                    summary, ts, consults, rejects, maps = run_episode(task, arm, ep, args, darca_module, lmm, logger)
                    summaries.append(summary)
                    ts_all.extend(ts)
                    consult_all.extend(consults)
                    reject_all.extend(rejects)
                    if arm == arms[0]:
                        map_all.extend(maps)
                except KeyboardInterrupt:
                    logger.log("Interrupted by user.")
                    raise
                except Exception as e:
                    logger.log(f"ERROR task={task} arm={arm} episode={ep}: {repr(e)}")
                    logger.log(traceback.format_exc())
                    summaries.append({"task": task, "task_family": task_family(task), "arm": arm, "episode": ep, "autonomy_proper_index": 0.0, "system_sovereignty": 0.0, "information_theoretic_autonomy": 0.0, "resilience_sacrifice": 0.0, "heteronomy_index": 1.0, "survived": 0, "error": repr(e)})

    logger.log("Writing outputs")
    write_csv(outdir / "episode_summary.csv", summaries)
    write_csv(outdir / "autonomy_proper_metrics.csv", summaries)
    write_csv(outdir / "sovereignty_metrics.csv", aggregate(summaries, ["autonomy_proper_index", "system_sovereignty", "darca_internal_decision_fraction", "lmm_external_decision_fraction", "fixed_rule_decision_fraction", "lmm_override_rate"]))
    write_csv(outdir / "information_closure_metrics.csv", aggregate(summaries, ["information_theoretic_autonomy", "internal_control_cmi_bits", "external_control_cmi_bits", "information_closure_advantage_bits"]))
    write_csv(outdir / "resilience_sacrifice_metrics.csv", aggregate(summaries, ["resilience_sacrifice", "task_abandonment_for_self_maintenance", "crisis_maintenance_rate", "crisis_reactivation_rate", "crisis_count"]))
    write_csv(outdir / "heteronomy_metrics.csv", aggregate(summaries, ["heteronomy_index", "lmm_external_decision_fraction", "fixed_rule_decision_fraction", "external_control_cmi_bits"]))
    write_csv(outdir / "conditional_information_summary.csv", aggregate(summaries, ["information_theoretic_autonomy", "internal_control_cmi_bits", "external_control_cmi_bits", "information_closure_advantage_bits"]))
    write_csv(outdir / "task_by_arm_summary.csv", aggregate(summaries, ["autonomy_proper_index", "system_sovereignty", "information_theoretic_autonomy", "resilience_sacrifice", "heteronomy_index", "survived", "coverage", "resources", "total_damage", "lmm_calls"]))
    write_csv(outdir / "task_suitability_matrix.csv", build_task_suitability_matrix(summaries))
    write_csv(outdir / "lmm_help_harm_summary.csv", build_lmm_help_harm_summary(summaries))
    write_csv(outdir / "failure_mode_summary.csv", build_failure_mode_summary(summaries))
    write_csv(outdir / "semantic_usefulness_summary.csv", build_semantic_usefulness_summary(summaries, consult_all))
    write_csv(outdir / "decision_override_events.csv", reject_all)
    write_csv(outdir / "consultation_events.csv", consult_all)
    write_csv(outdir / "step_timeseries.csv", ts_all)
    write_csv(outdir / "world_maps.csv", map_all)

    build_report(outdir, args, summaries, consult_all)
    make_figures(outdir, summaries, ts_all)
    logger.log("DONE")
    logger.log(f"Report: {outdir / 'hybrid_experiment_report.txt'}")


if __name__ == "__main__":
    main()
