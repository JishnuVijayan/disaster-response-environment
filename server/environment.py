"""
DisasterResponseEnvironment — Core Environment Implementation

Emergency Operations Center triage agent for OpenEnv.
Simulates real-world multi-zone disaster response with:
  - Dynamic zone stress trajectories
  - Finite, lockable resources (rescue / medical / evacuation credits)
  - Probabilistic alert generation with hidden ground-truth
  - Adversarial misinformation (spoofed alerts) in Task 3
  - Oracle-normalized scoring across 3 difficulty levels
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import ActionType, DisasterAction, DisasterObservation, DisasterState
    from .config import DEFAULT_TASK, TASK_CONFIGS
    from .messages import _NOISE_MSGS, _REAL_MSGS, _SPOOF_MSGS
    from .oracle import _oracle_create_alert, oracle_decide
except ImportError:
    from models import ActionType, DisasterAction, DisasterObservation, DisasterState
    from server.config import DEFAULT_TASK, TASK_CONFIGS
    from server.messages import _NOISE_MSGS, _REAL_MSGS, _SPOOF_MSGS
    from server.oracle import _oracle_create_alert, oracle_decide


class DisasterResponseEnvironment(Environment):
    """
    Emergency Operations Center Triage Agent environment.

    Implements the full OpenEnv spec:  reset() / step() / state property.
    Three tasks with escalating difficulty — easy / medium / hard.

    State is maintained as instance variables. The environment uses a
    singleton pattern in app.py so HTTP stateless requests share state.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # Class-level grader store — shared across all instances, only updated on episode end
    _last_grader_data: Dict[str, Any] = {
        "task": None,
        "score": None,
        "agent_reward": None,
        "oracle_reward": None,
        "steps_completed": 0,
    }

    def __init__(self) -> None:
        super().__init__()
        self._task_name: str = DEFAULT_TASK
        self._task_cfg: Dict[str, Any] = TASK_CONFIGS[DEFAULT_TASK]
        self._rng: random.Random = random.Random(42)
        self._step: int = 0
        self._done: bool = True
        self._zones: List[Dict[str, Any]] = [
            {"id": z["id"], "name": z["name"], "stress": z["initial_stress"]}
            for z in self._task_cfg["zones"]
        ]
        self._resources: Dict[str, Any] = {
            "rescue_teams_available": self._task_cfg["initial_rescue_teams"],
            "rescue_teams_locked": [],
            "medical_units_available": self._task_cfg["initial_medical_units"],
            "medical_units_locked": [],
            "broadcast_credits": self._task_cfg["broadcast_credits"],
        }
        self._alert_queue: List[Dict[str, Any]] = []
        self._current_alert: Optional[Dict[str, Any]] = None
        self._cumulative_reward: float = 0.0
        self._oracle_reward: float = 0.0
        self._normalized_score: Optional[float] = None
        self._last_action_result: Optional[str] = None
        # Episode metrics
        self._total_alerts: int = 0
        self._correct_dispatches: int = 0
        self._false_alarm_dispatches: int = 0
        self._correct_dismissals: int = 0
        self._missed_real_alerts: int = 0
        self._cascade_events: int = 0

    # ------------------------------------------------------------------
    # Public OpenEnv API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_name: Optional[str] = None,
        **kwargs: Any,
    ) -> DisasterObservation:
        """
        Reset the environment for a new episode.

        Args:
            seed: Random seed for reproducibility.
            episode_id: If a valid task name, selects that task (convenience API).
            task_name: Explicit task selector — one of:
                       task1_flood_easy | task2_multizone_medium | task3_compound_hard
        """
        resolved = task_name or episode_id or DEFAULT_TASK
        if resolved not in TASK_CONFIGS:
            resolved = DEFAULT_TASK
        self._task_name = resolved
        self._task_cfg = TASK_CONFIGS[resolved]

        eff_seed = seed if seed is not None else random.randint(0, 2 ** 31 - 1)
        self._rng = random.Random(eff_seed)
        self._episode_seed = eff_seed

        self._step = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._oracle_reward = 0.0
        self._normalized_score = None
        self._last_action_result = None
        self._total_alerts = 0
        self._correct_dispatches = 0
        self._false_alarm_dispatches = 0
        self._correct_dismissals = 0
        self._missed_real_alerts = 0
        self._cascade_events = 0

        self._zones = [
            {"id": z["id"], "name": z["name"], "stress": z["initial_stress"]}
            for z in self._task_cfg["zones"]
        ]
        self._resources = {
            "rescue_teams_available": self._task_cfg["initial_rescue_teams"],
            "rescue_teams_locked": [],
            "medical_units_available": self._task_cfg["initial_medical_units"],
            "medical_units_locked": [],
            "broadcast_credits": self._task_cfg["broadcast_credits"],
        }

        self._alert_queue = []
        self._alert_queue.extend(self._generate_alerts())
        if not self._alert_queue:
            self._alert_queue.append(self._create_alert(self._highest_stress_zone()))

        self._current_alert = self._alert_queue.pop(0)
        self._step = 1

        return self._build_observation(reward=0.0, done=False)

    def step(
        self,
        action: DisasterAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> DisasterObservation:
        """Execute one triage decision on the current active alert."""
        if self._done:
            return self._build_observation(reward=0.0, done=True)

        if self._current_alert is None:
            self._advance_alert()
            if self._current_alert is None:
                self._done = True
                return self._build_observation(reward=0.0, done=True)

        # Penalise wrong alert target
        if (
            action.alert_id is not None
            and action.alert_id != self._current_alert["alert_id"]
        ):
            reward = -0.20
            self._cumulative_reward += reward
            self._total_alerts += 1
            self._advance_alert()
            self._step += 1
            done = self._step > self._task_cfg["max_steps"]
            self._done = done
            if done:
                self._finalise_episode()
            self._last_action_result = "Wrong alert target — id mismatch"
            return self._build_observation(reward=reward, done=done)

        alert = self._current_alert
        zone = self._get_zone(alert["zone_id"]) or {
            "id": alert["zone_id"], "name": alert["zone_name"], "stress": 0.5
        }

        reward, result = self._calculate_reward(action, alert, zone)
        self._cumulative_reward += reward

        self._apply_resource_change(action)
        self._update_zone_stresses(action, alert["zone_id"])
        self._apply_cascade()
        self._unlock_resources()

        new_alerts = self._generate_alerts()
        self._alert_queue.extend(new_alerts)

        # Track metrics
        self._total_alerts += 1
        at = action.action_type
        is_real = alert["is_real"]
        if at in (ActionType.DISPATCH_RESCUE, ActionType.DISPATCH_MEDICAL) and is_real:
            self._correct_dispatches += 1
        elif at in (ActionType.DISPATCH_RESCUE, ActionType.DISPATCH_MEDICAL) and not is_real:
            self._false_alarm_dispatches += 1
        elif at == ActionType.DISMISS_FALSE_ALARM and not is_real:
            self._correct_dismissals += 1
        elif at == ActionType.DISMISS_FALSE_ALARM and is_real:
            self._missed_real_alerts += 1

        if at == ActionType.REQUEST_MORE_INFO:
            self._current_alert["deliberation_count"] += 1
        else:
            self._advance_alert()

        self._step += 1
        done = self._step > self._task_cfg["max_steps"]
        self._done = done
        if done:
            self._finalise_episode()

        self._last_action_result = result
        return self._build_observation(reward=reward, done=done)

    @property
    def state(self) -> DisasterState:
        """Return full internal state (for GET /state)."""
        return DisasterState(
            episode_id=None,
            step_count=self._step,
            task_name=self._task_name,
            task_description=self._task_cfg["description"],
            zones=self._serialise_zones(),
            resources=self._serialise_resources(),
            current_alert=self._public_alert(self._current_alert),
            alert_queue_size=len(self._alert_queue),
            cumulative_reward=round(self._cumulative_reward, 4),
            oracle_reward=round(self._oracle_reward, 4),
            normalized_score=self._normalized_score,
            total_alerts_processed=self._total_alerts,
            correct_dispatches=self._correct_dispatches,
            false_alarm_dispatches=self._false_alarm_dispatches,
            correct_dismissals=self._correct_dismissals,
            missed_real_alerts=self._missed_real_alerts,
            cascade_events=self._cascade_events,
        )

    def close(self) -> None:
        """No-op — state persists in the singleton across HTTP requests."""
        pass

    def get_metadata(self):
        from openenv.core.env_server.types import EnvironmentMetadata
        return EnvironmentMetadata(
            name="disaster_response",
            description=(
                "Emergency Operations Center Triage Agent — "
                "Multi-zone disaster alert triage with finite resources, "
                "temporal pressure, adversarial misinformation, and cascade dynamics. "
                "Three tasks: easy (1 zone), medium (3 zones), hard (5 zones + spoofed alerts)."
            ),
            version="1.0.0",
            author="DisasterResponseEnv",
        )

    # ------------------------------------------------------------------
    # Reward, resources, and zone stress
    # ------------------------------------------------------------------

    def _calculate_reward(
        self,
        action: DisasterAction,
        alert: Dict[str, Any],
        zone: Dict[str, Any],
    ) -> Tuple[float, str]:
        at = action.action_type
        is_real: bool = alert["is_real"]
        delay: int = self._step - alert["arrival_step"]
        time_decay: float = math.exp(-0.07 * delay)

        if at == ActionType.DISPATCH_RESCUE:
            if self._resources["rescue_teams_available"] <= 0:
                return -0.40, "No rescue teams available — wasted action"
            if is_real:
                return round(1.0 * time_decay, 4), f"Correct rescue dispatch (delay={delay}, decay={time_decay:.2f})"
            return -0.30, "False rescue dispatch — rescue team wasted on noise"

        if at == ActionType.DISPATCH_MEDICAL:
            if self._resources["medical_units_available"] <= 0:
                return -0.35, "No medical units available — wasted action"
            if is_real:
                return round(0.70 * time_decay, 4), f"Correct medical dispatch (delay={delay})"
            return -0.25, "False medical dispatch — medical unit wasted on noise"

        if at == ActionType.ISSUE_EVACUATION:
            if self._resources["broadcast_credits"] <= 0:
                return -0.50, "No broadcast credits remaining — evacuation failed"
            if zone["stress"] >= 0.65:
                return 0.40, f"Proactive evacuation on high-stress zone (stress={zone['stress']:.2f})"
            return -0.15, f"Evacuation on low-stress zone (stress={zone['stress']:.2f}) — credit wasted"

        if at == ActionType.REQUEST_MORE_INFO:
            return -0.05, "Deliberation — gathering more information"

        if at == ActionType.DISMISS_FALSE_ALARM:
            if is_real:
                return -1.00, "CRITICAL: Real victim dismissed as false alarm"
            return 0.20, "Correctly identified false alarm — resource preserved"

        return 0.0, "Unknown action"

    def _apply_resource_change(self, action: DisasterAction) -> None:
        at = action.action_type
        if at == ActionType.DISPATCH_RESCUE and self._resources["rescue_teams_available"] > 0:
            self._resources["rescue_teams_available"] -= 1
            ret = self._step + self._task_cfg["rescue_lock_steps"]
            self._resources["rescue_teams_locked"].append({"returns_at_step": ret})
        elif at == ActionType.DISPATCH_MEDICAL and self._resources["medical_units_available"] > 0:
            self._resources["medical_units_available"] -= 1
            ret = self._step + self._task_cfg["medical_lock_steps"]
            self._resources["medical_units_locked"].append({"returns_at_step": ret})
        elif at == ActionType.ISSUE_EVACUATION and self._resources["broadcast_credits"] > 0:
            self._resources["broadcast_credits"] -= 1

    def _unlock_resources(self) -> None:
        released = [t for t in self._resources["rescue_teams_locked"] if t["returns_at_step"] <= self._step]
        self._resources["rescue_teams_locked"] = [t for t in self._resources["rescue_teams_locked"] if t["returns_at_step"] > self._step]
        self._resources["rescue_teams_available"] += len(released)

        released = [u for u in self._resources["medical_units_locked"] if u["returns_at_step"] <= self._step]
        self._resources["medical_units_locked"] = [u for u in self._resources["medical_units_locked"] if u["returns_at_step"] > self._step]
        self._resources["medical_units_available"] += len(released)

    def _update_zone_stresses(self, action: DisasterAction, acted_zone_id: str) -> None:
        """Acted-upon zone gets action benefit; all other zones grow 12% per step."""
        at = action.action_type
        for zone in self._zones:
            if zone["id"] == acted_zone_id:
                if at == ActionType.DISPATCH_RESCUE:
                    zone["stress"] = max(0.0, zone["stress"] - 0.30)
                elif at == ActionType.DISPATCH_MEDICAL:
                    zone["stress"] = max(0.0, zone["stress"] - 0.15)
                elif at == ActionType.ISSUE_EVACUATION:
                    zone["stress"] = max(0.0, zone["stress"] - 0.20)
                elif at == ActionType.REQUEST_MORE_INFO:
                    zone["stress"] = min(1.0, zone["stress"] * 1.05)
                # dismiss_false_alarm: stress unchanged
            else:
                zone["stress"] = min(1.0, zone["stress"] * 1.12)

    def _apply_cascade(self) -> None:
        """Task 3: propagate stress from source zones to cascade-linked zones."""
        for (src_id, dst_id) in self._task_cfg.get("cascade_pairs", []):
            src = self._get_zone(src_id)
            dst = self._get_zone(dst_id)
            if src and dst and src["stress"] > 0.80:
                spill = (src["stress"] - 0.80) * 0.25
                old = dst["stress"]
                dst["stress"] = min(1.0, dst["stress"] + spill)
                if dst["stress"] > old + 0.001:
                    self._cascade_events += 1

    # ------------------------------------------------------------------
    # Alert generation
    # ------------------------------------------------------------------

    def _generate_alerts(self) -> List[Dict[str, Any]]:
        """Generate probabilistic alerts from zones based on current stress."""
        return [
            self._create_alert(zone)
            for zone in self._zones
            if self._rng.random() < zone["stress"] * 0.6
        ]

    def _create_alert(self, zone: Dict[str, Any]) -> Dict[str, Any]:
        real_rate: float = self._task_cfg["real_alert_rate"]
        spoof_rate: float = self._task_cfg.get("spoof_rate", 0.0)

        is_real = self._rng.random() < real_rate
        is_spoofed = False
        if not is_real and spoof_rate > 0:
            is_spoofed = self._rng.random() < spoof_rate

        if is_real:
            weights = [0.35, 0.35, 0.20, 0.10]
        elif is_spoofed:
            weights = [0.03, 0.05, 0.30, 0.62]
        else:
            weights = [0.05, 0.10, 0.38, 0.47]

        source = self._rng.choices(
            ["sensor", "radio", "sms", "social_media"], weights=weights
        )[0]

        if is_real:
            base = self._rng.uniform(0.45, 0.95)
            severity = min(1.0, base + zone["stress"] * 0.12)
        elif is_spoofed:
            severity = self._rng.uniform(0.75, 0.95)
        else:
            severity = self._rng.uniform(0.05, 0.40)

        # randint call must match oracle._oracle_create_alert for RNG sync
        _msg_roll = self._rng.randint(0, 7)
        if is_real:
            msgs = _REAL_MSGS[source]
            tmpl = msgs[_msg_roll % len(msgs)]
        elif is_spoofed:
            tmpl = _SPOOF_MSGS[_msg_roll % len(_SPOOF_MSGS)]
        else:
            opts = _NOISE_MSGS.get(source, _NOISE_MSGS["sms"])
            tmpl = opts[_msg_roll % len(opts)]

        # Deterministic alert_id from seeded RNG (must match oracle._oracle_create_alert)
        alert_id = f"{self._rng.randint(0x10000000, 0xFFFFFFFF):08x}"

        return {
            "alert_id": alert_id,
            "zone_id": zone["id"],
            "zone_name": zone["name"],
            "source": source,
            "severity": round(severity, 3),
            "message": tmpl.format(zone=zone["name"]),
            "is_real": is_real,
            "is_spoofed": is_spoofed,
            "arrival_step": self._step,
            "deliberation_count": 0,
        }

    def _advance_alert(self) -> None:
        """Pop the next alert from queue (generate one if empty)."""
        if self._alert_queue:
            self._current_alert = self._alert_queue.pop(0)
        else:
            self._current_alert = self._create_alert(self._highest_stress_zone())

    def _highest_stress_zone(self) -> Dict[str, Any]:
        return max(self._zones, key=lambda z: z["stress"])

    # ------------------------------------------------------------------
    # Episode finalisation and oracle scoring
    # ------------------------------------------------------------------

    def _finalise_episode(self) -> None:
        """Compute oracle reward on the same seed and normalise agent score."""
        oracle_total = self._run_oracle(self._episode_seed)
        self._oracle_reward = oracle_total
        if oracle_total > 0:
            raw = self._cumulative_reward / oracle_total
        else:
            raw = 0.5  # neutral fallback — oracle itself performed near-zero

        # Keep scores well inside (0, 1) so validator-side rounding never becomes 0.00/1.00.
        # raw is first clipped to [0, 1], then linearly mapped to [0.1, 0.9].
        clipped = min(1.0, max(0.0, raw))
        self._normalized_score = round(0.1 + 0.8 * clipped, 4)

        DisasterResponseEnvironment._last_grader_data = {
            "task": self._task_name,
            "score": self._normalized_score,
            "agent_reward": round(self._cumulative_reward, 4),
            "oracle_reward": round(self._oracle_reward, 4),
            "steps_completed": self._step,
            "metrics": {
                "correct_dispatches": self._correct_dispatches,
                "false_alarm_dispatches": self._false_alarm_dispatches,
                "correct_dismissals": self._correct_dismissals,
                "missed_real_alerts": self._missed_real_alerts,
                "cascade_events": self._cascade_events,
                "success_threshold": self._task_cfg["success_threshold"],
                "passed": self._normalized_score >= self._task_cfg["success_threshold"],
            },
        }

    def _run_oracle(self, seed: int) -> float:
        """Run the heuristic oracle on the same scenario; return total reward."""
        cfg = self._task_cfg
        rng = random.Random(seed)

        zones = [
            {"id": z["id"], "name": z["name"], "stress": z["initial_stress"]}
            for z in cfg["zones"]
        ]
        resources: Dict[str, Any] = {
            "rescue_teams_available": cfg["initial_rescue_teams"],
            "rescue_teams_locked": [],
            "medical_units_available": cfg["initial_medical_units"],
            "medical_units_locked": [],
            "broadcast_credits": cfg["broadcast_credits"],
        }
        queue: List[Dict[str, Any]] = []
        total_reward: float = 0.0

        def gen_alerts(step: int) -> List[Dict[str, Any]]:
            return [
                _oracle_create_alert(z, cfg, rng, step)
                for z in zones
                if rng.random() < z["stress"] * 0.6
            ]

        def unlock(step: int) -> None:
            freed = [t for t in resources["rescue_teams_locked"] if t["returns_at_step"] <= step]
            resources["rescue_teams_locked"] = [t for t in resources["rescue_teams_locked"] if t["returns_at_step"] > step]
            resources["rescue_teams_available"] += len(freed)
            freed = [u for u in resources["medical_units_locked"] if u["returns_at_step"] <= step]
            resources["medical_units_locked"] = [u for u in resources["medical_units_locked"] if u["returns_at_step"] > step]
            resources["medical_units_available"] += len(freed)

        def cascade() -> None:
            for (sid, did) in cfg.get("cascade_pairs", []):
                s = next((z for z in zones if z["id"] == sid), None)
                d = next((z for z in zones if z["id"] == did), None)
                if s and d and s["stress"] > 0.80:
                    d["stress"] = min(1.0, d["stress"] + (s["stress"] - 0.80) * 0.25)

        queue.extend(gen_alerts(0))
        if not queue:
            queue.append(_oracle_create_alert(max(zones, key=lambda z: z["stress"]), cfg, rng, 0))

        current = queue.pop(0)
        current["deliberation_count"] = 0

        for step in range(1, cfg["max_steps"] + 1):
            if current is None:
                if queue:
                    current = queue.pop(0)
                else:
                    current = _oracle_create_alert(max(zones, key=lambda z: z["stress"]), cfg, rng, step)
                current["deliberation_count"] = 0

            action_type = oracle_decide(current, resources, zones)
            zone = next((z for z in zones if z["id"] == current["zone_id"]), zones[0])
            is_real: bool = current["is_real"]
            delay: int = step - current["arrival_step"]
            decay = math.exp(-0.07 * delay)

            if action_type == ActionType.DISPATCH_RESCUE:
                if resources["rescue_teams_available"] <= 0:
                    total_reward -= 0.40
                elif is_real:
                    total_reward += 1.0 * decay
                else:
                    total_reward -= 0.30
            elif action_type == ActionType.DISPATCH_MEDICAL:
                if resources["medical_units_available"] <= 0:
                    total_reward -= 0.35
                elif is_real:
                    total_reward += 0.70 * decay
                else:
                    total_reward -= 0.25
            elif action_type == ActionType.ISSUE_EVACUATION:
                if resources["broadcast_credits"] <= 0:
                    total_reward -= 0.50
                elif zone["stress"] >= 0.65:
                    total_reward += 0.40
                else:
                    total_reward -= 0.15
            elif action_type == ActionType.REQUEST_MORE_INFO:
                total_reward -= 0.05
            elif action_type == ActionType.DISMISS_FALSE_ALARM:
                total_reward += 0.20 if not is_real else -1.00

            # Apply resource change
            if action_type == ActionType.DISPATCH_RESCUE and resources["rescue_teams_available"] > 0:
                resources["rescue_teams_available"] -= 1
                resources["rescue_teams_locked"].append({"returns_at_step": step + cfg["rescue_lock_steps"]})
            elif action_type == ActionType.DISPATCH_MEDICAL and resources["medical_units_available"] > 0:
                resources["medical_units_available"] -= 1
                resources["medical_units_locked"].append({"returns_at_step": step + cfg["medical_lock_steps"]})
            elif action_type == ActionType.ISSUE_EVACUATION and resources["broadcast_credits"] > 0:
                resources["broadcast_credits"] -= 1

            # Update zone stresses
            for z in zones:
                if z["id"] == current["zone_id"]:
                    if action_type == ActionType.DISPATCH_RESCUE:
                        z["stress"] = max(0.0, z["stress"] - 0.30)
                    elif action_type == ActionType.DISPATCH_MEDICAL:
                        z["stress"] = max(0.0, z["stress"] - 0.15)
                    elif action_type == ActionType.ISSUE_EVACUATION:
                        z["stress"] = max(0.0, z["stress"] - 0.20)
                    elif action_type == ActionType.REQUEST_MORE_INFO:
                        z["stress"] = min(1.0, z["stress"] * 1.05)
                else:
                    z["stress"] = min(1.0, z["stress"] * 1.12)

            cascade()
            unlock(step)
            queue.extend(gen_alerts(step))

            if action_type == ActionType.REQUEST_MORE_INFO:
                current["deliberation_count"] += 1
            else:
                if queue:
                    current = queue.pop(0)
                    current["deliberation_count"] = 0
                else:
                    current = _oracle_create_alert(max(zones, key=lambda z: z["stress"]), cfg, rng, step)
                    current["deliberation_count"] = 0

        return max(total_reward, 0.001)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def _build_observation(self, reward: float, done: bool) -> DisasterObservation:
        return DisasterObservation(
            done=done,
            reward=round(reward, 4),
            current_alert=self._public_alert(self._current_alert),
            zones=self._serialise_zones(),
            resources=self._serialise_resources(),
            step=self._step,
            max_steps=self._task_cfg["max_steps"],
            task_name=self._task_name,
            task_description=self._task_cfg["description"],
            cumulative_reward=round(self._cumulative_reward, 4),
            last_action_result=self._last_action_result,
            normalized_score=self._normalized_score if done else None,
        )

    def _public_alert(self, alert: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Strip hidden fields (is_real, is_spoofed) before exposing to agent."""
        if alert is None:
            return None
        return {
            "alert_id": alert["alert_id"],
            "zone_id": alert["zone_id"],
            "zone_name": alert["zone_name"],
            "source": alert["source"],
            "severity": alert["severity"],
            "message": alert["message"],
            "arrival_step": alert["arrival_step"],
            "deliberation_count": alert["deliberation_count"],
        }

    def _serialise_zones(self) -> List[Dict[str, Any]]:
        return [
            {
                "zone_id": z["id"],
                "name": z["name"],
                "stress": round(z["stress"], 4),
                "pending_alerts": sum(1 for a in self._alert_queue if a["zone_id"] == z["id"]),
            }
            for z in self._zones
        ]

    def _serialise_resources(self) -> Dict[str, Any]:
        return {
            "rescue_teams_available": self._resources["rescue_teams_available"],
            "rescue_teams_locked": list(self._resources["rescue_teams_locked"]),
            "medical_units_available": self._resources["medical_units_available"],
            "medical_units_locked": list(self._resources["medical_units_locked"]),
            "broadcast_credits": self._resources["broadcast_credits"],
        }

    def _get_zone(self, zone_id: str) -> Optional[Dict[str, Any]]:
        return next((z for z in self._zones if z["id"] == zone_id), None)
