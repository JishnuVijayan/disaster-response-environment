"""Oracle policy and alert factory for DisasterResponseEnv.

These are pure functions — no environment instance required.
oracle_decide() uses only the same observable information as the agent (no is_real).
_oracle_create_alert() mirrors _create_alert() RNG call-by-call to stay in sync.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List
from uuid import uuid4

try:
    from ..models import ActionType
except ImportError:
    from models import ActionType


def oracle_decide(
    alert: Dict[str, Any],
    resources: Dict[str, Any],
    zones: List[Dict[str, Any]],
) -> ActionType:
    """
    Rule-based oracle that uses only the same observable information as the agent.
    Does NOT use the hidden is_real field.
    Returns the oracle's chosen ActionType for a given alert.
    """
    severity: float = alert["severity"]
    source: str = alert["source"]
    deliberation: int = alert.get("deliberation_count", 0)
    zone_id: str = alert["zone_id"]

    zone_stress = next(
        (z["stress"] for z in zones if z["id"] == zone_id), 0.0
    )
    has_rescue = resources["rescue_teams_available"] > 0
    has_medical = resources["medical_units_available"] > 0
    has_credits = resources["broadcast_credits"] > 0

    # Rule 1: obvious noise
    if severity < 0.30:
        return ActionType.DISMISS_FALSE_ALARM

    # Rule 2: high severity + reliable source → immediate rescue
    if severity >= 0.80 and source in ("sensor", "radio"):
        if has_rescue:
            return ActionType.DISPATCH_RESCUE
        if has_medical:
            return ActionType.DISPATCH_MEDICAL
        if deliberation == 0:
            return ActionType.REQUEST_MORE_INFO
        return ActionType.DISMISS_FALSE_ALARM

    # Rule 3: high severity + unreliable source → deliberate first, then commit
    if severity >= 0.80 and source in ("sms", "social_media"):
        if deliberation == 0:
            return ActionType.REQUEST_MORE_INFO
        if has_rescue:
            return ActionType.DISPATCH_RESCUE
        return ActionType.DISMISS_FALSE_ALARM

    # Rule 4: moderate severity + reliable source → dispatch
    if 0.30 <= severity < 0.80 and source in ("sensor", "radio"):
        if has_rescue:
            return ActionType.DISPATCH_RESCUE
        if has_medical:
            return ActionType.DISPATCH_MEDICAL
        if zone_stress >= 0.65 and has_credits:
            return ActionType.ISSUE_EVACUATION
        return ActionType.REQUEST_MORE_INFO if deliberation == 0 else ActionType.DISMISS_FALSE_ALARM

    # Rule 5: moderate severity + unreliable source → deliberate
    if 0.30 <= severity < 0.80 and source in ("sms", "social_media"):
        if deliberation == 0:
            return ActionType.REQUEST_MORE_INFO
        if has_rescue:
            return ActionType.DISPATCH_RESCUE
        return ActionType.DISMISS_FALSE_ALARM

    # Rule 6: proactive evacuation on very high stress zones
    if zone_stress >= 0.85 and has_credits:
        return ActionType.ISSUE_EVACUATION

    return ActionType.DISMISS_FALSE_ALARM


def _oracle_create_alert(
    zone: Dict[str, Any],
    cfg: Dict[str, Any],
    rng: random.Random,
    step: int,
) -> Dict[str, Any]:
    """
    Mirror of environment._create_alert() that makes the same RNG calls
    in the same order, keeping oracle and main-env RNG sequences in sync.
    """
    real_rate: float = cfg["real_alert_rate"]
    spoof_rate: float = cfg.get("spoof_rate", 0.0)

    is_real = rng.random() < real_rate
    is_spoofed = False
    if not is_real and spoof_rate > 0:
        is_spoofed = rng.random() < spoof_rate

    if is_real:
        weights = [0.35, 0.35, 0.20, 0.10]
    elif is_spoofed:
        weights = [0.03, 0.05, 0.30, 0.62]
    else:
        weights = [0.05, 0.10, 0.38, 0.47]

    source = rng.choices(["sensor", "radio", "sms", "social_media"], weights=weights)[0]

    if is_real:
        base = rng.uniform(0.45, 0.95)
        zone_boost = zone["stress"] * 0.12
        severity = min(1.0, base + zone_boost)
    elif is_spoofed:
        severity = rng.uniform(0.75, 0.95)
    else:
        severity = rng.uniform(0.05, 0.40)

    # Dummy call — matches the randint(0, 7) in _create_alert for message selection.
    # Required to keep oracle RNG in sync with the main environment RNG.
    _ = rng.randint(0, 7)

    return {
        "alert_id": str(uuid4())[:8],
        "zone_id": zone["id"],
        "zone_name": zone["name"],
        "source": source,
        "severity": round(severity, 3),
        "message": f"[{source.upper()}] Alert from {zone['name']} (severity={round(severity, 2)})",
        "is_real": is_real,
        "is_spoofed": is_spoofed,
        "arrival_step": step,
        "deliberation_count": 0,
    }
