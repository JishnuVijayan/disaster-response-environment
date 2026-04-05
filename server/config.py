"""Task configuration catalogue for DisasterResponseEnv."""

from __future__ import annotations

from typing import Any, Dict

TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "task1_flood_easy": {
        "description": (
            "Single Zone Flood Triage — A residential neighbourhood is experiencing "
            "a severe flash flood. Alerts stream in from one zone; ~75 % are genuine. "
            "Resources are plentiful, locks are short (2 steps). "
            "Learn the fundamental real-vs-noise signal before tackling multi-zone."
        ),
        "difficulty": "easy",
        "num_zones": 1,
        "max_steps": 30,
        "rescue_lock_steps": 2,
        "medical_lock_steps": 2,
        "initial_rescue_teams": 3,
        "initial_medical_units": 3,
        "broadcast_credits": 2,
        "real_alert_rate": 0.75,
        "spoof_rate": 0.0,
        "cascade_pairs": [],
        "zones": [
            {"id": "zone_a", "name": "Riverside District", "initial_stress": 0.55},
        ],
        "success_threshold": 0.75,
    },
    "task2_multizone_medium": {
        "description": (
            "Multi-Zone Flash Flood — The flood spreads across three independent zones. "
            "Resource locks extend to 5 / 4 steps. Spatial prioritisation now matters: "
            "which zone do you help when you can only act on one alert per step? "
            "Evacuation credits become strategically valuable."
        ),
        "difficulty": "medium",
        "num_zones": 3,
        "max_steps": 55,
        "rescue_lock_steps": 5,
        "medical_lock_steps": 4,
        "initial_rescue_teams": 2,
        "initial_medical_units": 2,
        "broadcast_credits": 3,
        "real_alert_rate": 0.70,
        "spoof_rate": 0.0,
        "cascade_pairs": [],
        "zones": [
            {"id": "zone_a", "name": "Downtown Core", "initial_stress": 0.45},
            {"id": "zone_b", "name": "Harbor District", "initial_stress": 0.60},
            {"id": "zone_c", "name": "Westside Heights", "initial_stress": 0.35},
        ],
        "success_threshold": 0.60,
    },
    "task3_compound_hard": {
        "description": (
            "Cascading Compound Disaster — Earthquake triggers secondary flooding across "
            "five zones. Resource locks: 8 / 6 steps. 20 % of noise alerts are deliberately "
            "spoofed with inflated severity to drain resources. High-stress zones cascade "
            "stress into adjacent zones. Only long-horizon strategic planning succeeds."
        ),
        "difficulty": "hard",
        "num_zones": 5,
        "max_steps": 130,
        "rescue_lock_steps": 8,
        "medical_lock_steps": 6,
        "initial_rescue_teams": 3,
        "initial_medical_units": 3,
        "broadcast_credits": 5,
        "real_alert_rate": 0.65,
        "spoof_rate": 0.20,
        "cascade_pairs": [("zone_a", "zone_d"), ("zone_b", "zone_e")],
        "zones": [
            {"id": "zone_a", "name": "Old City Center", "initial_stress": 0.50},
            {"id": "zone_b", "name": "Industrial Waterfront", "initial_stress": 0.45},
            {"id": "zone_c", "name": "Medical District", "initial_stress": 0.30},
            {"id": "zone_d", "name": "Suburban East", "initial_stress": 0.25},
            {"id": "zone_e", "name": "Port Authority", "initial_stress": 0.35},
        ],
        "success_threshold": 0.45,
    },
}

DEFAULT_TASK: str = "task1_flood_easy"
