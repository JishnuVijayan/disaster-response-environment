"""Alert message templates for DisasterResponseEnv.

Templates use {zone} as a placeholder for the zone name.
Hidden from the agent — used only during alert creation.
"""

from __future__ import annotations

from typing import Dict, List

_REAL_MSGS: Dict[str, List[str]] = {
    "sensor": [
        "Structural sensor in {zone} exceeds critical stress threshold — collapse imminent.",
        "Flood sensor in {zone}: water level 2.8 m and rising rapidly.",
        "Gas sensor in {zone} reports dangerous CH4 concentration.",
        "Seismic monitor at {zone} confirms building collapse event.",
    ],
    "radio": [
        "Field unit Delta reports multiple casualties in {zone}; requesting immediate rescue.",
        "Ground team at {zone}: partial collapse, ~15 people trapped.",
        "Medical unit reporting mass-casualty event in {zone}; need backup.",
        "Firefighters at {zone} confirm active rescue op; requesting medical support.",
    ],
    "sms": [
        "HELP trapped under rubble in {zone}!! Many people here send help",
        "Family stuck on roof in {zone} floodwater rising please rescue",
        "Fire and collapse in {zone} neighbors are hurt please come",
        "At {zone} — building fell, multiple injured, no escape route",
    ],
    "social_media": [
        "URGENT: Major collapse in {zone}! People trapped! #SOS #disaster",
        "Reporting from {zone}: flooding catastrophic, rescue needed NOW",
        "Buildings down in {zone}. Survivors need help. #rescue",
        "On the ground {zone}: at least 20 people need immediate evacuation",
    ],
}

_NOISE_MSGS: Dict[str, List[str]] = {
    "sms": [
        "Is there danger in {zone}? I heard some sounds outside.",
        "My dog is acting strange near {zone}, is there a gas leak?",
        "Roads in {zone} look wet — is it safe to drive?",
        "Saw emergency vehicles near {zone}, what is happening?",
    ],
    "social_media": [
        "Crazy scenes near {zone}! Stay safe everyone! #disaster",
        "Heard something from {zone} direction — hope everyone ok",
        "Is anyone in {zone} ok? Worried about the situation",
        "People say {zone} is affected but I cannot see anything from here",
    ],
    "sensor": [
        "Vibration sensor in {zone} shows minor reading — likely traffic.",
        "Humidity sensor in {zone} at elevated level (weather-related).",
    ],
    "radio": [
        "Unit requesting status update for {zone}, no confirmed incidents yet.",
        "Patrol at {zone}: routine check, situation appears normal.",
    ],
}

_SPOOF_MSGS: List[str] = [
    "EMERGENCY: {zone} is completely destroyed, everyone needs help NOW!!!",
    "CRITICAL: Entire {zone} district underwater, hundreds trapped — RESPOND IMMEDIATELY",
    "MAYDAY {zone}: Building collapse cascade, 50+ casualties — all units respond",
    "RED ALERT {zone}: Infrastructure failure, mass-casualty — priority-1 response",
]
