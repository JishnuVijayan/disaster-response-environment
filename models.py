"""
Data models for the DisasterResponseEnv.

Defines the Action, Observation, and State Pydantic types
used by the Emergency Operations Center triage environment.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from openenv.core.env_server.types import Action, Observation, State


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """Five discrete actions available to the triage coordinator."""
    DISPATCH_RESCUE = "dispatch_rescue"
    DISPATCH_MEDICAL = "dispatch_medical"
    ISSUE_EVACUATION = "issue_evacuation"
    REQUEST_MORE_INFO = "request_more_info"
    DISMISS_FALSE_ALARM = "dismiss_false_alarm"


class AlertSource(str, Enum):
    """Signal channel through which an alert was received."""
    SENSOR = "sensor"
    RADIO = "radio"
    SMS = "sms"
    SOCIAL_MEDIA = "social_media"


# ---------------------------------------------------------------------------
# Nested observation sub-models (plain BaseModel, not Action/Observation)
# ---------------------------------------------------------------------------

class AlertObservation(BaseModel):
    """Details of the single active alert the agent must respond to."""
    model_config = {"arbitrary_types_allowed": True}

    alert_id: str = Field(..., description="Unique identifier for this alert")
    zone_id: str = Field(..., description="Originating zone identifier")
    zone_name: str = Field(..., description="Human-readable zone name")
    source: str = Field(..., description="Signal channel: sensor | radio | sms | social_media")
    severity: float = Field(..., ge=0.0, le=1.0, description="Reported urgency 0.0–1.0")
    message: str = Field(..., description="Raw alert text content")
    arrival_step: int = Field(..., description="Step at which this alert arrived")
    deliberation_count: int = Field(
        default=0,
        description="Times request_more_info has been used on this alert"
    )


class ZoneObservation(BaseModel):
    """Live state of one geographic zone."""
    model_config = {"arbitrary_types_allowed": True}

    zone_id: str = Field(..., description="Unique zone identifier")
    name: str = Field(..., description="Human-readable zone name")
    stress: float = Field(..., ge=0.0, le=1.0, description="Current stress level 0.0–1.0")
    pending_alerts: int = Field(
        default=0,
        description="Number of unhandled alerts from this zone in the queue"
    )


class ResourceObservation(BaseModel):
    """Current resource pool status."""
    model_config = {"arbitrary_types_allowed": True}

    rescue_teams_available: int = Field(..., description="Rescue teams ready to deploy")
    rescue_teams_locked: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Locked teams with their return step: [{'returns_at_step': N}]"
    )
    medical_units_available: int = Field(..., description="Medical units ready to deploy")
    medical_units_locked: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Locked medical units with their return step"
    )
    broadcast_credits: int = Field(..., description="Remaining evacuation broadcast credits")


# ---------------------------------------------------------------------------
# Primary Action / Observation / State models
# ---------------------------------------------------------------------------

class DisasterAction(Action):
    """
    Action taken by the Emergency Triage Coordinator agent.

    Each step the agent picks one action type and optionally supplies the
    alert_id it is responding to (must match the current active alert).
    """

    action_type: ActionType = Field(
        ...,
        description=(
            "Action to take on the current alert:\n"
            "  dispatch_rescue    — deploy a rescue team to the alert zone\n"
            "  dispatch_medical   — deploy a medical unit to the alert zone\n"
            "  issue_evacuation   — issue an evacuation order for the alert zone\n"
            "  request_more_info  — deliberate one step (keeps same alert active)\n"
            "  dismiss_false_alarm — classify alert as noise and close it"
        )
    )
    alert_id: Optional[str] = Field(
        default=None,
        description=(
            "ID of the alert being acted upon. "
            "Must match the current active alert's alert_id if provided."
        )
    )


class DisasterObservation(Observation):
    """
    Observation returned at each step of the triage episode.

    Contains the current active alert, world state, and resource inventory.
    The agent must not be shown the hidden `is_real` field on any alert —
    it must infer genuineness from observable signals (source, severity, zone stress).
    """

    current_alert: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Active alert the agent must respond to. "
            "Fields: alert_id, zone_id, zone_name, source, severity, message, "
            "arrival_step, deliberation_count"
        )
    )
    zones: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of ZoneObservation dicts: zone_id, name, stress, pending_alerts"
    )
    resources: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "ResourceObservation: rescue_teams_available, rescue_teams_locked, "
            "medical_units_available, medical_units_locked, broadcast_credits"
        )
    )
    step: int = Field(default=0, description="Current episode step (1-indexed)")
    max_steps: int = Field(default=0, description="Total steps in this episode")
    task_name: str = Field(default="", description="Active task identifier")
    task_description: str = Field(default="", description="Human-readable task description")
    cumulative_reward: float = Field(default=0.0, description="Sum of all rewards this episode")
    last_action_result: Optional[str] = Field(
        default=None,
        description="Plain-language description of the previous action outcome"
    )
    normalized_score: Optional[float] = Field(
        default=None,
        description=(
            "Oracle-normalized score in [0, 1]. "
            "Only populated when done=True."
        )
    )


class DisasterState(State):
    """
    Full internal environment state (returned by GET /state).

    Includes everything in DisasterObservation plus bookkeeping metrics
    that are useful for debugging and the grader.
    """

    task_name: str = Field(default="", description="Active task identifier")
    task_description: str = Field(default="", description="Task description")
    zones: List[Dict[str, Any]] = Field(default_factory=list)
    resources: Dict[str, Any] = Field(default_factory=dict)
    current_alert: Optional[Dict[str, Any]] = Field(default=None)
    alert_queue_size: int = Field(default=0, description="Alerts waiting in queue")
    cumulative_reward: float = Field(default=0.0)
    oracle_reward: float = Field(
        default=0.0,
        description="Oracle cumulative reward for the same episode (computed at episode end)"
    )
    normalized_score: Optional[float] = Field(
        default=None,
        description="Agent reward / Oracle reward, capped at 1.0"
    )
    total_alerts_processed: int = Field(default=0)
    correct_dispatches: int = Field(default=0)
    false_alarm_dispatches: int = Field(default=0)
    correct_dismissals: int = Field(default=0)
    missed_real_alerts: int = Field(default=0)
    cascade_events: int = Field(default=0, description="Number of cascade triggers (task 3)")
