"""
DisasterResponseEnv — Baseline Inference Script
================================================

MANDATORY ENVIRONMENT VARIABLES
  API_BASE_URL      LLM API endpoint (default: https://router.huggingface.co/v1)
  MODEL_NAME        Model identifier
  HF_TOKEN          Your Hugging Face / API key
  ENV_BASE_URL      Running DisasterResponseEnv server URL (default: http://localhost:8000)

STDOUT FORMAT (required by OpenEnv spec)
  [START] task=<task_name> env=disaster_response model=<model_name>
  [WARN]  <message>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...>

FALLBACK BEHAVIOUR
  If HF_TOKEN / API_KEY is missing or the LLM becomes unreachable mid-episode,
  inference.py automatically switches to a built-in heuristic fallback policy
  (same logic as the oracle in server/oracle.py) so episodes always run to
  completion.  A [WARN] line is printed whenever the fallback is activated.

Run all 3 tasks:
  python inference.py

Run a specific task:
  DISASTER_TASK=task2_multizone_medium python inference.py
"""

from __future__ import annotations

import json
import os
import re
import textwrap
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY: Optional[str] = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
ENV_BASE_URL: str = os.getenv("ENV_BASE_URL", "http://localhost:8000").rstrip("/")
DISASTER_TASK: Optional[str] = os.getenv("DISASTER_TASK")  # None → run all 3
# MAX_STEPS_CAP: optional hard cap override. When 0 or unset the per-task
# max_steps from the reset response is used (task3_compound_hard = 130).
_max_steps_env: str = os.getenv("MAX_STEPS", "0").strip()
MAX_STEPS_CAP: Optional[int] = (int(_max_steps_env) or None) if _max_steps_env else None
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "256"))
BENCHMARK: str = "disaster_response"

ALL_TASKS: List[str] = [
    "task1_flood_easy",
    "task2_multizone_medium",
    "task3_compound_hard",
]

SYSTEM_PROMPT: str = textwrap.dedent(
    """
    You are an AI Emergency Triage Coordinator inside an Emergency Operations Center (EOC).
    Disaster alerts are arriving from multiple geographic zones. You must decide how to respond
    to each alert to save the most lives while managing finite resources.

    AVAILABLE ACTIONS (respond with exactly one JSON object):
      {"action_type": "dispatch_rescue",    "alert_id": "<id>"}  — Deploy rescue team (locks resource K steps)
      {"action_type": "dispatch_medical",   "alert_id": "<id>"}  — Deploy medical unit (locks resource K steps)
      {"action_type": "issue_evacuation",   "alert_id": "<id>"}  — Issue evacuation order (uses broadcast credit)
      {"action_type": "request_more_info",  "alert_id": "<id>"}  — Deliberate one step (costs -0.05, keeps same alert)
      {"action_type": "dismiss_false_alarm","alert_id": "<id>"}  — Classify as noise

    DECISION HEURISTICS:
    - Sensor / radio alerts with severity > 0.80 → dispatch_rescue immediately
    - Social media / SMS alerts with high severity → request_more_info first (may be spoofed)
    - Severity < 0.30 → likely noise → dismiss_false_alarm
    - Zone stress ≥ 0.65 and broadcast credit available → issue_evacuation
    - Missing a real victim costs -1.00. Wasting a resource costs -0.30.
    - Respond quickly: reward decays with delay.

    Respond with ONLY a valid JSON object. No explanation, no extra text.
    """
).strip()


# ---------------------------------------------------------------------------
# Logging helpers (OpenEnv spec format)
# ---------------------------------------------------------------------------

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def log_warn(message: str) -> None:
    print(f"[WARN]  {message}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Environment HTTP helpers
# ---------------------------------------------------------------------------

def env_reset(task_name: str, seed: int = 42) -> Dict[str, Any]:
    resp = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_name": task_name, "seed": seed},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("observation", data)


def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"action": action},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(obs: Dict[str, Any], history: List[str]) -> str:
    alert = obs.get("current_alert") or {}
    zones = obs.get("zones", [])
    resources = obs.get("resources", {})
    step = obs.get("step", 0)
    max_steps = obs.get("max_steps", 0)

    zone_lines = "\n".join(
        f"  {z['name']} (id={z['zone_id']}): stress={z['stress']:.2f}, "
        f"pending_alerts={z.get('pending_alerts', 0)}"
        for z in zones
    )

    locked_rescue = resources.get("rescue_teams_locked", [])
    locked_medical = resources.get("medical_units_locked", [])

    history_text = "\n".join(history[-5:]) if history else "None"

    prompt = textwrap.dedent(
        f"""
        === EOC TRIAGE DECISION REQUIRED ===
        Step: {step} / {max_steps}
        Task: {obs.get('task_name', '')}
        Cumulative reward so far: {obs.get('cumulative_reward', 0.0):.2f}

        CURRENT ALERT:
          alert_id : {alert.get('alert_id', 'none')}
          zone     : {alert.get('zone_name', '?')} ({alert.get('zone_id', '?')})
          source   : {alert.get('source', '?')}
          severity : {alert.get('severity', 0.0):.3f}
          message  : {alert.get('message', '(none)')}
          waiting  : {step - alert.get('arrival_step', step)} steps
          deliber. : {alert.get('deliberation_count', 0)} times

        ZONE STATUS:
        {zone_lines}

        RESOURCES:
          Rescue teams available : {resources.get('rescue_teams_available', 0)}
          Rescue teams locked    : {len(locked_rescue)} (returns at steps {[t['returns_at_step'] for t in locked_rescue]})
          Medical units available: {resources.get('medical_units_available', 0)}
          Medical units locked   : {len(locked_medical)} (returns at steps {[t['returns_at_step'] for t in locked_medical]})
          Broadcast credits      : {resources.get('broadcast_credits', 0)}

        RECENT HISTORY:
        {history_text}

        Respond with ONE JSON object only.
        """
    ).strip()
    return prompt


# ---------------------------------------------------------------------------
# LLM action parser
# ---------------------------------------------------------------------------

def parse_action(response_text: str, alert_id: str) -> Dict[str, Any]:
    """Extract action JSON from LLM response. Fallback to dismiss on parse error."""
    text = response_text.strip()
    # Try direct JSON parse
    try:
        obj = json.loads(text)
        if "action_type" in obj:
            if "alert_id" not in obj or not obj["alert_id"]:
                obj["alert_id"] = alert_id
            return obj
    except json.JSONDecodeError:
        pass
    # Find first JSON-like block
    match = re.search(r"\{[^}]+\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if "action_type" in obj:
                obj.setdefault("alert_id", alert_id)
                return obj
        except json.JSONDecodeError:
            pass
    # Keyword-based fallback
    text_lower = text.lower()
    if "dispatch_rescue" in text_lower:
        return {"action_type": "dispatch_rescue", "alert_id": alert_id}
    if "dispatch_medical" in text_lower:
        return {"action_type": "dispatch_medical", "alert_id": alert_id}
    if "issue_evacuation" in text_lower:
        return {"action_type": "issue_evacuation", "alert_id": alert_id}
    if "request_more_info" in text_lower or "more_info" in text_lower:
        return {"action_type": "request_more_info", "alert_id": alert_id}
    return {"action_type": "dismiss_false_alarm", "alert_id": alert_id}


# ---------------------------------------------------------------------------
# Heuristic fallback policy
# ---------------------------------------------------------------------------

def fallback_action(obs: Dict[str, Any], alert_id: str) -> Dict[str, Any]:
    """
    Rule-based heuristic policy used when the LLM is unavailable.
    Mirrors the oracle in server/oracle.py — uses only observable signals
    (source, severity, zone stress, resources) with no access to is_real.
    """
    alert = obs.get("current_alert") or {}
    severity: float = alert.get("severity", 0.0)
    source: str = alert.get("source", "sms")
    deliberation: int = alert.get("deliberation_count", 0)
    zone_id: str = alert.get("zone_id", "")
    resources = obs.get("resources", {})
    zones = obs.get("zones", [])

    zone_stress: float = next(
        (z["stress"] for z in zones if z.get("zone_id") == zone_id), 0.0
    )
    has_rescue: bool = resources.get("rescue_teams_available", 0) > 0
    has_medical: bool = resources.get("medical_units_available", 0) > 0
    has_credits: bool = resources.get("broadcast_credits", 0) > 0

    if severity < 0.30:
        action_type = "dismiss_false_alarm"

    elif severity >= 0.80 and source in ("sensor", "radio"):
        if has_rescue:
            action_type = "dispatch_rescue"
        elif has_medical:
            action_type = "dispatch_medical"
        elif deliberation == 0:
            action_type = "request_more_info"
        else:
            action_type = "dismiss_false_alarm"

    elif severity >= 0.80 and source in ("sms", "social_media"):
        if deliberation == 0:
            action_type = "request_more_info"
        elif has_rescue:
            action_type = "dispatch_rescue"
        else:
            action_type = "dismiss_false_alarm"

    elif 0.30 <= severity < 0.80 and source in ("sensor", "radio"):
        if has_rescue:
            action_type = "dispatch_rescue"
        elif has_medical:
            action_type = "dispatch_medical"
        elif zone_stress >= 0.65 and has_credits:
            action_type = "issue_evacuation"
        elif deliberation == 0:
            action_type = "request_more_info"
        else:
            action_type = "dismiss_false_alarm"

    elif 0.30 <= severity < 0.80 and source in ("sms", "social_media"):
        if deliberation == 0:
            action_type = "request_more_info"
        elif has_rescue:
            action_type = "dispatch_rescue"
        else:
            action_type = "dismiss_false_alarm"

    elif zone_stress >= 0.85 and has_credits:
        action_type = "issue_evacuation"

    else:
        action_type = "dismiss_false_alarm"

    return {"action_type": action_type, "alert_id": alert_id}


# ---------------------------------------------------------------------------
# Single-task episode runner
# ---------------------------------------------------------------------------

def run_episode(client: Optional[OpenAI], task_name: str, seed: int = 42) -> None:
    log_start(task=task_name, model=MODEL_NAME)

    # Start in fallback mode immediately if no client is available
    using_fallback: bool = client is None
    if using_fallback:
        log_warn(
            f"No LLM client available — running {task_name} with heuristic fallback policy."
        )

    rewards: List[float] = []
    steps_taken: int = 0
    success: bool = False
    history: List[str] = []
    last_error: Optional[str] = None

    try:
        try:
            obs = env_reset(task_name, seed=seed)
        except Exception as exc:
            last_error = str(exc)[:200]
            log_warn(f"Environment reset failed: {last_error}")
            log_end(success=False, steps=0, rewards=[])
            return

        task_max_steps: int = int(obs.get("max_steps") or 50)
        effective_max_steps: int = (
            min(MAX_STEPS_CAP, task_max_steps) if MAX_STEPS_CAP else task_max_steps
        )

        for step in range(1, effective_max_steps + 1):
            if obs.get("done"):
                break

            alert = obs.get("current_alert") or {}
            alert_id = alert.get("alert_id", "")

            # ----------------------------------------------------------------
            # Decide action: LLM when available, fallback heuristic otherwise
            # ----------------------------------------------------------------
            if using_fallback:
                action = fallback_action(obs, alert_id)
                last_error = None
            else:
                user_prompt = build_prompt(obs, history)
                try:
                    completion = client.chat.completions.create(  # type: ignore[union-attr]
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                    )
                    response_text = completion.choices[0].message.content or ""
                    last_error = None
                    action = parse_action(response_text, alert_id)
                except Exception as exc:
                    reason = str(exc)[:120]
                    log_warn(
                        f"LLM call failed at step {step} "
                        f"({reason}) — switching to heuristic fallback for remaining steps."
                    )
                    using_fallback = True
                    action = fallback_action(obs, alert_id)
                    last_error = f"llm_failed:{reason[:80]}"

            action_str = json.dumps(action, separators=(",", ":"))

            # ----------------------------------------------------------------
            # Step the environment
            # ----------------------------------------------------------------
            try:
                result = env_step(action)
                obs = result.get("observation", result)
                reward = float(result.get("reward") or 0.0)
                done = bool(result.get("done", False))
            except Exception as exc:
                reward = 0.0
                done = False
                last_error = str(exc)[:120]

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=last_error)

            action_label = action.get("action_type", "?")
            history.append(f"Step {step}: {action_label} → reward {reward:+.2f}")

            if done:
                norm_score = obs.get("normalized_score")
                threshold = {
                    "task1_flood_easy": 0.75,
                    "task2_multizone_medium": 0.60,
                    "task3_compound_hard": 0.45,
                }.get(task_name, 0.5)
                success = (norm_score is not None and norm_score >= threshold) or (reward > 0)
                break
        else:
            success = False

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    client: Optional[OpenAI] = None

    if not API_KEY:
        log_warn(
            "HF_TOKEN / API_KEY is not set — LLM unavailable. "
            "All tasks will run with heuristic fallback policy."
        )
    else:
        try:
            client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        except Exception as exc:
            log_warn(
                f"Failed to initialise OpenAI client ({exc}) — "
                "falling back to heuristic policy for all tasks."
            )

    tasks_to_run = [DISASTER_TASK] if DISASTER_TASK else ALL_TASKS

    for i, task_name in enumerate(tasks_to_run):
        seed = 42 + i
        run_episode(client, task_name, seed=seed)


if __name__ == "__main__":
    main()
