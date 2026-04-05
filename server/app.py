"""
FastAPI application for DisasterResponseEnv.

Exposes the OpenEnv standard endpoints PLUS the competition-required
custom endpoints:  /tasks  /grader  /baseline

Endpoint summary
----------------
POST /reset          Reset the environment; optional body: {"seed": N, "task_name": "..."}
POST /step           Execute one action; body: {"action": {"action_type": "...", ...}}
GET  /state          Full internal environment state
GET  /schema         JSON schemas for Action / Observation / State
GET  /health         Liveness check
GET  /tasks          Available tasks and the action schema
GET  /grader         Score from the most recently completed episode
POST /baseline       Run oracle baseline on all 3 tasks and return scores
GET  /metadata       Environment description and metadata
WS   /ws             WebSocket for persistent sessions (concurrent agents)
"""

from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.types import (
        HealthResponse,
        HealthStatus,
        ResetRequest,
        SchemaResponse,
    )
except ImportError as exc:
    raise ImportError("openenv-core is required. pip install openenv-core") from exc

try:
    from ..models import DisasterAction, DisasterObservation, DisasterState
    from .config import TASK_CONFIGS
    from .environment import DisasterResponseEnvironment
except ImportError:
    from models import DisasterAction, DisasterObservation, DisasterState
    from server.config import TASK_CONFIGS
    from server.environment import DisasterResponseEnvironment

# ---------------------------------------------------------------------------
# Singleton environment — persists state across HTTP requests
# ---------------------------------------------------------------------------

_env_singleton = DisasterResponseEnvironment()


def _env_factory() -> DisasterResponseEnvironment:
    """Factory that always returns the same instance (stateful singleton)."""
    return _env_singleton


# ---------------------------------------------------------------------------
# Create base OpenEnv app (handles /reset /step /state /schema /health /ws)
# ---------------------------------------------------------------------------

app = create_app(
    _env_factory,
    DisasterAction,
    DisasterObservation,
    env_name="disaster_response",
    max_concurrent_envs=4,
)

# Override /state to return full DisasterState (base create_app only returns base State fields)
for _route in list(app.routes):
    if hasattr(_route, "path") and _route.path == "/state":
        app.routes.remove(_route)
        break

# Same override for /metadata — create_app registers one that only returns
# base EnvironmentMetadata fields; ours adds the competition-required task list.
for _route in list(app.routes):
    if hasattr(_route, "path") and _route.path == "/metadata":
        app.routes.remove(_route)
        break


@app.get(
    "/state",
    summary="Get current environment state",
    tags=["State Management"],
    response_model=None,
)
def get_full_state() -> Dict[str, Any]:
    """Return full internal DisasterState including task metrics and grader data."""
    return _env_singleton.state.model_dump()


# ---------------------------------------------------------------------------
# Custom competition-required endpoints
# ---------------------------------------------------------------------------


@app.get(
    "/tasks",
    summary="List available tasks and action schema",
    tags=["Competition"],
)
def get_tasks() -> Dict[str, Any]:
    """
    Returns all 3 tasks with their configuration and the action schema
    (fields required for an action in a /step call).
    """
    tasks = []
    for name, cfg in TASK_CONFIGS.items():
        tasks.append(
            {
                "task_name": name,
                "description": cfg["description"],
                "difficulty": cfg["difficulty"],
                "num_zones": cfg["num_zones"],
                "max_steps": cfg["max_steps"],
                "rescue_lock_steps": cfg["rescue_lock_steps"],
                "medical_lock_steps": cfg["medical_lock_steps"],
                "initial_rescue_teams": cfg["initial_rescue_teams"],
                "initial_medical_units": cfg["initial_medical_units"],
                "broadcast_credits": cfg["broadcast_credits"],
                "success_threshold": cfg["success_threshold"],
                "zones": [
                    {"id": z["id"], "name": z["name"]}
                    for z in cfg["zones"]
                ],
            }
        )

    action_schema = DisasterAction.model_json_schema()

    return {
        "tasks": tasks,
        "action_schema": action_schema,
        "usage": {
            "reset_with_task": "POST /reset with body: {\"task_name\": \"task1_flood_easy\"}",
            "step_example": (
                "POST /step with body: "
                "{\"action\": {\"action_type\": \"dispatch_rescue\", \"alert_id\": \"<id>\"}}"
            ),
            "available_action_types": [
                "dispatch_rescue",
                "dispatch_medical",
                "issue_evacuation",
                "request_more_info",
                "dismiss_false_alarm",
            ],
        },
    }


@app.get(
    "/grader",
    summary="Score from the most recently completed episode",
    tags=["Competition"],
)
def get_grader() -> Dict[str, Any]:
    """
    Returns the oracle-normalised score for the last completed episode.
    Score is in [0, 1].  Only populated after an episode ends (done=True).
    """
    data = getattr(DisasterResponseEnvironment, "_last_grader_data", {})
    if not data or data.get("score") is None:
        return {
            "status": "no_episode_completed",
            "message": (
                "No episode has been completed yet. "
                "Run a full episode via /reset → /step until done=True."
            ),
        }
    return {
        "status": "ok",
        "task": data["task"],
        "normalized_score": data["score"],
        "agent_reward": data["agent_reward"],
        "oracle_reward": data["oracle_reward"],
        "steps_completed": data["steps_completed"],
        "metrics": data.get("metrics", {}),
        "interpretation": {
            "1.0": "Agent matched or exceeded oracle (human-expert) performance",
            "0.75": "Strong performance — approaches expert-level triage",
            "0.50": "Moderate performance — basic triage learned",
            "0.25": "Weak performance — mostly random or greedy",
            "0.0": "Failed — likely worse than random",
        },
    }


class BaselineRequest(BaseModel):
    """Optional request body for /baseline."""
    seed: int = Field(default=42, description="Random seed for baseline runs")
    num_seeds: int = Field(default=3, ge=1, le=10, description="Seeds to average over")


@app.post(
    "/baseline",
    summary="Run oracle baseline on all 3 tasks",
    tags=["Competition"],
)
def run_baseline(request: BaselineRequest = Body(default_factory=BaselineRequest)) -> Dict[str, Any]:
    """
    Runs the oracle heuristic policy on all 3 tasks and returns baseline scores.

    The oracle uses only observable information (same as the agent) and
    represents expert-level heuristic performance. By definition, normalised
    oracle score = 1.0 on each task it was calibrated against.

    Scores can vary across seeds due to stochastic alert generation.
    """
    import math
    import random
    from server.config import TASK_CONFIGS
    from server.oracle import _oracle_create_alert, oracle_decide

    results = []

    for task_name, cfg in TASK_CONFIGS.items():
        seed_scores = []

        for seed_offset in range(request.num_seeds):
            seed = request.seed + seed_offset
            # Use a fresh env instance to run the oracle
            env_instance = DisasterResponseEnvironment()
            env_instance.reset(seed=seed, task_name=task_name)
            oracle_total = env_instance._run_oracle(seed)

            seed_scores.append(oracle_total)

        avg_oracle = sum(seed_scores) / len(seed_scores)

        results.append(
            {
                "task": task_name,
                "difficulty": cfg["difficulty"],
                "oracle_reward_avg": round(avg_oracle, 4),
                "normalized_score": 1.0,  # oracle is the reference
                "success_threshold": cfg["success_threshold"],
                "seeds_used": [request.seed + i for i in range(request.num_seeds)],
                "note": (
                    "Oracle score is the normalisation baseline. "
                    "An RL/LLM agent is scored as agent_reward / oracle_reward."
                ),
            }
        )

    return {
        "baseline_agent": "oracle_heuristic",
        "description": (
            "Rule-based oracle using only observable signals (same information as agent). "
            "Represents calibrated human-expert level performance."
        ),
        "results": results,
    }


@app.get(
    "/metadata",
    summary="Environment description and metadata",
    tags=["Competition"],
)
def get_metadata() -> Dict[str, Any]:
    """Returns high-level metadata about this OpenEnv environment."""
    return {
        "name": "disaster_response",
        "display_name": "DisasterResponseEnv",
        "version": "1.0.0",
        "description": (
            "Emergency Operations Center (EOC) triage agent. "
            "The agent coordinates rescue resources across multiple disaster zones, "
            "triaging incoming alerts (sensor / radio / SMS / social-media) while "
            "managing finite resources, misinformation, and zone stress cascades."
        ),
        "domain": "emergency_response",
        "tasks": list(TASK_CONFIGS.keys()),
        "action_space": {
            "type": "discrete",
            "n": 5,
            "actions": [
                "dispatch_rescue",
                "dispatch_medical",
                "issue_evacuation",
                "request_more_info",
                "dismiss_false_alarm",
            ],
        },
        "observation_space": {
            "current_alert": "dict — source, severity, message, zone_id, arrival_step",
            "zones": "list of zone stress levels [0.0–1.0]",
            "resources": "rescue teams, medical units, broadcast credits",
            "step": "current episode step",
        },
        "reward_range": [-1.0, 1.0],
        "tags": ["openenv", "disaster-response", "emergency", "rl", "multi-zone"],
    }


# ---------------------------------------------------------------------------
# Root landing page
# ---------------------------------------------------------------------------


@app.get("/", summary="Landing page", tags=["Info"], response_class=HTMLResponse)
def root() -> HTMLResponse:
    """HTML landing page — environment overview, task cards, endpoint reference."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DisasterResponseEnv</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Segoe UI',system-ui,sans-serif;background:#0d1117;color:#e6edf3;min-height:100vh}
  a{color:#58a6ff;text-decoration:none}
  a:hover{text-decoration:underline}
  .topbar{background:#161b22;border-bottom:1px solid #30363d;padding:14px 32px;display:flex;align-items:center;justify-content:space-between}
  .topbar h1{font-size:1.15rem;font-weight:700;letter-spacing:.5px}
  .topbar h1 span{color:#f85149}
  .badge{font-size:.7rem;background:#238636;color:#fff;padding:2px 8px;border-radius:12px;font-weight:600;margin-left:10px}
  .topnav a{font-size:.85rem;margin-left:20px;color:#8b949e}
  .topnav a:hover{color:#e6edf3}
  .hero{padding:56px 32px 40px;text-align:center;background:linear-gradient(180deg,#161b22 0%,#0d1117 100%);border-bottom:1px solid #21262d}
  .hero h2{font-size:2rem;font-weight:700;margin-bottom:12px}
  .hero h2 em{color:#f85149;font-style:normal}
  .hero p{color:#8b949e;max-width:640px;margin:0 auto 28px;line-height:1.7;font-size:.95rem}
  .hero-btns a{display:inline-block;padding:9px 22px;border-radius:6px;font-size:.875rem;font-weight:600;margin:0 6px}
  .btn-primary{background:#238636;color:#fff;border:1px solid #2ea043}
  .btn-primary:hover{background:#2ea043;text-decoration:none}
  .btn-secondary{background:transparent;color:#58a6ff;border:1px solid #30363d}
  .btn-secondary:hover{background:#21262d;text-decoration:none}
  .section{padding:40px 32px;max-width:1100px;margin:0 auto}
  .section-title{font-size:1rem;font-weight:700;color:#8b949e;text-transform:uppercase;letter-spacing:1px;margin-bottom:20px;padding-bottom:8px;border-bottom:1px solid #21262d}
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px;margin-bottom:40px}
  .card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:20px}
  .card-title{font-weight:700;margin-bottom:6px;display:flex;align-items:center;gap:8px}
  .card-body{color:#8b949e;font-size:.875rem;line-height:1.6}
  .pill{font-size:.65rem;padding:2px 8px;border-radius:10px;font-weight:700;text-transform:uppercase}
  .easy{background:#1a4731;color:#3fb950}
  .medium{background:#3d2e00;color:#d29922}
  .hard{background:#3d0e11;color:#f85149}
  table{width:100%;border-collapse:collapse;font-size:.85rem}
  th{text-align:left;padding:10px 14px;background:#161b22;color:#8b949e;font-weight:600;border-bottom:1px solid #21262d}
  td{padding:10px 14px;border-bottom:1px solid #21262d;color:#c9d1d9}
  td:first-child{font-family:monospace;color:#58a6ff;white-space:nowrap}
  tr:last-child td{border-bottom:none}
  .method{display:inline-block;font-size:.7rem;font-weight:700;padding:1px 6px;border-radius:4px;margin-right:6px}
  .get{background:#1a3a2a;color:#3fb950}
  .post{background:#2d2a1e;color:#d29922}
  .reward-table td:first-child{color:#e6edf3;font-family:inherit}
  .reward-table td:nth-child(2){font-family:monospace;color:#79c0ff}
  pre{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:16px;font-size:.8rem;color:#79c0ff;overflow-x:auto;line-height:1.6}
  .footer{text-align:center;padding:24px;color:#484f58;font-size:.8rem;border-top:1px solid #21262d;margin-top:40px}
  .stat-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:40px}
  .stat{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;text-align:center}
  .stat-val{font-size:1.6rem;font-weight:700;color:#58a6ff}
  .stat-lbl{font-size:.75rem;color:#8b949e;margin-top:4px}
</style>
</head>
<body>

<div class="topbar">
  <h1>&#x1F6A8; Disaster<span>Response</span>Env <span class="badge">v1.0.0</span></h1>
  <nav class="topnav">
    <a href="/dashboard">Live Dashboard</a>
    <a href="/docs">API Docs</a>
    <a href="/tasks">Tasks JSON</a>
    <a href="/health">Health</a>
  </nav>
</div>

<div class="hero">
  <h2>Emergency Operations Center<br><em>Triage Agent</em> Environment</h2>
  <p>An OpenEnv reinforcement-learning environment where an AI agent coordinates disaster response
  across multiple geographic zones — triaging real vs. fake alerts, dispatching finite rescue
  resources, and preventing zone stress cascades under strict time pressure.</p>
  <div class="hero-btns">
    <a href="/dashboard" class="btn-primary">&#x25B6; Open Live Dashboard</a>
    <a href="/docs" class="btn-secondary">API Reference</a>
  </div>
</div>

<div class="section">
  <div class="stat-row">
    <div class="stat"><div class="stat-val">3</div><div class="stat-lbl">Tasks</div></div>
    <div class="stat"><div class="stat-val">5</div><div class="stat-lbl">Actions</div></div>
    <div class="stat"><div class="stat-val">±1.0</div><div class="stat-lbl">Reward Range</div></div>
    <div class="stat"><div class="stat-val">130</div><div class="stat-lbl">Max Steps (hard)</div></div>
    <div class="stat"><div class="stat-val">20%</div><div class="stat-lbl">Spoof Rate (hard)</div></div>
  </div>

  <div class="section-title">Tasks</div>
  <div class="cards">
    <div class="card">
      <div class="card-title"><span class="pill easy">Easy</span> task1_flood_easy</div>
      <div class="card-body">Single-zone flash flood. ~75% real alerts, 3 rescue teams, 2-step locks, 30 steps. Learn the core real-vs-noise signal.</div>
    </div>
    <div class="card">
      <div class="card-title"><span class="pill medium">Medium</span> task2_multizone_medium</div>
      <div class="card-body">Three-zone flood. Resources shared across zones with 5/4-step locks, 55 steps. Spatial prioritisation becomes critical.</div>
    </div>
    <div class="card">
      <div class="card-title"><span class="pill hard">Hard</span> task3_compound_hard</div>
      <div class="card-body">Five-zone earthquake + flood. 20% spoofed alerts, 8/6-step locks, cascade stress propagation, 130 steps. Requires long-horizon strategy.</div>
    </div>
  </div>

  <div class="section-title">Actions</div>
  <table style="margin-bottom:40px">
    <tr><th>Action</th><th>Effect</th><th>Reward</th></tr>
    <tr><td>dispatch_rescue</td><td>Deploy rescue team — locks resource for K steps</td><td>+1.0 × time_decay (real) / −0.30 (noise)</td></tr>
    <tr><td>dispatch_medical</td><td>Deploy medical unit — locks resource for K steps</td><td>+0.70 × time_decay (real) / −0.25 (noise)</td></tr>
    <tr><td>issue_evacuation</td><td>Broadcast evacuation for a high-stress zone</td><td>+0.40 (stress ≥ 0.65) / −0.15 (low stress)</td></tr>
    <tr><td>request_more_info</td><td>Wait one step — same alert stays active</td><td>−0.05 always</td></tr>
    <tr><td>dismiss_false_alarm</td><td>Close alert as noise</td><td>+0.20 (correct) / −1.00 (missed real victim)</td></tr>
  </table>

  <div class="section-title">API Endpoints</div>
  <table style="margin-bottom:40px">
    <tr><th>Endpoint</th><th>Purpose</th></tr>
    <tr><td><span class="method post">POST</span>/reset</td><td>Start new episode — body: {"task_name": "...", "seed": 42}</td></tr>
    <tr><td><span class="method post">POST</span>/step</td><td>Take one action — body: {"action": {"action_type": "...", "alert_id": "..."}}</td></tr>
    <tr><td><span class="method get">GET</span>/state</td><td>Full environment state with zone stress, resources, and metrics</td></tr>
    <tr><td><span class="method get">GET</span>/tasks</td><td>All tasks with configuration and action JSON schema</td></tr>
    <tr><td><span class="method get">GET</span>/grader</td><td>Normalized score [0–1] from the last completed episode</td></tr>
    <tr><td><span class="method post">POST</span>/baseline</td><td>Run oracle baseline across all 3 tasks</td></tr>
    <tr><td><span class="method get">GET</span>/schema</td><td>JSON schemas for Action / Observation / State</td></tr>
    <tr><td><span class="method get">GET</span>/health</td><td>Liveness check</td></tr>
    <tr><td><span class="method get">GET</span>/docs</td><td>Interactive Swagger UI</td></tr>
  </table>

  <div class="section-title">Quickstart</div>
  <pre># 1. Start an episode
curl -X POST http://localhost:8000/reset \\
     -H "Content-Type: application/json" \\
     -d '{"task_name": "task1_flood_easy", "seed": 42}'

# 2. Take an action (use alert_id from the reset response)
curl -X POST http://localhost:8000/step \\
     -H "Content-Type: application/json" \\
     -d '{"action": {"action_type": "dispatch_rescue", "alert_id": "&lt;id&gt;"}}'

# 3. Repeat until done=true, then check your score
curl http://localhost:8000/grader</pre>
</div>

<div class="footer">DisasterResponseEnv v1.0.0 &mdash; OpenEnv Hackathon Submission &mdash; <a href="/docs">Swagger UI</a> &mdash; <a href="/dashboard">Live Dashboard</a></div>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.get("/dashboard", summary="Interactive EOC dashboard", tags=["Info"], response_class=HTMLResponse)
def web_dashboard() -> HTMLResponse:
    """Interactive EOC dashboard — run episodes in the browser with live state updates."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EOC Dashboard — DisasterResponseEnv</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Segoe UI',system-ui,sans-serif;background:#0d1117;color:#e6edf3;height:100vh;display:flex;flex-direction:column;overflow:hidden}
  a{color:#58a6ff;text-decoration:none}
  .topbar{background:#161b22;border-bottom:1px solid #30363d;padding:10px 20px;display:flex;align-items:center;justify-content:space-between;flex-shrink:0}
  .topbar h1{font-size:1rem;font-weight:700}
  .topbar h1 span{color:#f85149}
  .topnav a{font-size:.8rem;margin-left:16px;color:#8b949e}
  .topnav a:hover{color:#e6edf3}
  .main{display:grid;grid-template-columns:260px 1fr 260px;flex:1;overflow:hidden;min-height:0}
  .panel{background:#161b22;border:1px solid #21262d;overflow-y:auto}
  .panel-title{font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#8b949e;padding:10px 14px;background:#0d1117;border-bottom:1px solid #21262d;position:sticky;top:0;z-index:1}
  /* Left — controls */
  .left{border-right:1px solid #30363d;overflow-y:auto;min-height:0}
  .ctrl-section{padding:12px 14px;border-bottom:1px solid #21262d}
  .ctrl-label{font-size:.7rem;color:#8b949e;margin-bottom:5px;display:block;font-weight:600;text-transform:uppercase;letter-spacing:.5px}
  select,input[type=number]{width:100%;background:#0d1117;border:1px solid #30363d;color:#e6edf3;padding:7px 10px;border-radius:5px;font-size:.82rem;outline:none}
  select:focus,input:focus{border-color:#58a6ff}
  .btn{width:100%;padding:9px;border-radius:5px;font-size:.82rem;font-weight:700;cursor:pointer;border:none;margin-top:8px;transition:.15s}
  .btn-red{background:#da3633;color:#fff}.btn-red:hover{background:#f85149}
  .btn-rescue{background:#1f6feb;color:#fff}.btn-rescue:hover{background:#388bfd}
  .btn-medical{background:#1a7f37;color:#fff}.btn-medical:hover{background:#2ea043}
  .btn-evac{background:#9a6700;color:#fff}.btn-evac:hover{background:#d29922}
  .btn-info{background:#30363d;color:#e6edf3}.btn-info:hover{background:#484f58}
  .btn-dismiss{background:#3d0e11;color:#f85149;border:1px solid #f85149}.btn-dismiss:hover{background:#f85149;color:#fff}
  .btn:disabled{opacity:.35;cursor:not-allowed}
  .btn-auto{background:#6e40c9;color:#fff}.btn-auto:hover{background:#8957e5}
  .btn-stop{background:#484f58;color:#fff}.btn-stop:hover{background:#6e7681}
  .speed-row{display:flex;align-items:center;gap:8px;margin-top:8px}
  .speed-row label{font-size:.7rem;color:#8b949e;white-space:nowrap}
  input[type=range]{flex:1;accent-color:#6e40c9;height:4px}
  .speed-val{font-size:.7rem;color:#e6edf3;min-width:32px;text-align:right}
  .auto-divider{border:none;border-top:1px solid #21262d;margin:12px 0}
  .policy-row{display:flex;gap:6px;margin-top:8px}
  .policy-btn{flex:1;padding:5px 4px;font-size:.68rem;border-radius:4px;cursor:pointer;border:1px solid #30363d;background:#0d1117;color:#8b949e;font-weight:600;transition:.15s}
  .policy-btn.active{border-color:#6e40c9;background:#2d1f56;color:#bc8cff}
  .policy-btn:hover:not(.active){background:#161b22;color:#e6edf3}
  /* Center — alert */
  .center-col{display:flex;flex-direction:column;overflow:hidden;min-height:0}
  .center-top{flex:0 0 auto;max-height:220px;overflow-y:auto;border-bottom:1px solid #30363d}
  .center-bottom{flex:1;overflow-y:auto;min-height:0}
  .alert-card{padding:14px 16px}
  .alert-id{font-family:monospace;font-size:.72rem;color:#484f58;margin-bottom:4px}
  .alert-src{display:inline-block;font-size:.68rem;font-weight:700;text-transform:uppercase;padding:2px 7px;border-radius:10px;margin-bottom:8px}
  .src-sensor{background:#1a3a2a;color:#3fb950}
  .src-radio{background:#1f3a5f;color:#79c0ff}
  .src-sms{background:#3d2e00;color:#d29922}
  .src-social_media{background:#2e1a3a;color:#bc8cff}
  .sev-bar-wrap{margin:8px 0;background:#21262d;border-radius:4px;height:8px;overflow:hidden}
  .sev-bar{height:8px;border-radius:4px;transition:.3s}
  .alert-msg{font-size:.82rem;color:#c9d1d9;line-height:1.5;margin-top:8px;padding:8px 10px;background:#0d1117;border-radius:5px;border-left:3px solid #30363d}
  .alert-meta{display:flex;gap:16px;margin-top:8px;font-size:.75rem;color:#8b949e}
  .no-alert{padding:20px 16px;color:#484f58;font-size:.85rem;text-align:center}
  /* Log */
  .log{padding:8px 14px}
  .log-entry{font-size:.75rem;padding:5px 8px;border-radius:4px;margin-bottom:4px;display:flex;gap:8px;align-items:flex-start;font-family:monospace}
  .log-step{color:#484f58;flex-shrink:0;min-width:50px}
  .log-action{flex:1;color:#c9d1d9}
  .log-reward{flex-shrink:0;font-weight:700}
  .pos{color:#3fb950}.neg{color:#f85149}.neu{color:#8b949e}
  .log-info{background:#0d1117;border:1px solid #21262d}
  /* Right — state */
  .right{border-left:1px solid #30363d;overflow-y:auto;min-height:0}
  .zone-item{padding:10px 14px;border-bottom:1px solid #21262d}
  .zone-name{font-size:.8rem;font-weight:600;margin-bottom:5px;display:flex;justify-content:space-between}
  .zone-stress-wrap{background:#21262d;border-radius:3px;height:6px;overflow:hidden}
  .zone-stress{height:6px;border-radius:3px;transition:.4s}
  .zone-meta{font-size:.7rem;color:#8b949e;margin-top:4px}
  .res-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:10px 14px}
  .res-item{background:#0d1117;border:1px solid #21262d;border-radius:6px;padding:8px 10px;text-align:center}
  .res-val{font-size:1.3rem;font-weight:700;color:#58a6ff}
  .res-lbl{font-size:.68rem;color:#8b949e;margin-top:2px}
  /* Score banner */
  .score-banner{display:none;padding:12px 16px;text-align:center;font-size:.9rem;font-weight:700;border-radius:6px;margin:10px 14px}
  .score-pass{background:#1a4731;color:#3fb950;border:1px solid #2ea043}
  .score-fail{background:#3d0e11;color:#f85149;border:1px solid #f85149}
  /* Status bar */
  .statusbar{background:#161b22;border-top:1px solid #30363d;padding:5px 16px;font-size:.72rem;color:#8b949e;display:flex;gap:20px;flex-shrink:0}
  .statusbar span b{color:#e6edf3}
</style>
</head>
<body>

<div class="topbar">
  <h1>&#x1F6A8; Disaster<span>Response</span>Env — EOC Dashboard</h1>
  <nav class="topnav">
    <a href="/">Home</a>
    <a href="/docs">API Docs</a>
    <a href="/grader">Grader</a>
    <a href="/state">State JSON</a>
  </nav>
</div>

<div class="main">

  <!-- LEFT: Controls -->
  <div class="panel left">
    <div class="panel-title">&#x2699; Episode Controls</div>
    <div class="ctrl-section">
      <label class="ctrl-label">Task</label>
      <select id="sel-task">
        <option value="task1_flood_easy">task1 — Flood Easy</option>
        <option value="task2_multizone_medium">task2 — Multizone Medium</option>
        <option value="task3_compound_hard">task3 — Compound Hard</option>
      </select>
      <label class="ctrl-label" style="margin-top:10px">Seed</label>
      <input type="number" id="inp-seed" value="42" min="0">
      <button class="btn btn-red" id="btn-reset">&#x25B6; New Episode</button>
    </div>

    <div class="ctrl-section">
      <label class="ctrl-label">&#x25B6; Auto Run</label>
      <div class="policy-row">
        <button class="policy-btn active" id="pol-heuristic">Heuristic</button>
        <button class="policy-btn" id="pol-aggressive">Aggressive</button>
        <button class="policy-btn" id="pol-cautious">Cautious</button>
      </div>
      <div class="speed-row">
        <label>Speed</label>
        <input type="range" id="inp-speed" min="100" max="2000" step="100" value="600">
        <span class="speed-val" id="lbl-speed">0.6s</span>
      </div>
      <div style="display:flex;gap:6px">
        <button class="btn btn-auto" id="btn-autorun" style="flex:1" disabled>&#x25B6;&#x25B6; Auto Run</button>
        <button class="btn btn-stop" id="btn-stop" style="flex:0 0 52px;margin-top:8px" disabled>&#x23F9;</button>
      </div>
      <hr class="auto-divider">
      <label class="ctrl-label">Manual Action</label>
      <button class="btn btn-rescue" id="btn-rescue" disabled>&#x1F691; Dispatch Rescue</button>
      <button class="btn btn-medical" id="btn-medical" disabled>&#x1FA7A; Dispatch Medical</button>
      <button class="btn btn-evac" id="btn-evac" disabled>&#x1F4E2; Issue Evacuation</button>
      <button class="btn btn-info" id="btn-info" disabled>&#x1F50D; Request More Info</button>
      <button class="btn btn-dismiss" id="btn-dismiss" disabled>&#x274C; Dismiss False Alarm</button>
    </div>

    <div class="ctrl-section">
      <div id="score-banner" class="score-banner"></div>
      <div style="font-size:.72rem;color:#8b949e;line-height:1.6;margin-top:4px">
        <b style="color:#e6edf3">Reward design:</b><br>
        Correct rescue: <span style="color:#3fb950">+1.0 × e<sup>−0.07×delay</sup></span><br>
        False dispatch: <span style="color:#f85149">−0.30</span><br>
        Miss real victim: <span style="color:#f85149">−1.00</span><br>
        Correct dismiss: <span style="color:#3fb950">+0.20</span><br>
        Deliberate: <span style="color:#8b949e">−0.05</span>
      </div>
    </div>
  </div>

  <!-- CENTER: Alert (capped height) + scrollable log -->
  <div class="center-col">
    <div class="panel center-top">
      <div class="panel-title">&#x26A0; Current Alert</div>
      <div id="alert-area">
        <div class="no-alert">Start an episode to see alerts.</div>
      </div>
    </div>
    <div class="panel center-bottom">
      <div class="panel-title">&#x1F4CB; Step Log</div>
      <div id="log" class="log">
        <div class="log-entry log-info"><span class="log-step">—</span><span class="log-action" style="color:#484f58">No episode running.</span></div>
      </div>
    </div>
  </div>

  <!-- RIGHT: Zone + Resources -->
  <div class="panel right">
    <div class="panel-title">&#x1F5FA; Zone Status</div>
    <div id="zones"><div style="padding:16px;color:#484f58;font-size:.82rem">Waiting for episode…</div></div>

    <div class="panel-title" style="margin-top:0">&#x1F4E6; Resources</div>
    <div id="resources"><div style="padding:16px;color:#484f58;font-size:.82rem">Waiting…</div></div>
  </div>

</div>

<div class="statusbar">
  <span>Task: <b id="st-task">—</b></span>
  <span>Step: <b id="st-step">—</b></span>
  <span>Cumulative Reward: <b id="st-reward">—</b></span>
  <span>Alerts Processed: <b id="st-alerts">—</b></span>
  <span>Status: <b id="st-status" style="color:#3fb950">Ready</b></span>
</div>

<script>
const API = '';
let currentAlertId = null;
let episodeActive = false;
let stepNum = 0;

const actions = {
  'btn-rescue':  'dispatch_rescue',
  'btn-medical': 'dispatch_medical',
  'btn-evac':    'issue_evacuation',
  'btn-info':    'request_more_info',
  'btn-dismiss': 'dismiss_false_alarm',
};

function setStatus(msg, color='#8b949e') {
  const el = document.getElementById('st-status');
  el.textContent = msg; el.style.color = color;
}

function setActionBtns(enabled) {
  Object.keys(actions).forEach(id => {
    document.getElementById(id).disabled = !enabled;
  });
}

function sevColor(s) {
  if (s >= 0.80) return '#f85149';
  if (s >= 0.55) return '#d29922';
  return '#3fb950';
}

function stressColor(s) {
  if (s >= 0.75) return '#f85149';
  if (s >= 0.50) return '#d29922';
  return '#3fb950';
}

function renderAlert(alert) {
  if (!alert) { document.getElementById('alert-area').innerHTML = '<div class="no-alert">No active alert.</div>'; return; }
  const sev = alert.severity;
  const srcClass = 'src-' + alert.source;
  document.getElementById('alert-area').innerHTML = `
    <div class="alert-card">
      <div class="alert-id">ID: ${alert.alert_id} &nbsp;|&nbsp; Zone: ${alert.zone_name} (${alert.zone_id})</div>
      <span class="alert-src ${srcClass}">${alert.source}</span>
      <div style="display:flex;align-items:center;gap:10px">
        <div class="sev-bar-wrap" style="flex:1"><div class="sev-bar" style="width:${Math.round(sev*100)}%;background:${sevColor(sev)}"></div></div>
        <span style="font-size:.85rem;font-weight:700;color:${sevColor(sev)}">${sev.toFixed(3)}</span>
      </div>
      <div class="alert-msg">${alert.message}</div>
      <div class="alert-meta">
        <span>Arrived: step ${alert.arrival_step}</span>
        <span>Deliberated: ${alert.deliberation_count}×</span>
        <span>Waiting: ${stepNum - alert.arrival_step} steps</span>
      </div>
    </div>`;
  currentAlertId = alert.alert_id;
}

function renderZones(zones) {
  if (!zones || !zones.length) return;
  document.getElementById('zones').innerHTML = zones.map(z => `
    <div class="zone-item">
      <div class="zone-name">
        <span>${z.name}</span>
        <span style="color:${stressColor(z.stress)};font-size:.75rem">${(z.stress*100).toFixed(1)}%</span>
      </div>
      <div class="zone-stress-wrap"><div class="zone-stress" style="width:${Math.round(z.stress*100)}%;background:${stressColor(z.stress)}"></div></div>
      <div class="zone-meta">${z.pending_alerts} pending alert${z.pending_alerts!==1?'s':''}</div>
    </div>`).join('');
}

function renderResources(res) {
  if (!res) return;
  const locked_r = (res.rescue_teams_locked||[]).length;
  const locked_m = (res.medical_units_locked||[]).length;
  document.getElementById('resources').innerHTML = `
    <div class="res-grid">
      <div class="res-item"><div class="res-val" style="color:#1f6feb">${res.rescue_teams_available}</div><div class="res-lbl">Rescue Free</div></div>
      <div class="res-item"><div class="res-val" style="color:#484f58">${locked_r}</div><div class="res-lbl">Rescue Locked</div></div>
      <div class="res-item"><div class="res-val" style="color:#1a7f37">${res.medical_units_available}</div><div class="res-lbl">Medical Free</div></div>
      <div class="res-item"><div class="res-val" style="color:#484f58">${locked_m}</div><div class="res-lbl">Medical Locked</div></div>
      <div class="res-item" style="grid-column:1/-1"><div class="res-val" style="color:#9a6700">${res.broadcast_credits}</div><div class="res-lbl">Broadcast Credits</div></div>
    </div>`;
}

function addLog(step, action, reward, extra='') {
  const cls = reward > 0 ? 'pos' : reward < 0 ? 'neg' : 'neu';
  const sign = reward > 0 ? '+' : '';
  const log = document.getElementById('log');
  const entry = document.createElement('div');
  entry.className = 'log-entry';
  entry.innerHTML = `<span class="log-step">step ${step}</span><span class="log-action">${action}${extra ? ' <span style="color:#484f58">'+extra+'</span>' : ''}</span><span class="log-reward ${cls}">${sign}${reward.toFixed(2)}</span>`;
  log.insertBefore(entry, log.firstChild);
}

function updateStatus(obs) {
  document.getElementById('st-step').textContent = `${obs.step || stepNum} / ${obs.max_steps || '?'}`;
  document.getElementById('st-reward').textContent = (obs.cumulative_reward||0).toFixed(3);
  document.getElementById('st-task').textContent = obs.task_name || '—';
  document.getElementById('st-alerts').textContent = '—';
}

async function doReset() {
  const task = document.getElementById('sel-task').value;
  const seed = parseInt(document.getElementById('inp-seed').value)||42;
  setStatus('Resetting…', '#d29922');
  document.getElementById('score-banner').style.display = 'none';
  document.getElementById('log').innerHTML = '';
  stepNum = 0;
  try {
    const r = await fetch(API+'/reset', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({task_name:task, seed})});
    const d = await r.json();
    const obs = d.observation || d;
    stepNum = obs.step || 1;
    episodeActive = true;
    renderAlert(obs.current_alert);
    renderZones(obs.zones);
    renderResources(obs.resources);
    updateStatus(obs);
    setActionBtns(true);
    setStatus('Running', '#3fb950');
    addLog('—', 'Episode started', 0, `task=${task} seed=${seed}`);
  } catch(e) { setStatus('Error: '+e.message, '#f85149'); }
}

async function doStep(actionType) {
  if (!episodeActive) return;
  setActionBtns(false);
  setStatus('Stepping…', '#d29922');
  try {
    const r = await fetch(API+'/step', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({action:{action_type:actionType, alert_id:currentAlertId}})});
    const d = await r.json();
    const obs = d.observation || d;
    const reward = d.reward ?? 0;
    const done = d.done ?? false;
    stepNum = obs.step || stepNum+1;
    renderAlert(obs.current_alert);
    renderZones(obs.zones);
    renderResources(obs.resources);
    updateStatus(obs);
    addLog(stepNum-1, actionType, reward, obs.last_action_result||'');
    if (done) {
      episodeActive = false;
      setStatus('Episode complete', '#58a6ff');
      // fetch grader
      const gr = await fetch(API+'/grader');
      const gd = await gr.json();
      const score = gd.normalized_score ?? 0;
      const passed = gd.metrics?.passed;
      const banner = document.getElementById('score-banner');
      banner.className = 'score-banner ' + (passed ? 'score-pass' : 'score-fail');
      banner.textContent = (passed ? '✓ PASSED' : '✗ FAILED') + ' — Score: ' + score.toFixed(4) + ' / 1.0';
      banner.style.display = 'block';
    } else {
      setActionBtns(true);
      setStatus('Running', '#3fb950');
    }
  } catch(e) { setStatus('Error: '+e.message, '#f85149'); setActionBtns(true); }
}

// ---------- Auto-run ----------
let autoTimer = null;
let activePolicy = 'heuristic';

const policies = {
  heuristic(alert) {
    const {severity: sev, source: src, deliberation_count: dc} = alert;
    if (sev < 0.30) return 'dismiss_false_alarm';
    if (sev >= 0.80 && (src==='sensor'||src==='radio')) return 'dispatch_rescue';
    if (sev >= 0.70) return 'dispatch_medical';
    if (dc===0 && (src==='sms'||src==='social_media')) return 'request_more_info';
    if (sev >= 0.50) return 'dispatch_rescue';
    return 'dismiss_false_alarm';
  },
  aggressive(alert) {
    // Dispatch everything with severity > 0.35 — high recall, wastes resources
    const {severity: sev} = alert;
    if (sev < 0.25) return 'dismiss_false_alarm';
    if (sev >= 0.60) return 'dispatch_rescue';
    return 'dispatch_medical';
  },
  cautious(alert) {
    // Deliberate once on anything uncertain — low waste, misses urgent alerts
    const {severity: sev, source: src, deliberation_count: dc} = alert;
    if (sev < 0.40) return 'dismiss_false_alarm';
    if (dc === 0) return 'request_more_info';
    if (sev >= 0.70) return 'dispatch_rescue';
    return 'dispatch_medical';
  }
};

function getSpeed() {
  return parseInt(document.getElementById('inp-speed').value) || 600;
}

function startAutoRun() {
  if (!episodeActive || autoTimer) return;
  document.getElementById('btn-autorun').disabled = true;
  document.getElementById('btn-stop').disabled = false;
  setActionBtns(false);
  setStatus('Auto running…', '#bc8cff');

  async function tick() {
    if (!episodeActive) { stopAutoRun(); return; }
    const alertEl = document.querySelector('.alert-card');
    if (!alertEl || !currentAlertId) { stopAutoRun(); return; }
    // Read current alert data from state to feed the policy
    const stateR = await fetch(API+'/state');
    const state = await stateR.json();
    const alert = state.current_alert;
    if (!alert) { stopAutoRun(); return; }
    const at = policies[activePolicy](alert);
    await doStep(at);
    if (episodeActive) {
      autoTimer = setTimeout(tick, getSpeed());
    } else {
      stopAutoRun();
    }
  }
  autoTimer = setTimeout(tick, 200);
}

function stopAutoRun() {
  if (autoTimer) { clearTimeout(autoTimer); autoTimer = null; }
  document.getElementById('btn-autorun').disabled = !episodeActive;
  document.getElementById('btn-stop').disabled = true;
  if (episodeActive) {
    setActionBtns(true);
    setStatus('Running', '#3fb950');
  }
}

// Policy selector
['heuristic','aggressive','cautious'].forEach(p => {
  document.getElementById('pol-'+p).addEventListener('click', () => {
    activePolicy = p;
    document.querySelectorAll('.policy-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('pol-'+p).classList.add('active');
  });
});

// Speed label
document.getElementById('inp-speed').addEventListener('input', e => {
  const ms = parseInt(e.target.value);
  document.getElementById('lbl-speed').textContent = ms >= 1000 ? (ms/1000).toFixed(1)+'s' : ms+'ms';
});

document.getElementById('btn-autorun').addEventListener('click', startAutoRun);
document.getElementById('btn-stop').addEventListener('click', stopAutoRun);

// Enable auto button after reset (patch doReset to also enable btn-autorun)
const _origReset = doReset;
// eslint-disable-next-line no-global-assign
window._patchedReset = async function() {
  stopAutoRun();
  await _origReset();
  document.getElementById('btn-autorun').disabled = false;
};
document.getElementById('btn-reset').removeEventListener('click', doReset);
document.getElementById('btn-reset').addEventListener('click', window._patchedReset);

// ---------- Manual actions ----------
Object.entries(actions).forEach(([id, at]) => {
  document.getElementById(id).addEventListener('click', () => { stopAutoRun(); doStep(at); });
});
</script>
</body>
</html>"""
    return HTMLResponse(content=html)


# ---------------------------------------------------------------------------
# Entry point for direct execution
# ---------------------------------------------------------------------------

def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
