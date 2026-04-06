"""
Integration tests for DisasterResponseEnv HTTP API.

Requires a running server. Skipped automatically when ENV_BASE_URL is
unreachable, so the suite still passes in offline/CI environments.

Run against local server:
  uvicorn server.app:app --port 8000 &
  pytest tests/test_api.py -v

Run against live HF Space:
  ENV_BASE_URL=https://jishnu-vijayan-03-disaster-response.hf.space pytest tests/test_api.py -v
"""
from __future__ import annotations

import pytest
import requests

ALL_TASKS = ["task1_flood_easy", "task2_multizone_medium", "task3_compound_hard"]


# ---------------------------------------------------------------------------
# Fixtures / skip logic
# ---------------------------------------------------------------------------

def _server_available(base_url: str) -> bool:
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="session", autouse=True)
def require_server(base_url):
    if not _server_available(base_url):
        pytest.skip(f"Server not reachable at {base_url} — skipping API tests")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_200(self, base_url):
        r = requests.get(f"{base_url}/health", timeout=10)
        assert r.status_code == 200

    def test_health_payload(self, base_url):
        r = requests.get(f"{base_url}/health", timeout=10)
        data = r.json()
        assert "status" in data
        assert data["status"] == "healthy"


# ---------------------------------------------------------------------------
# Tasks endpoint
# ---------------------------------------------------------------------------

class TestTasks:
    def test_tasks_returns_200(self, base_url):
        r = requests.get(f"{base_url}/tasks", timeout=10)
        assert r.status_code == 200

    def test_tasks_lists_all_three(self, base_url):
        r = requests.get(f"{base_url}/tasks", timeout=10)
        data = r.json()
        task_names = [t.get("task_name") or t.get("name") if isinstance(t, dict) else t for t in data.get("tasks", data)]
        for expected in ALL_TASKS:
            assert expected in task_names, f"Task '{expected}' missing from /tasks response"


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    @pytest.mark.parametrize("task_name", ALL_TASKS)
    def test_reset_returns_200(self, base_url, task_name):
        r = requests.post(
            f"{base_url}/reset",
            json={"task_name": task_name, "seed": 42},
            timeout=30,
        )
        assert r.status_code == 200, f"/reset failed for {task_name}: {r.text}"

    @pytest.mark.parametrize("task_name", ALL_TASKS)
    def test_reset_observation_fields(self, base_url, task_name):
        r = requests.post(
            f"{base_url}/reset",
            json={"task_name": task_name, "seed": 42},
            timeout=30,
        )
        data = r.json()
        obs = data.get("observation", data)
        assert "current_alert" in obs, "Observation missing current_alert"
        assert "zones" in obs, "Observation missing zones"
        assert "resources" in obs, "Observation missing resources"
        assert "max_steps" in obs, "Observation missing max_steps"
        assert obs["max_steps"] > 0, "max_steps must be positive"

    @pytest.mark.parametrize("task_name", ALL_TASKS)
    def test_reset_is_real_hidden(self, base_url, task_name):
        r = requests.post(
            f"{base_url}/reset",
            json={"task_name": task_name, "seed": 42},
            timeout=30,
        )
        obs = r.json().get("observation", r.json())
        alert = obs.get("current_alert") or {}
        assert "is_real" not in alert, "is_real leaked into /reset response"
        assert "is_spoofed" not in alert, "is_spoofed leaked into /reset response"


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

class TestStep:
    def _reset_and_get_alert(self, base_url, task_name="task1_flood_easy"):
        r = requests.post(
            f"{base_url}/reset",
            json={"task_name": task_name, "seed": 42},
            timeout=30,
        )
        r.raise_for_status()
        obs = r.json().get("observation", r.json())
        alert_id = (obs.get("current_alert") or {}).get("alert_id", "")
        return alert_id

    def test_step_returns_200(self, base_url):
        alert_id = self._reset_and_get_alert(base_url)
        r = requests.post(
            f"{base_url}/step",
            json={"action": {"action_type": "dispatch_rescue", "alert_id": alert_id}},
            timeout=30,
        )
        assert r.status_code == 200, f"/step returned {r.status_code}: {r.text}"

    def test_step_response_fields(self, base_url):
        alert_id = self._reset_and_get_alert(base_url)
        r = requests.post(
            f"{base_url}/step",
            json={"action": {"action_type": "dispatch_rescue", "alert_id": alert_id}},
            timeout=30,
        )
        data = r.json()
        assert "reward" in data, "Step response missing reward"
        assert "done" in data, "Step response missing done"
        obs = data.get("observation", data)
        assert "step" in obs, "Step observation missing step counter"

    def test_step_reward_in_range(self, base_url):
        alert_id = self._reset_and_get_alert(base_url)
        r = requests.post(
            f"{base_url}/step",
            json={"action": {"action_type": "dispatch_rescue", "alert_id": alert_id}},
            timeout=30,
        )
        reward = r.json().get("reward", 0.0)
        assert -1.0 <= reward <= 1.0, f"Step reward {reward} outside [-1.0, 1.0]"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class TestState:
    def test_state_returns_200(self, base_url):
        requests.post(f"{base_url}/reset", json={"task_name": "task1_flood_easy", "seed": 42}, timeout=30)
        r = requests.get(f"{base_url}/state", timeout=10)
        assert r.status_code == 200

    def test_state_has_task_name(self, base_url):
        requests.post(f"{base_url}/reset", json={"task_name": "task1_flood_easy", "seed": 42}, timeout=30)
        r = requests.get(f"{base_url}/state", timeout=10)
        data = r.json()
        assert "task_name" in data, "/state missing task_name"


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

class TestGrader:
    def _run_full_episode(self, base_url, task_name="task1_flood_easy"):
        r = requests.post(
            f"{base_url}/reset",
            json={"task_name": task_name, "seed": 42},
            timeout=30,
        )
        r.raise_for_status()
        obs = r.json().get("observation", r.json())
        max_steps = obs.get("max_steps", 30)

        done = False
        for _ in range(max_steps):
            if done:
                break
            alert_id = (obs.get("current_alert") or {}).get("alert_id", "")
            r = requests.post(
                f"{base_url}/step",
                json={"action": {"action_type": "dispatch_rescue", "alert_id": alert_id}},
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            obs = data.get("observation", data)
            done = data.get("done", False)

    def test_grader_returns_200_after_episode(self, base_url):
        self._run_full_episode(base_url)
        r = requests.get(f"{base_url}/grader", timeout=10)
        assert r.status_code == 200

    def test_grader_score_in_range(self, base_url):
        self._run_full_episode(base_url)
        r = requests.get(f"{base_url}/grader", timeout=10)
        data = r.json()
        score = data.get("score") or data.get("normalized_score")
        if score is not None:
            assert 0.0 <= float(score) <= 1.0, f"/grader score {score} outside [0.0, 1.0]"


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

class TestBaseline:
    def test_baseline_returns_200(self, base_url):
        r = requests.post(
            f"{base_url}/baseline",
            json={"seed": 42, "num_seeds": 1},
            timeout=120,
        )
        assert r.status_code == 200, f"/baseline returned {r.status_code}: {r.text}"

    def test_baseline_scores_in_range(self, base_url):
        r = requests.post(
            f"{base_url}/baseline",
            json={"seed": 42, "num_seeds": 1},
            timeout=120,
        )
        data = r.json()
        scores = data.get("scores") or data.get("results") or {}
        if isinstance(scores, dict):
            for task, score in scores.items():
                if score is not None:
                    assert 0.0 <= float(score) <= 1.0, f"Baseline score for {task}={score} out of range"
        elif isinstance(scores, list):
            for entry in scores:
                score = entry.get("score") or entry.get("normalized_score")
                if score is not None:
                    assert 0.0 <= float(score) <= 1.0, f"Baseline score {score} out of range"
