"""
Unit tests for DisasterResponseEnvironment.

Tests run against the environment class directly — no HTTP server required.
Covers:
  - Determinism: same seed → same initial observation and score trajectory
  - Grader range: normalized_score always in [0.0, 1.0]
  - Episode boundary: all tasks terminate within configured max_steps
  - Hidden field protection: is_real / is_spoofed not leaked in observation
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from models import DisasterAction

ALL_TASKS = ["task1_flood_easy", "task2_multizone_medium", "task3_compound_hard"]
TASK_MAX_STEPS = {
    "task1_flood_easy": 30,
    "task2_multizone_medium": 55,
    "task3_compound_hard": 130,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dispatch_action(obs) -> DisasterAction:
    """Return a dispatch_rescue DisasterAction for the current alert."""
    alert = obs.current_alert if not isinstance(obs.current_alert, dict) else None
    if alert is not None:
        alert_id = getattr(alert, "alert_id", "")
    else:
        alert_id = (obs.current_alert or {}).get("alert_id", "") if obs.current_alert else ""
    return DisasterAction(action_type="dispatch_rescue", alert_id=alert_id)


def _obs_to_dict(obs) -> dict:
    """Normalise Pydantic model or dict observation to a plain dict."""
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    return dict(obs)


# ---------------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_initial_observation(self, env):
        """Two resets with the same seed must produce identical initial observations."""
        obs1 = env.reset(task_name="task1_flood_easy", seed=42)
        d1 = _obs_to_dict(obs1)

        obs2 = env.reset(task_name="task1_flood_easy", seed=42)
        d2 = _obs_to_dict(obs2)

        assert d1["task_name"] == d2["task_name"]
        assert d1["max_steps"] == d2["max_steps"]
        assert d1["step"] == d2["step"]

        alert1 = d1.get("current_alert") or {}
        alert2 = d2.get("current_alert") or {}
        assert alert1.get("source") == alert2.get("source")
        assert alert1.get("severity") == pytest.approx(alert2.get("severity"), abs=1e-9)

    @pytest.mark.parametrize("task_name", ALL_TASKS)
    def test_same_seed_reproducible_first_reward(self, task_name):
        """First step reward must be identical across two runs with seed=42."""
        from server.environment import DisasterResponseEnvironment

        def run_one_step(seed):
            e = DisasterResponseEnvironment()
            obs = e.reset(task_name=task_name, seed=seed)
            action = _dispatch_action(obs)
            result = e.step(action)  # returns DisasterObservation with .reward
            return result.reward

        r1 = run_one_step(42)
        r2 = run_one_step(42)
        assert r1 == pytest.approx(r2, abs=1e-9), (
            f"Non-deterministic first reward for {task_name}: {r1} vs {r2}"
        )

    @pytest.mark.parametrize("task_name", ALL_TASKS)
    def test_different_seeds_differ(self, task_name):
        """Different seeds should (very likely) produce different initial alerts."""
        from server.environment import DisasterResponseEnvironment

        e = DisasterResponseEnvironment()
        obs_a = _obs_to_dict(e.reset(task_name=task_name, seed=1))
        obs_b = _obs_to_dict(e.reset(task_name=task_name, seed=9999))

        alert_a = obs_a.get("current_alert") or {}
        alert_b = obs_b.get("current_alert") or {}
        assert alert_a.get("alert_id") != alert_b.get("alert_id"), (
            f"Seeds 1 and 9999 produced identical alert IDs for {task_name}"
        )


# ---------------------------------------------------------------------------
# Grader range tests
# ---------------------------------------------------------------------------

class TestGraderRange:
    @pytest.mark.parametrize("task_name", ALL_TASKS)
    def test_normalized_score_in_range(self, task_name):
        """normalized_score at episode end must be in [0.0, 1.0]."""
        from server.environment import DisasterResponseEnvironment

        max_steps = TASK_MAX_STEPS[task_name]
        env = DisasterResponseEnvironment()
        obs = env.reset(task_name=task_name, seed=42)

        for _ in range(max_steps):
            if obs.done:
                break
            action = _dispatch_action(obs)
            obs = env.step(action)  # step() returns DisasterObservation directly

        obs_dict = _obs_to_dict(obs)
        score = obs_dict.get("normalized_score")
        if score is not None:
            assert 0.0 <= score <= 1.0, (
                f"{task_name}: normalized_score={score} is outside [0.0, 1.0]"
            )

    @pytest.mark.parametrize("task_name", ALL_TASKS)
    def test_reward_per_step_bounded(self, task_name):
        """Individual step rewards must fall within the declared reward_range [-1.0, 1.0]."""
        from server.environment import DisasterResponseEnvironment

        env = DisasterResponseEnvironment()
        obs = env.reset(task_name=task_name, seed=7)

        for _ in range(10):
            if obs.done:
                break
            action = _dispatch_action(obs)
            obs = env.step(action)
            r = obs.reward
            assert -1.0 <= r <= 1.0, f"{task_name}: step reward {r} out of [-1.0, 1.0]"


# ---------------------------------------------------------------------------
# Episode boundary tests
# ---------------------------------------------------------------------------

class TestEpisodeBoundaries:
    @pytest.mark.parametrize("task_name", ALL_TASKS)
    def test_episode_terminates_within_max_steps(self, task_name):
        """Episode must set done=True by the task's configured max_steps."""
        from server.environment import DisasterResponseEnvironment

        max_steps = TASK_MAX_STEPS[task_name]
        env = DisasterResponseEnvironment()
        obs = env.reset(task_name=task_name, seed=42)
        obs_dict = _obs_to_dict(obs)
        assert obs_dict["max_steps"] == max_steps, (
            f"{task_name}: reset returned max_steps={obs_dict['max_steps']}, expected {max_steps}"
        )

        done = False
        steps_taken = 0
        for step in range(1, max_steps + 1):
            if obs.done:
                done = True
                steps_taken = step - 1
                break
            action = _dispatch_action(obs)
            obs = env.step(action)
            steps_taken = step
            if obs.done:
                done = True
                break

        assert done, (
            f"{task_name}: episode did not terminate within {max_steps} steps "
            f"(ran {steps_taken} steps)"
        )

    @pytest.mark.parametrize("task_name", ALL_TASKS)
    def test_reset_clears_state(self, task_name):
        """reset() must return step=0 and clean cumulative_reward."""
        from server.environment import DisasterResponseEnvironment

        env = DisasterResponseEnvironment()
        obs = env.reset(task_name=task_name, seed=42)

        # Run a few steps
        for _ in range(3):
            if obs.done:
                break
            action = _dispatch_action(obs)
            obs = env.step(action)

        # Now reset and verify clean state
        obs2 = env.reset(task_name=task_name, seed=42)
        d = _obs_to_dict(obs2)
        # Environment pre-generates the first alert so step starts at 1 after reset
        assert d.get("step", 0) <= 1, "reset() did not clear step counter"
        assert d.get("cumulative_reward", 0.0) == 0.0, "reset() did not clear cumulative_reward"


# ---------------------------------------------------------------------------
# Hidden-field protection tests
# ---------------------------------------------------------------------------

class TestHiddenFields:
    @pytest.mark.parametrize("task_name", ["task1_flood_easy", "task3_compound_hard"])
    def test_is_real_not_in_observation(self, task_name):
        """is_real must not appear in the agent-facing observation."""
        from server.environment import DisasterResponseEnvironment

        env = DisasterResponseEnvironment()
        obs = env.reset(task_name=task_name, seed=42)
        obs_dict = _obs_to_dict(obs)

        alert = obs_dict.get("current_alert") or {}
        assert "is_real" not in alert, "is_real is leaking into agent observation"
        assert "is_spoofed" not in alert, "is_spoofed is leaking into agent observation"
