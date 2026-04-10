import os
import sys
import uvicorn
from fastapi import FastAPI, Body
from typing import Optional

# Robustly add the project root (/app) to sys.path regardless of where
# Python was invoked from (fixes Docker CMD from server/ subdirectory).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from environment import (
    SmartAgricultureEnv,
    AgriAction,
    VALID_TASKS,
    grade_easy_irrigation,
    grade_medium_resource_management,
    grade_hard_weather_adaptation,
)

app = FastAPI(title="Smart Agriculture OpenEnv Server")

# One shared env instance; task can be swapped via reset(task_id=...)
DEFAULT_TASK = os.getenv("OPENENV_TASK_ID", "easy_irrigation")
env = SmartAgricultureEnv(task_id=DEFAULT_TASK)

_GRADERS = {
    "easy_irrigation": grade_easy_irrigation,
    "medium_resource_management": grade_medium_resource_management,
    "hard_weather_adaptation": grade_hard_weather_adaptation,
}


# ---------------------------------------------------------------------------
# Health / ping
# ---------------------------------------------------------------------------

@app.get("/")
def health():
    return {"status": "ready", "task_id": env.task_id, "tasks": VALID_TASKS}


# ---------------------------------------------------------------------------
# OpenEnv core endpoints
# ---------------------------------------------------------------------------

@app.post("/reset")
def reset_post(payload: dict = Body(default={})):
    task_id = payload.get("task_id")
    obs = env.reset(task_id=task_id)
    return obs.model_dump()


@app.get("/reset")
def reset_get(task_id: Optional[str] = None):
    obs = env.reset(task_id=task_id)
    return obs.model_dump()


@app.get("/state")
@app.post("/state")
def state():
    return env.state().model_dump()


@app.post("/step")
def step(action: AgriAction):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


# ---------------------------------------------------------------------------
# Grader endpoint — THIS is what the Phase 2 validator calls
# ---------------------------------------------------------------------------

@app.get("/grade")
@app.post("/grade")
def grade(payload: dict = Body(default={})):
    """
    Run the grader for the current (or specified) task and return its score.
    The OpenEnv validator calls GET /grade?task_id=<id> or POST /grade.
    Returns {"task_id": str, "score": float, "success": bool}
    """
    task_id = payload.get("task_id") or env.task_id
    if task_id not in VALID_TASKS:
        return {"error": f"Unknown task_id '{task_id}'", "valid": VALID_TASKS}

    grader_fn = _GRADERS[task_id]
    score = grader_fn(env)
    from environment import _SUCCESS_THRESHOLD
    return {
        "task_id": task_id,
        "score": score,
        "success": score >= _SUCCESS_THRESHOLD[task_id],
    }


@app.get("/grade/{task_id}")
def grade_task(task_id: str):
    """Convenience: GET /grade/easy_irrigation"""
    if task_id not in VALID_TASKS:
        return {"error": f"Unknown task_id '{task_id}'", "valid": VALID_TASKS}
    grader_fn = _GRADERS[task_id]
    # Reset env to that task first so grader sees a clean run
    env.reset(task_id=task_id)
    score = grader_fn(env)
    from environment import _SUCCESS_THRESHOLD
    return {
        "task_id": task_id,
        "score": score,
        "success": score >= _SUCCESS_THRESHOLD[task_id],
    }


@app.get("/tasks")
def list_tasks():
    """Return all tasks with their grader status — used by validators."""
    from environment import _SUCCESS_THRESHOLD
    return {
        "tasks": [
            {
                "id": tid,
                "grader": f"environment:grade_{tid}",
                "success_threshold": _SUCCESS_THRESHOLD[tid],
            }
            for tid in VALID_TASKS
        ]
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
