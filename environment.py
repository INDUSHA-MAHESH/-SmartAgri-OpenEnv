from typing import Dict, List, Any, Tuple, Optional
from pydantic import BaseModel, Field
import random

# ---------------------------------------------------------------------------
# 1. PYDANTIC MODELS
# ---------------------------------------------------------------------------

class FieldState(BaseModel):
    field_id: int
    moisture: float = Field(description="Moisture level 0.0-1.0. Ideal: 0.4-0.6.")
    nutrition: float = Field(description="Nutrient level 0.0-1.0. Ideal: >0.5.")
    growth: float = Field(description="Crop growth progress 0.0-1.0.")

    model_config = {"frozen": False}  # allow in-place mutation safely


class AgriObservation(BaseModel):
    day: int
    weather_forecast: str = Field(description="Forecast: 'Sunny' or 'Rainy'")
    water_budget: float = Field(description="Remaining water units.")
    fields: List[FieldState]


class AgriAction(BaseModel):
    water_allocations: Dict[int, float] = Field(
        description="Water per field (0.0-0.3). Keys are field_id as int."
    )
    fertilizer_allocations: Dict[int, float] = Field(
        description="Fertilizer per field (0.0-0.2). Keys are field_id as int."
    )


# ---------------------------------------------------------------------------
# 2. ENVIRONMENT CLASS
# ---------------------------------------------------------------------------

VALID_TASKS = ["easy_irrigation", "medium_resource_management", "hard_weather_adaptation"]


class SmartAgricultureEnv:
    def __init__(self, task_id: str = "easy_irrigation", **kwargs):
        if task_id not in VALID_TASKS:
            raise ValueError(f"Unknown task_id '{task_id}'. Valid: {VALID_TASKS}")
        self.task_id = task_id
        self.max_days = 7
        self.current_day = 0
        self.water_budget = 0.0
        # Safe default state so state() never crashes before reset()
        self._init_state_data()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_state_data(self):
        """Build a minimal valid state so /state never 500 before reset()."""
        self.state_data = {
            "day": 0,
            "weather_forecast": "Sunny",
            "water_budget": 0.0,
            "fields": [FieldState(field_id=0, moisture=0.3, nutrition=0.3, growth=0.0)],
        }

    def _get_weather(self) -> str:
        if self.task_id == "hard_weather_adaptation":
            return random.choice(["Sunny", "Sunny", "Rainy"])
        return "Sunny"

    def _num_fields_and_budget(self) -> Tuple[int, float]:
        if self.task_id == "easy_irrigation":
            return 1, 10.0
        elif self.task_id == "medium_resource_management":
            return 3, 4.0
        else:  # hard_weather_adaptation
            return 3, 2.5

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self, task_id: Optional[str] = None) -> AgriObservation:
        if task_id and task_id in VALID_TASKS:
            self.task_id = task_id

        self.current_day = 1
        num_fields, self.water_budget = self._num_fields_and_budget()

        self.state_data = {
            "day": self.current_day,
            "weather_forecast": self._get_weather(),
            "water_budget": self.water_budget,
            "fields": [
                FieldState(field_id=i, moisture=0.3, nutrition=0.3, growth=0.0)
                for i in range(num_fields)
            ],
        }
        return AgriObservation(**self.state_data)

    def state(self) -> AgriObservation:
        return AgriObservation(**self.state_data)

    def step(self, action: AgriAction) -> Tuple[AgriObservation, float, bool, Dict]:
        reward_step = 0.0

        for field in self.state_data["fields"]:
            fid = field.field_id
            water_applied = float(action.water_allocations.get(fid, 0.0))
            fert_applied = float(action.fertilizer_allocations.get(fid, 0.0))

            # Clamp to budget
            water_applied = min(water_applied, self.state_data["water_budget"])
            self.state_data["water_budget"] = max(
                0.0, self.state_data["water_budget"] - water_applied
            )

            # Rainfall bonus
            if self.state_data["weather_forecast"] == "Rainy":
                water_applied += 0.3

            # Physics updates
            field.moisture = round(
                min(1.0, max(0.0, field.moisture - 0.1 + water_applied)), 4
            )
            field.nutrition = round(
                min(1.0, max(0.0, field.nutrition - 0.05 + fert_applied)), 4
            )

            # Growth & reward signal (partial, every step)
            growth_inc = 0.0
            if 0.4 <= field.moisture <= 0.7:
                growth_inc += 0.1
                reward_step += 0.1
            else:
                reward_step -= 0.05  # penalise bad moisture

            if field.nutrition >= 0.5:
                growth_inc += 0.05
                reward_step += 0.05

            field.growth = round(min(1.0, field.growth + growth_inc), 4)

        self.current_day += 1
        self.state_data["day"] = self.current_day
        self.state_data["weather_forecast"] = self._get_weather()

        done = self.current_day > self.max_days

        # Final score from grader
        score = self._compute_score()
        info = {
            "score": round(score, 2),
            "success": score >= _SUCCESS_THRESHOLD[self.task_id],
            "task_id": self.task_id,
        }
        return self.state(), round(reward_step, 2), done, info

    # ------------------------------------------------------------------
    # Score helper (shared by graders)
    # ------------------------------------------------------------------

    def _compute_score(self) -> float:
        fields = self.state_data["fields"]
        avg_growth = sum(f.growth for f in fields) / max(len(fields), 1)
        return max(0.0, min(1.0, avg_growth))


# ---------------------------------------------------------------------------
# 3. SUCCESS THRESHOLDS (used by graders and env)
# ---------------------------------------------------------------------------

_SUCCESS_THRESHOLD = {
    "easy_irrigation": 0.7,
    "medium_resource_management": 0.6,
    "hard_weather_adaptation": 0.5,
}


# ---------------------------------------------------------------------------
# 4. GRADER FUNCTIONS — required by openenv.yaml  (one per task)
#    Signature: grader(env: SmartAgricultureEnv) -> float  (0.0 – 1.0)
#    The validator instantiates the env, runs reset(), calls step() up to
#    max_days times with a neutral action, then calls the grader.
# ---------------------------------------------------------------------------

def _run_episode_with_neutral_action(task_id: str) -> SmartAgricultureEnv:
    """Run a full episode with zero allocations and return the final env state."""
    env = SmartAgricultureEnv(task_id=task_id)
    env.reset()
    done = False
    while not done:
        num_fields = len(env.state_data["fields"])
        action = AgriAction(
            water_allocations={i: 0.15 for i in range(num_fields)},
            fertilizer_allocations={i: 0.1 for i in range(num_fields)},
        )
        _, _, done, _ = env.step(action)
    return env


def grade_easy_irrigation(env: SmartAgricultureEnv) -> float:
    """
    Grader for easy_irrigation.
    Score = average growth across all fields, normalised to [0, 1].
    Full marks (1.0) when average growth >= 0.7.
    """
    score = env._compute_score()
    return round(score, 2)


def grade_medium_resource_management(env: SmartAgricultureEnv) -> float:
    """
    Grader for medium_resource_management.
    Score = average growth. Penalises wasted water (leftover budget hurts).
    """
    avg_growth = env._compute_score()
    # Small efficiency bonus for using the water budget wisely
    leftover_ratio = env.state_data["water_budget"] / 4.0  # original budget
    efficiency_penalty = leftover_ratio * 0.1  # max -0.1 for wasting all water
    score = max(0.0, min(1.0, avg_growth - efficiency_penalty))
    return round(score, 2)


def grade_hard_weather_adaptation(env: SmartAgricultureEnv) -> float:
    """
    Grader for hard_weather_adaptation.
    Score = average growth. Harder success threshold (0.5).
    Weather randomness means perfect scores are rare — 0.5+ is a win.
    """
    score = env._compute_score()
    return round(score, 2)


# ---------------------------------------------------------------------------
# 5. STANDALONE GRADER ENTRY POINTS
#    Some validators call grader(task_id) directly rather than grader(env).
#    These wrappers make both call styles work.
# ---------------------------------------------------------------------------

def run_grader(task_id: str) -> float:
    """Run a full episode and return the score for task_id."""
    env = _run_episode_with_neutral_action(task_id)
    graders = {
        "easy_irrigation": grade_easy_irrigation,
        "medium_resource_management": grade_medium_resource_management,
        "hard_weather_adaptation": grade_hard_weather_adaptation,
    }
    grader_fn = graders.get(task_id)
    if grader_fn is None:
        raise ValueError(f"No grader for task_id '{task_id}'")
    return grader_fn(env)


if __name__ == "__main__":
    for tid in VALID_TASKS:
        s = run_grader(tid)
        print(f"[GRADER] task={tid} score={s:.2f}")
