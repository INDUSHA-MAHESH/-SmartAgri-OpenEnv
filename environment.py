from typing import Dict, List, Any, Tuple
from pydantic import BaseModel, Field
import random

# --- 1. PYDANTIC MODELS (Strictly Typed for OpenEnv) ---

class FieldState(BaseModel):
    field_id: int
    moisture: float = Field(description="Moisture level from 0.0 (dry) to 1.0 (flooded). Ideal is 0.4-0.6.")
    nutrition: float = Field(description="Nutrient level from 0.0 to 1.0. Ideal is >0.5.")
    growth: float = Field(description="Crop growth progress from 0.0 to 1.0 (ready for harvest).")

class AgriObservation(BaseModel):
    day: int
    weather_forecast: str = Field(description="Forecast for tomorrow: 'Sunny', 'Cloudy', or 'Rainy'")
    water_budget: float = Field(description="Total water units available for the season.")
    fields: List[FieldState]

class AgriAction(BaseModel):
    water_allocations: Dict[int, float] = Field(
        description="Dictionary mapping field_id to units of water to apply (0.0 to 0.3 per field)."
    )
    fertilizer_allocations: Dict[int, float] = Field(
        description="Dictionary mapping field_id to units of fertilizer to apply (0.0 to 0.2 per field)."
    )

class AgriReward(BaseModel):
    reward: float
    feedback: str

# --- 2. ENVIRONMENT CLASS ---

class SmartAgricultureEnv:
    def __init__(self, task_name: str = "easy_irrigation"):
        self.task_name = task_name
        self.max_days = 7
        self.current_day = 0
        self.state_data = None
        
    def reset(self) -> AgriObservation:
        self.current_day = 1
        # Task Difficulty setup
        if self.task_name == "easy_irrigation":
            num_fields = 1
            self.water_budget = 10.0
        elif self.task_name == "medium_resource_management":
            num_fields = 3
            self.water_budget = 4.0 # Tight budget
        else: # hard_weather_adaptation
            num_fields = 3
            self.water_budget = 3.0 # Very tight, must rely on rain
            
        self.state_data = {
            "day": self.current_day,
            "weather_forecast": self._get_weather(),
            "water_budget": self.water_budget,
            "fields":[
                FieldState(field_id=i, moisture=0.3, nutrition=0.3, growth=0.0)
                for i in range(num_fields)
            ]
        }
        return AgriObservation(**self.state_data)

    def _get_weather(self):
        if self.task_name == "hard_weather_adaptation":
            return random.choice(["Sunny", "Sunny", "Rainy"]) # Rain happens sometimes
        return "Sunny"

    def state(self) -> AgriObservation:
        return AgriObservation(**self.state_data)

    def step(self, action: AgriAction) -> Tuple[AgriObservation, float, bool, Dict]:
        reward_step = 0.0
        feedback =[]
        
        # Apply actions and update fields
        for field in self.state_data["fields"]:
            fid = field.field_id
            water_applied = action.water_allocations.get(fid, 0.0)
            fert_applied = action.fertilizer_allocations.get(fid, 0.0)
            
            # Deduct budget
            if water_applied > 0:
                if self.state_data["water_budget"] >= water_applied:
                    self.state_data["water_budget"] -= water_applied
                else:
                    water_applied = self.state_data["water_budget"] # Use remaining
                    self.state_data["water_budget"] = 0.0
                    feedback.append(f"Ran out of water budget for field {fid}!")

            # Environmental dynamics (Weather effects)
            if self.state_data["weather_forecast"] == "Rainy":
                water_applied += 0.3 # Rain adds moisture
            
            # Update moisture and nutrition
            field.moisture = min(1.0, field.moisture - 0.1 + water_applied) # -0.1 daily evaporation
            field.nutrition = min(1.0, field.nutrition - 0.05 + fert_applied) # -0.05 daily depletion
            
            # Calculate Growth based on optimal ranges (Moisture 0.4-0.7, Nutrition > 0.5)
            growth_increment = 0.0
            if 0.4 <= field.moisture <= 0.7:
                growth_increment += 0.1
                reward_step += 0.05
            else:
                feedback.append(f"Field {fid} moisture non-optimal ({field.moisture:.2f}).")
                reward_step -= 0.05 # Penalty for over/under watering
                
            if field.nutrition >= 0.5:
                growth_increment += 0.05
                reward_step += 0.02
                
            field.growth = min(1.0, field.growth + growth_increment)
        
        # Advance Day
        self.current_day += 1
        self.state_data["day"] = self.current_day
        self.state_data["weather_forecast"] = self._get_weather()
        
        done = self.current_day > self.max_days
        
        # GRADER: Final Score Calculation (0.0 to 1.0)
        info = {}
        if done:
            total_growth = sum(f.growth for f in self.state_data["fields"])
            max_possible_growth = len(self.state_data["fields"]) * 1.0
            final_score = total_growth / max_possible_growth
            info["score"] = max(0.0, min(1.0, final_score)) # Clamp between 0.0 and 1.0
        
        return self.state(), reward_step, done, info