# AgriEnv - OpenEnv RL Environment

## Motivation
AgriEnv simulates a real-world precision agriculture and water conservation task. Agents act as automated irrigation controllers, balancing crop health against water scarcity and unpredictable weather. This addresses crucial social and ecological challenges in water management.

## Observation Space
The agent receives an `AgriObservation` containing:
- `day` (int): Current day of the season.
- `soil_moisture` (float): Current moisture (0.0 to 1.0). Ideal range is 0.4 - 0.7.
- `weather_forecast` (str): Tomorrow's weather ('sunny', 'cloudy', 'rainy', 'drought').
- `water_reserves` (float): Remaining water limit in liters.
- `crop_health` (float): Overall health of the crop (0.0 to 1.0).

## Action Space
The agent outputs an `AgriAction`:
- `action_type` (str): 'irrigate' or 'wait'.
- `amount` (float): Liters of water to use (0.0 to 10.0).

## Tasks & Difficulty
1. **Easy**: 10-day season, abundant water (1000L), predictable sunny weather.
2. **Medium**: 20-day season, limited water (100L), mixed weather requiring planning.
3. **Hard**: 30-day season, severe water constraints (80L), risk of drought. Agent must strictly optimize water usage.

## Setup & Inference
Run the baseline inference:
```bash
docker build -t agri-env .
docker run -e HF_TOKEN="your_token" -e MODEL_NAME="Qwen/Qwen2.5-72B-Instruct" agri-env