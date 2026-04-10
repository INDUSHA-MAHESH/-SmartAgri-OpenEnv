"""
inference.py -- SmartAgri OpenEnv Baseline

MANDATORY ENV VARS:
  API_BASE_URL   The API endpoint for the LLM.
  MODEL_NAME     The model identifier to use for inference.
  HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT (strictly followed):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import json
import re
from openai import OpenAI
from environment import SmartAgricultureEnv, AgriAction, VALID_TASKS, _SUCCESS_THRESHOLD

# ---------------------------------------------------------------------------
# Config (mandatory per hackathon rules)
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "")
BENCHMARK    = "SmartAgricultureEnv"
MAX_STEPS    = 10


def parse_json_safe(text):
    """Extract JSON from LLM response even if wrapped in markdown fences."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


def build_prompt(obs, task_id):
    num_fields = len(obs.fields)
    field_keys = list(range(num_fields))
    fields_info = [
        {"id": f.field_id, "moisture": f.moisture,
         "nutrition": f.nutrition, "growth": f.growth}
        for f in obs.fields
    ]
    example_w = ", ".join('"' + str(k) + '": 0.15' for k in field_keys)
    example_f = ", ".join('"' + str(k) + '": 0.10' for k in field_keys)
    example_json = (
        '{"water_allocations": {' + example_w + '}, '
        '"fertilizer_allocations": {' + example_f + '}}'
    )
    return "\n".join([
        "You are an AI agricultural assistant managing {} crop field(s).".format(num_fields),
        "Task: {}".format(task_id),
        "Current state (day {}/7):".format(obs.day),
        "  Weather forecast: {}".format(obs.weather_forecast),
        "  Remaining water budget: {:.2f}".format(obs.water_budget),
        "  Fields: {}".format(fields_info),
        "",
        "Rules:",
        "  - water_allocations: each field gets 0.0-0.3 units. Total must not exceed budget.",
        "  - fertilizer_allocations: each field gets 0.0-0.2 units (no budget limit).",
        "  - Ideal moisture is 0.4-0.6. Rainy days add 0.3 automatically.",
        "  - Nutrition >= 0.5 boosts growth.",
        "",
        "Reply ONLY with a valid JSON object, no markdown, no explanation:",
        example_json,
    ])


def run_task(client, task_id):
    """Run one full episode and print [START] / [STEP] / [END] lines."""
    # FIXED: constructor uses task_id= (not task_name=)
    env = SmartAgricultureEnv(task_id=task_id)
    obs = env.reset()

    print("[START] task={} env={} model={}".format(task_id, BENCHMARK, MODEL_NAME))

    done = False
    step = 0
    rewards = []
    info = {}

    while not done and step < MAX_STEPS:
        step += 1
        error_msg = "null"
        action_str = "{fallback}"

        prompt = build_prompt(obs, task_id)

        try:
            # FIXED: no response_format param -- not universally supported
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=256,
            )
            raw = response.choices[0].message.content or ""
            action_dict = parse_json_safe(raw)

            # JSON keys are strings -- convert to int for Pydantic model
            w_alloc = {int(k): float(v) for k, v in action_dict.get("water_allocations", {}).items()}
            f_alloc = {int(k): float(v) for k, v in action_dict.get("fertilizer_allocations", {}).items()}

            action = AgriAction(water_allocations=w_alloc, fertilizer_allocations=f_alloc)
            action_str = json.dumps({"w": w_alloc, "f": f_alloc}, separators=(",", ":"))

        except Exception as e:
            num_fields = len(obs.fields)
            action = AgriAction(
                water_allocations={i: 0.1 for i in range(num_fields)},
                fertilizer_allocations={i: 0.1 for i in range(num_fields)},
            )
            error_msg = str(e).replace("\n", " ")[:80]

        obs, reward, done, info = env.step(action)
        rewards.append(reward)

        done_str = "true" if done else "false"
        print("[STEP] step={} action={} reward={:.2f} done={} error={}".format(
            step, action_str, reward, done_str, error_msg))

    # FIXED: always read score from last info dict, regardless of how loop ended
    score = info.get("score", 0.0)
    threshold = _SUCCESS_THRESHOLD.get(task_id, 0.7)
    success_str = "true" if score >= threshold else "false"
    rewards_str = ",".join("{:.2f}".format(r) for r in rewards)
    print("[END] success={} steps={} score={:.2f} rewards={}".format(
        success_str, step, score, rewards_str))


def main():
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    # FIXED: loop over ALL 3 tasks (required by hackathon spec)
    for task_id in VALID_TASKS:
        run_task(client, task_id)


if __name__ == "__main__":
    main()
