import os
import json
from openai import OpenAI
from environment import SmartAgricultureEnv, AgriAction

# --- MANDATORY ENV VARS ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

# Task Setup
TASK_NAME = os.getenv("AGRI_TASK", "easy_irrigation")
BENCHMARK = "SmartAgricultureEnv"

def main():
    # Initialize OpenAI Client
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    
    # Initialize Environment
    env = SmartAgricultureEnv(task_name=TASK_NAME)
    obs = env.reset()

    # 1. STRICT LOGGING: [START]
    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}")

    done = False
    step = 0
    rewards =[]
    score = 0.0

    while not done and step < 10:  # Safety limit
        step += 1
        error_msg = "null"
        
        # Build prompt for the LLM
        prompt = (
            f"You are an AI agricultural assistant. Current day state: {obs.model_dump_json()}\n"
            "Provide optimal water and fertilizer allocations to maximize crop growth.\n"
            "Reply ONLY with a JSON object matching this schema:\n"
            '{"water_allocations": {"0": 0.2}, "fertilizer_allocations": {"0": 0.1}}'
        )

        try:
            # Call LLM
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            action_dict = json.loads(response.choices[0].message.content)
            
            # Convert string keys to int for our Pydantic model if necessary
            w_alloc = {int(k): v for k, v in action_dict.get("water_allocations", {}).items()}
            f_alloc = {int(k): v for k, v in action_dict.get("fertilizer_allocations", {}).items()}
            
            action = AgriAction(water_allocations=w_alloc, fertilizer_allocations=f_alloc)
            action_str = json.dumps({"w": w_alloc, "f": f_alloc}).replace(" ", "")
            
        except Exception as e:
            # Fallback to prevent crash, log error
            action = AgriAction(water_allocations={0: 0.1}, fertilizer_allocations={0: 0.1})
            action_str = "{fallback}"
            error_msg = str(e).replace('\n', ' ')[:50]

        # Step the environment
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        
        # 2. STRICT LOGGING: [STEP]
        done_str = "true" if done else "false"
        print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_str} error={error_msg}")

        if done:
            score = info.get("score", 0.0)

    # 3. STRICT LOGGING: [END]
    success_str = "true" if score >= 0.5 else "false"
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(f"[END] success={success_str} steps={step} score={score:.2f} rewards={rewards_str}")

if __name__ == "__main__":
    main()