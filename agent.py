"""
System prompt, agent loop, per-step logging

approach/flow of an episode:
1. take starting url and goal/instruction
2. call get_observation() to get initial observation
3. build user message with observation, goal, and history
4. call llm to get action
5. parse and validate action
6. execute action
7. log action in jsonl
8. advance action history
9. repeat 2-8 (one step) until stop or max_steps

using single-turn system prompt/user message for now, dont think we need context of past outputs yet
as long as we give action history in user message, model can reason about it
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from actions import ACTION_VOCABULARY, ActionParseError, parse_action
from browser_env import BrowserEnv
from llm import chat


_SYSTEM_PROMPT = f"""You are an autonomous browser agent. You control a real web browser.

Each turn you will receive:
  - GOAL: the task you must complete
  - URL: the current page URL
  - PAGE: a numbered list of interactive elements on the current page
  - HISTORY: the actions you have already taken (empty on the first turn)

Your job is to decide the single next action that best makes progress toward the GOAL.

{ACTION_VOCABULARY}

Rules:
1. Output EXACTLY ONE action per reply, on a single line. Nothing else — no explanation, no preamble.
2. Use only the actions listed above. Any other output will be treated as a parse error.
3. When the GOAL is achieved, output: stop
4. If the page does not help and you cannot make progress, output: stop
""".strip()


def build_system_prompt() -> str:
    """Returns the fixed system message for every episode."""
    return _SYSTEM_PROMPT


# per-turn user message
def build_user_message(
    goal: str,
    obs_text: str,
    action_history: list[str],
) -> str:
    """
    Builds the user message for one step.

    Args:
        goal:           The task instruction (set once per episode, repeated every turn)
        obs_text:       Full text from BrowserEnv.get_observation() for this step
        action_history: All action lines taken so far this episode (empty on first step)
    """
    history_block = (
        "\n".join(f"  {i + 1}. {a}" for i, a in enumerate(action_history))
        if action_history
        else "  (none)"
    )
    return (
        f"GOAL: {goal}\n\n"
        f"HISTORY:\n{history_block}\n\n"
        f"PAGE:\n{obs_text}"
    )


# agent loop

_LOGS_DIR = Path("logs")


def run_episode(
    url: str,
    goal: str,
    model: str = "gpt-5.4-mini",
    max_steps: int = 20,
    headless: bool = True,
) -> list[dict]:
    """
    Runs one episode: open url, observe->act->log loop until stop or max_steps

    Returns the list of step dicts (same content written to the log file)
    Log file: logs/run_<run_id>.jsonl (one json object per line per step)
    """
    _LOGS_DIR.mkdir(exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = _LOGS_DIR / f"run_{run_id}.jsonl"

    system_msg = {"role": "system", "content": build_system_prompt()}
    action_history: list[str] = []
    steps: list[dict] = []

    print(f"[run {run_id}] goal: {goal}")
    print(f"[run {run_id}] log:  {log_path}")

    with BrowserEnv(headless=headless) as env, open(log_path, "a") as log_f:
        env.goto(url)

        for step_num in range(max_steps):
            # observe
            obs = env.get_observation()

            # build messages
            user_msg = {
                "role": "user",
                "content": build_user_message(goal, obs.text, action_history),
            }
            messages = [system_msg, user_msg]

            # call llm
            raw = chat(messages, model=model).strip()
            print(f"[step {step_num}] model: {raw!r}")

            # parse
            parse_error: str | None = None
            parsed = None
            try:
                parsed = parse_action(raw)
            except ActionParseError as e:
                parse_error = str(e)
                print(f"[step {step_num}] parse error: {parse_error}")

            # execute (skip if parse failed)
            exec_result: dict | None = None
            if parsed is not None:
                exec_result = env.execute_action(parsed)
                print(f"[step {step_num}] exec: {exec_result}")

            # log step
            step_record = {
                "run_id": run_id,
                "step": step_num,
                "url": env.page.url,
                "n_elements": obs.n_elements,
                "obs_truncated": obs.truncated,
                "obs_snippet": obs.text[:200],
                "raw_model_output": raw,
                "parse_ok": parse_error is None,
                "parse_error": parse_error,
                "exec_result": exec_result,
            }
            steps.append(step_record)
            log_f.write(json.dumps(step_record) + "\n")
            log_f.flush()

            # advance history (only on successful parse)
            if parsed is not None:
                action_history.append(raw)

            # stop conditions
            if parse_error is not None:
                print(f"[run {run_id}] stopping: parse error on step {step_num}")
                break
            if parsed is not None and (parsed.action_type == "stop" or (exec_result or {}).get("done")):
                print(f"[run {run_id}] done at step {step_num}")
                break

        else:
            print(f"[run {run_id}] reached max_steps ({max_steps})")

    print(f"[run {run_id}] log saved: {log_path}")
    return steps


if __name__ == "__main__":
    # needs api key set in env and playwright install chromium
    steps = run_episode(
        url="https://duckduckgo.com", # agent loop works,
                                      # ddg is much less adversarial to headed playwright than google.
                                      # annoyingly headless playwright is still blocked
        goal="Search for 'playwright' in the search bar, then click the first organic search result link on the results page.",
        model="gpt-5.4-mini",
        max_steps=5,
        headless=False, # great for testing
    )
    print(f"\nTotal steps recorded: {len(steps)}")
