#!/usr/bin/env python3
# ReAct-style Text ALFWorld runner

import os
import sys
import json
import yaml
import time
from typing import List, Tuple
import argparse
import re
import random
from tqdm import tqdm
import pdb


# OpenAI legacy completion (as per your snippet)
from openai import OpenAI

# Ensure your API key is set in the environment (OpenAI client reads env var)
client = OpenAI()
 # Select model from env; defaults to chat model gpt-4o. Set to an *-instruct model to use the Completions API.
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
# Limit generation length to control TPM usage (can override via env)
MAX_ACTION_TOKENS = int(os.environ.get("OPENAI_MAX_ACTION_TOKENS", "60"))
# # Select model from env; defaults to chat model gpt-4o. Set to an *-instruct model to use the Completions API.
# MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

# ------------------------------------------------------------
# Output configuration
# ------------------------------------------------------------
# Centralized transcript output directory. Change here to redirect all run logs.
SAVE_DIR = "./react_runs_misalign_model/"

# ------------------------------------------------------------
# Exact token usage accounting (from OpenAI API responses)
# ------------------------------------------------------------
TOKEN_USAGE = {"prompt": 0, "completion": 0}

def _record_usage(resp):
    try:
        usage = getattr(resp, "usage", None)
        if usage:
            TOKEN_USAGE["prompt"] += int(getattr(usage, "prompt_tokens", 0) or 0)
            TOKEN_USAGE["completion"] += int(getattr(usage, "completion_tokens", 0) or 0)
    except Exception:
        pass

def get_token_usage():
    return dict(TOKEN_USAGE)

def llm(prompt: str, stop: List[str] = None) -> str:
    """
    Deterministic single-line generation for ReAct.
    - Chat models (e.g., gpt-4o, gpt-4o-mini) use Chat Completions.
    - Instruct models (e.g., gpt-3.5-turbo-instruct) use Completions.
    """
    stop = stop or ["\n", "\nObservation:", "\n>", "\nthink:", "\nThought:"]

    # Retry loop for rate limits / transient errors
    max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", "5"))
    base_sleep = float(os.environ.get("OPENAI_RETRY_BASE", "1.5"))
    last_err = None

    for attempt in range(max_retries):
        try:
            if MODEL.endswith("-instruct"):
                # Seed the line so the model *must* continue an Action
                seeded_prompt = prompt + " Action:"
                resp = client.completions.create(
                    model=MODEL,
                    prompt=seeded_prompt,
                    temperature=0,
                    max_tokens=MAX_ACTION_TOKENS,
                    top_p=1,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    stop=stop,
                )
                _record_usage(resp)
                text = resp.choices[0].text
            else:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "output exactly one action line."
                                "Interact with a household to solve a task."
                                "Always say 'put in/on' instead of just 'put in' or 'put on'."
                                "Only use the aciton described"
                            ),
                        },
                        {"role": "user", "content": prompt},
                        # Seed the assistant so the model continues this line
                        {"role": "assistant", "content": "Action:"},
                    ],
                    temperature=0,
                    max_tokens=MAX_ACTION_TOKENS,
                    top_p=1,
                    stop=stop,
                )
                _record_usage(resp)
                text = resp.choices[0].message.content

            # Successful call; break retry loop
            break
        except Exception as e:
            last_err = e
            # Basic classification for 429 or transient network issues
            msg = str(e)
            if "429" in msg or "Rate limit" in msg or "Too Many Requests" in msg:
                # Exponential backoff with jitter
                sleep_s = base_sleep * (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(sleep_s)
                continue
            # Other httpx/httpcore temporary errors
            if any(tok in msg for tok in [
                "timeout",
                "Temporary failure",
                "Connection reset",
                "Read error",
                # Upstream/connectivity resets often seen from gateways/proxies
                "upstream connect error",
                "disconnect/reset before headers",
                "connection termination",
                # Generic 5xx markers
                "InternalServerError",
                "502",
                "503",
                "504",
                "Bad Gateway",
                "Service Unavailable",
                "Gateway Timeout",
            ]):
                sleep_s = 0.5 * (2 ** attempt) + random.uniform(0, 0.25)
                time.sleep(sleep_s)
                continue
            # Non-retryable
            raise

    if last_err and 'text' not in locals():
        # All retries exhausted
        raise last_err

    # --- cleanup: drop leading '>' and whitespace, normalize put in/on ---
    cleaned = text.lstrip("> \t\r\n").strip()
    if not re.search(r"\bput\b.*\bin/on\b", cleaned, flags=re.IGNORECASE):
        cleaned = re.sub(r"(?i)\bput\b(.*?)\b(in|on)\b", r"put\1in/on", cleaned)

    # Remove trailing punctuation like periods that some models append to actions
    cleaned = re.sub(r"[\.!?]+$", "", cleaned).strip()

    # # If the model still tried to give you a think line, strip it off.
    # if cleaned.lower().startswith("think:"):
    #     # keep only the first non-empty line that isn't a think line
    #     for line in cleaned.splitlines():
    #         line = line.strip()
    #         if line and not line.lower().startswith("think:"):
    #             cleaned = line
    #             break
    cleaned = cleaned.lower()

    return cleaned

def llm_think(prompt: str, stop: List[str] = None) -> str:
    """
    Deterministic single-line 'think:' generation used before each action.
    Produces exactly one line starting with 'think:' and no trailing punctuation.
    """
    stop = stop or ["\n", "\n>", "\nObservation:", "\nThought:"]

    max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", "5"))
    base_sleep = float(os.environ.get("OPENAI_RETRY_BASE", "1.5"))
    last_err = None

    for attempt in range(max_retries):
        try:
            if MODEL.endswith("-instruct"):
                seeded = prompt + " think:"
                resp = client.completions.create(
                    model=MODEL,
                    prompt=seeded,
                    temperature=0,
                    max_tokens=MAX_ACTION_TOKENS,
                    top_p=1,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    stop=stop,
                )
                _record_usage(resp)
                text = resp.choices[0].text
            else:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": (
                            "Produce one internal reasoning line starting with 'think:' only. "
                            "Do not output any action."
                        )},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": "think:"},
                    ],
                    temperature=0,
                    max_tokens=MAX_ACTION_TOKENS,
                    top_p=1,
                    stop=stop,
                )
                _record_usage(resp)
                text = resp.choices[0].message.content
            break
        except Exception as e:
            last_err = e
            msg = str(e)
            if "429" in msg or "Rate limit" in msg or "Too Many Requests" in msg:
                sleep_s = base_sleep * (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(sleep_s)
                continue
            if any(tok in msg for tok in [
                "timeout","Temporary failure","Connection reset","Read error",
                "upstream connect error","disconnect/reset before headers","connection termination",
                "InternalServerError","502","503","504","Bad Gateway","Service Unavailable","Gateway Timeout",
            ]):
                sleep_s = 0.5 * (2 ** attempt) + random.uniform(0, 0.25)
                time.sleep(sleep_s)
                continue
            raise

    if last_err and 'text' not in locals():
        raise last_err

    think_line = text.lstrip("> \t\r\n").strip()
    # Ensure prefix and normalize
    if not think_line.lower().startswith("think:"):
        think_line = "think: " + think_line
    think_line = re.sub(r"[\.!?]+$", "", think_line).strip()
    return think_line.lower()

def process_ob(ob: str) -> str:
    """Clean up ALFWorld obs prefix for readability (as in your snippet)."""
    if ob.startswith("You arrive at loc "):
        ob = ob[ob.find(". ") + 2 :]
    return ob


def misalign_text(text: str) -> str:
    """Rewrite goal/observation text with the requested misaligned vocabulary.

    Mappings (case-insensitive, word-boundary):
      - "hot"  -> "cook"
      - "cool" -> "chill"
      - "clean" -> "wash"
      - "put" -> "store"
      - "with the" -> "underneath"

    Applied to the goal string provided to LLMs and to every observation appended
    to the prompt (i.e., the observation space).
    """
    if not isinstance(text, str) or not text:
        return text

    rules = [
        (r"\bwith the\b", "underneath"),
        (r"\bhot\b", "cook"),
        (r"\bcool\b", "chill"),
        (r"\bclean\b", "wash"),
        (r"\bput\b", "store"),
    ]
    out = text
    for pat, repl in rules:
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)
    return out


def misalign_goal_only(text: str) -> str:
    """Apply misalignment only to the goal line that starts with 'Your task is to:'."""
    if not isinstance(text, str) or not text:
        return text
    lines = text.splitlines()
    out_lines = []
    for line in lines:
        if line.strip().lower().startswith("your task is to:"):
            out_lines.append(misalign_text(line))
        else:
            out_lines.append(line)
    return "\n".join(out_lines)

def load_prompts(prompt_path: str) -> dict:
    with open(prompt_path, "r", encoding="utf-8") as f:
        return json.load(f)


def default_action_templates() -> list:
    """Fallback set of ALFWorld text actions with masked objects as 'obj'."""
    return [
        "go to obj 1",
        "open obj 1",
        "close obj 1",
        "take obj 1 from obj 1",
        "put obj 1 in/on obj 1",
        "clean obj 1 with obj 1",
        "heat obj 1 with obj 1",
        "cool obj 1 with obj 1",
        "turn on obj 1",
        "turn off obj 1",
    ]


def extract_action_templates_from_examples(prompt_bank: dict) -> list:
    """Parse exemplar texts to infer distinct action templates and mask objects as 'obj'."""
    actions = set()
    pattern = re.compile(r"^>\s*(?!think:)(.+)$")
    # Gather all exemplar strings
    for v in prompt_bank.values():
        for line in v.splitlines():
            m = pattern.match(line.strip())
            if not m:
                continue
            act = m.group(1).strip()
            # Normalize whitespace
            act = re.sub(r"\s+", " ", act)
            # Mask object type tokens heuristically by replacing specific nouns with 'obj'
            # We also mask integers to '1' for consistency
            act = re.sub(r"\b\d+\b", "1", act)
            # Replace known prepositions remain the same; replace other words except verbs/preps with 'obj'
            # Simple heuristic per common action forms
            if act.startswith("go to "):
                actions.add("go to obj 1")
            elif act.startswith("open "):
                actions.add("open obj 1")
            elif act.startswith("close "):
                actions.add("close obj 1")
            elif act.startswith("take ") and " from " in act:
                actions.add("take obj 1 from obj 1")
            elif act.startswith("put ") and (" in/on " in act or " in " in act or " on " in act):
                actions.add("put obj 1 in/on obj 1")
            elif act.startswith("clean ") and " with " in act:
                actions.add("clean obj 1 with obj 1")
            elif act.startswith("heat ") and " with " in act:
                actions.add("heat obj 1 with obj 1")
            elif act.startswith("cool ") and " with " in act:
                actions.add("cool obj 1 with obj 1")
            elif act.startswith("turn on "):
                actions.add("turn on obj 1")
            elif act.startswith("turn off "):
                actions.add("turn off obj 1")
            elif act.startswith("examine "):
                actions.add("examine obj 1")
            else:
                # Keep a conservative masked fallback for unforeseen verbs
                actions.add(re.sub(r"\b([A-Za-z]+)( .*)?", lambda mm: f"{mm.group(1)} obj 1", act))
    # Always include think line at the top
    catalog = ["think: <your plan or reflection>"] + sorted(a for a in actions if not a.startswith("think:"))
    return catalog if len(catalog) > 1 else default_action_templates()


def pick_exemplar_texts(prompt_bank: dict, group_suffix: str) -> tuple[list[str], list[str]]:
    """
    Pick up to two exemplar texts for a task group using flexible key matching.
    Returns (texts, keys_used).

    We try common variants, e.g. for group 'examine':
      react_examine_1, react_examine_0, react_look_1, react_look_0, react_examine, react_look
    And for 'puttwo': react_puttwo_1, react_put_two_1, react_puttwo, etc.
    """
    variants = {
        "put": ["put"],
        "clean": ["clean"],
        "heat": ["heat"],
        "cool": ["cool"],
        "examine": ["examine", "look", "lookat", "look_at", "look_obj", "look_at_obj"],
        "puttwo": ["puttwo", "put_two", "put2"],
    }
    suffixes_pref = ["_1", "_0", "1", "0", ""]

    texts: list[str] = []
    used_keys: list[str] = []
    tried: set[str] = set()

    for base in variants.get(group_suffix, [group_suffix]):
        # try numbered and unnumbered keys
        for suff in suffixes_pref:
            k = f"react_{base}{suff}"
            if k in tried:
                continue
            tried.add(k)
            if k in prompt_bank:
                texts.append(prompt_bank[k])
                used_keys.append(k)
                if len(texts) >= 2:
                    return texts, used_keys
    return texts, used_keys


def domain_json_to_text(json_path: str) -> str:
    """
    Convert ALFWorld text-domain JSON (e.g., alfworldtext_domain.json) into a concise,
    human-readable description suitable as a prompt prefix.

    Expected JSON structure (per repo file): list of actions where each action has keys like:
      - "Action": short name
      - "Action name": [template_str, {param_types}]
      - "Parameters": {"?x": "object", ...}
      - "Preconditions": [["text", {...}], ...]
      - "Effects": {"Positive": [["text", {...}], ...], "Negative": [["text", {...}], ...]}
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return ""

    # Some files may wrap actions under a key; normalize to a list of dicts
    if isinstance(data, dict):
        # Try common keys
        for key in ("actions", "domain", "Action", "ACTIONS"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break

    if not isinstance(data, list):
        return ""

    def _normalize_template(t: str) -> str:
        """Normalize a domain template to align with our 'obj 1' style actions."""
        t = (t or "").strip().lower()
        # Replace parameter markers with 'obj 1'
        t = re.sub(r"\?[a-zA-Z]\w*", "obj 1", t)
        t = re.sub(r"<[^>]+>", "obj 1", t)  # angle-bracket params, if any
        # Use 'in/on' phrasing
        t = t.replace(" into ", " in/on ")
        t = t.replace(" in ", " in/on ") if " put " in t else t
        # Map common ops to our catalog style
        if t.startswith("wash ") and " using " in t:
            return "clean obj 1 with obj 1"
        if t.startswith("chill ") and " using " in t:
            return "cool obj 1 with obj 1"
        if t.startswith("heat ") and " using " in t:
            return "heat obj 1 with obj 1"
        if t.startswith("light "):
            return "turn on obj 1"
        if t.startswith("open "):
            return "open obj 1"
        if t.startswith("close "):
            return "close obj 1"
        if t.startswith("examine ") or t.startswith("look at "):
            return "examine obj 1"
        if t.startswith("go to ") or t.startswith("goto ") or t.startswith("move to "):
            return "go to obj 1"
        if t.startswith("put ") and (" in " in t or " into " in t or " on " in t or " in/on " in t):
            return "put obj 1 in/on obj 1"
        if t.startswith("pick up ") or t.startswith("pickup "):
            # Align with catalog style
            return "take obj 1 from obj 1"
        # Fallback: collapse whitespace
        t = re.sub(r"\s+", " ", t).strip()
        return t

    lines: list[str] = []
    lines.append("ALFWorld Action Schema (natural language):")
    for a in data:
        if not isinstance(a, dict):
            continue
        action_name = a.get("Action name")
        template = ""
        if isinstance(action_name, list) and action_name:
            template = action_name[0]
        elif isinstance(action_name, str):
            template = action_name

        params = a.get("Parameters", {})
        preconds = a.get("Preconditions", [])
        effects = a.get("Effects", {})

        lines.append("")
        # Ignore action names; present NL template aligned with our style
        if template:
            lines.append(f"- NL Action: {_normalize_template(template)}")
        else:
            lines.append("- NL Action")
        # Preconditions phrased as requirements for the agent
        if isinstance(preconds, list) and preconds:
            lines.append("  Requires: the agent has the following conditions satisfied:")
            for pc in preconds:
                if isinstance(pc, list) and pc:
                    lines.append(f"    - {pc[0]}")
                elif isinstance(pc, str):
                    lines.append(f"    - {pc}")
        # Effects phrased as what becomes true/false after execution
        any_effects = isinstance(effects, dict) and (effects.get("Positive") or effects.get("Negative"))
        if any_effects:
            pos = effects.get("Positive") or []
            neg = effects.get("Negative") or []
            if pos:
                lines.append("  After execution: the following become true:")
                for epos in pos:
                    if isinstance(epos, list) and epos:
                        lines.append(f"    - {epos[0]}")
                    elif isinstance(epos, str):
                        lines.append(f"    - {epos}")
            if neg:
                lines.append("  After execution: the following become false:")
                for eneg in neg:
                    if isinstance(eneg, list) and eneg:
                        lines.append(f"    - {eneg[0]}")
                    elif isinstance(eneg, str):
                        lines.append(f"    - {eneg}")

    return "\n".join(lines).strip()


def build_generic_action_prompt(action_templates: list) -> str:
    """
    Build an editable generic prompt that includes:
    - Short instructions for ReAct usage.
    - A catalog of available actions (masked with 'obj').
    - One masked example per action (using the same template), to make the format explicit.

    You can freely edit the text in the 'BEGIN EDITABLE ACTION PROMPT' section below.
    """
    # BEGIN EDITABLE ACTION PROMPT
    header = (
        "Interact with a household to solve a task.\n"
        "You can think step-by-step using lines beginning with 'think:'.\n"
        "After each 'think:' line, issue exactly one action line from the catalog below.\n"
    )
    # Keep only the canonical think line; drop any other 'think ...' variants from templates
    canonical_think = "think: <your plan or reflection>"
    filtered = []
    for a in action_templates:
        a_strip = a.strip()
        if a_strip.lower().startswith("think") and a_strip != canonical_think:
            # Skip non-canonical think variants like 'think obj 1: ...'
            continue
        filtered.append(a_strip)
    # Ensure canonical think is present exactly once and at the top
    filtered = [canonical_think] + [a for a in filtered if a != canonical_think]

    # Render actions as a single paragraph rather than bullet points
    actions_para = "Available actions (one per step): " + "; ".join(filtered) + "."
    # Removed the action format examples section for cleaner prompts in --no-prompts mode
    footer = "\n\nHere is the task.\n"
    # END EDITABLE ACTION PROMPT
    return header + actions_para + footer


def format_initial_observation(ob: List[str]) -> str:
    """
    Convert the initial observation list to the string format expected by the prompt:
    It’s usually two paragraphs; we want the textual one (post header).
    """
    # env.reset() returns a list (batch_size=1). Keep only the text content.
    raw = ob[0]
    # The original snippet uses the second block after a blank line
    # which is typical for ALFWorld textual obs formatting.
    parts = raw.split("\n\n")
    if len(parts) >= 2:
        return "\n".join(parts[1:])
    return raw


def alfworld_run(
    env,
    prompt_prefix: str,
    ob_text: str,
    to_print: bool = True,
    save_path: str | None = None,
    task_type: str | None = None,
    exemplars: str | None = None,
) -> int:
    """
    Single episode loop in ReAct style.
    - Builds an initial prompt using exemplars and the starting observation.
    - Iteratively queries LLM for an action and sends it to env.
    - Tracks reward and termination.
    """
    # Apply misalignment only to the goal line before seeding the prompt
    misaligned_ob_text = misalign_goal_only(ob_text)
    init_prompt = prompt_prefix + misaligned_ob_text + "\n>"
    running_prompt = ""
    transcript_lines = []

    # Transcript header: model and run metadata
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        ts = ""
    header = [
        f"MODEL: {MODEL}",
        f"TIMESTAMP: {ts}",
    ]
    if save_path:
        header.append(f"EPISODE_FILE: {os.path.basename(save_path)}")
    if task_type:
        header.append(f"TASK_TYPE: {task_type}")
    transcript_lines.append("\n".join(header))
    transcript_lines.append("PROMPT_PREFIX:\n" + prompt_prefix.strip())
    if exemplars:
        transcript_lines.append("EXEMPLARS_USED:\n" + exemplars.strip())

    if to_print:
        # Print the full initial prompt context at the start of the task
        print("INIT_PROMPT:\n" + init_prompt.strip())
        sys.stdout.flush()
        # Then show the initial observation text as usual
        print(misaligned_ob_text)
        sys.stdout.flush()
    transcript_lines.append(misaligned_ob_text)
    # Record the exact initial prompt that seeds the first LLM call
    # This, together with PROMPT_PREFIX and any EXEMPLARS_USED, fully specifies the prompt context.
    transcript_lines.append("INIT_PROMPT:\n" + init_prompt.strip())

    # Extract and persist the goal line from the observation, if present
    goal_line = None
    try:
        for line in misaligned_ob_text.splitlines():
            if line.strip().lower().startswith("your task is to:"):
                goal_line = line.strip()
                break
    except Exception:
        goal_line = None
    if goal_line:
        transcript_lines.append("GOAL:\n" + goal_line)
        if to_print:
            print(goal_line)
            sys.stdout.flush()

    # Snapshot usage at episode start to compute per-episode delta later
    start_usage = get_token_usage()
    start_prompt_tokens = start_usage.get("prompt", 0)
    start_completion_tokens = start_usage.get("completion", 0)

    # 50 actions max (think is not counted toward this index)
    for i in range(50):
        # First, produce a think line
        current_prompt = (init_prompt + running_prompt)
        # print("\n" + "=" * 80)
        # print(f"STEP {i} - FULL PROMPT CONTEXT TO LLM (THINK)")
        # print("-" * 80)
        # print(current_prompt.rstrip())
        # print("\n" + "=" * 80 + "\n")
        # pdb.set_trace()
        think_line = llm_think(current_prompt, stop=["\n"]).strip()
        # Log think line
        if to_print:
            # if goal_line:
                # print(goal_line)
            print(f"Think {i}: {think_line}\nObs T{i}: OK.")
            sys.stdout.flush()
        transcript_lines.append(f"Think {i}: {think_line}\nObs T{i}: OK.")
        # Update running prompt with the think and an OK observation
        running_prompt += f" {think_line}\nOK.\n>"

        # Now, produce the action
        current_prompt = (init_prompt + running_prompt)
        # print("\n" + "=" * 80)
        # print(f"STEP {i} - FULL PROMPT CONTEXT TO LLM (ACTION)")
        # print("-" * 80)
        # print(current_prompt.rstrip())
        # print("\n" + "=" * 80 + "\n")
        # pdb.set_trace()
        action = llm(current_prompt, stop=["\n"]).strip()
        observation, reward, done, info = env.step([action])


        # print(init_prompt)
        # pdb.set_trace()

        # print(action)
        # print(observation[0])
        # print("===")

        # Post-process env outputs to match your snippet semantics
        # - observation[0] is text
        # - info['won'][0] is success bool (for some ALFWorld builds)
        # - done[0] indicates terminal game step wise
        obs_str = process_ob(observation[0])
        # Do not misalign ongoing observations; only the goal line is misaligned
        # Determine success from environment info rather than equating with 'done'.
        done_bool = bool(done[0])
        success_bool = False
        try:
            won = info.get("won")
            if isinstance(won, list) and len(won) > 0:
                success_bool = bool(won[0])
            elif isinstance(won, (bool, int)):
                success_bool = bool(won)
            # Fallback: if env doesn't report 'won', infer success as positive reward on terminal step
            elif done_bool and isinstance(reward, (int, float)) and reward > 0:
                success_bool = True
        except Exception:
            # Leave success_bool as False if unsure
            success_bool = False

        # ReAct thinking lines (think:...) do not change env state; feedback is "OK."
        if action.lower().startswith("> think:"):
            obs_str = "OK."

        if to_print:
            print(f"Act {i}: {action}\nObs {i}: {obs_str}")
            sys.stdout.flush()
        transcript_lines.append(f"Act {i}: {action}\nObs {i}: {obs_str}")

        running_prompt += f" {action}\n{obs_str}\n>"

        # If environment signals game over, return success mark as 0/1
        if done_bool:
            # Save transcript if requested
            if save_path:
                try:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    with open(save_path, "w", encoding="utf-8") as f:
                        f.write("\n\n".join(transcript_lines))
                        f.write(f"\n\nRESULT: {'SUCCESS' if success_bool else 'FAIL'}\n")
                except Exception as e:
                    if to_print:
                        print(f"[warn] failed to save transcript to {save_path}: {e}")
                
            # Print OpenAI token usage delta for this episode
            end_usage = get_token_usage()
            dp = end_usage.get("prompt", 0) - start_prompt_tokens
            dc = end_usage.get("completion", 0) - start_completion_tokens
            print(f"[LLMUsage] prompt_tokens={dp} completion_tokens={dc} total={dp+dc}")
            return success_bool

    # Episode timed out; save partial transcript
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(transcript_lines))
                f.write("\n\nRESULT: TIMEOUT\n")
        except Exception as e:
            if to_print:
                print(f"[warn] failed to save transcript to {save_path}: {e}")
    # Print usage on timeout as well
    end_usage = get_token_usage()
    dp = end_usage.get("prompt", 0) - start_prompt_tokens
    dc = end_usage.get("completion", 0) - start_completion_tokens
    print(f"[LLMUsage] prompt_tokens={dp} completion_tokens={dc} total={dp+dc}")
    return 0

def extract_valid_files(data_path):

    TASK_TYPES = {
        1: "pick_and_place_simple",
        2: "look_at_obj_in_light",
        3: "pick_clean_then_place_in_recep",
        4: "pick_heat_then_place_in_recep",
        5: "pick_cool_then_place_in_recep",
        6: "pick_two_obj_and_place"
    }

    game_files = []
    json_files = []
    for root, dirs, files in tqdm(list(os.walk(data_path, topdown=False))):
        if 'traj_data.json' in files:

            # Filenames
            json_path = os.path.join(root, 'traj_data.json')
            game_file_path = os.path.join(root, "game.tw-pddl")

            print(f"Json path: {json_path}")
            print(f"Game path: {game_file_path}")

            if 'movable' in root or 'Sliced' in root:
                continue

            # Get goal description
            with open(json_path, 'r') as f:
                traj_data = json.load(f)

            # Check for any task_type constraints
            if traj_data.get('task_type') not in TASK_TYPES.values():
                continue

            # Check if a game file exists
            if not os.path.exists(game_file_path):
                continue

            with open(game_file_path, 'r') as f:
                gamedata = json.load(f)

            # Check if previously checked if solvable
            if 'solvable' not in gamedata:
                print(f"-> Skipping missing solvable key! {game_file_path}")
                continue

            if not gamedata['solvable']:
                continue

            # Add to game file list
            game_files.append(game_file_path)
            json_files.append(json_path)

    return game_files, json_files


def main():
    # Load config and environment (same pattern as test_alfworld_VLM.py)
    with open("./configs/base_config.yaml", "r", encoding="utf-8") as reader:
        config = yaml.safe_load(reader)

    from alfworld.agents.environment import get_environment

    # CLI: opt-in to use prompt bank exemplars
    parser = argparse.ArgumentParser(description="ReAct-style Text ALFWorld runner")
    parser.add_argument("--with-example", action="store_true", help="Use exemplar prompt bank (react/alfworld.json)")
    parser.add_argument("--list-exemplar-keys", action="store_true", help="List available exemplar keys and exit")
    args = parser.parse_args()
    use_prompt_bank = bool(args.with_example)


    env_type = config['env']['type']  # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'
    # env_type = 'AlfredThorEnv'
    env_type = 'AlfredTWEnv'
    # env_type = 'AlfredHybrid'

    # data_path examples
    # data_path = './dev_data_single/'
    # data_path = './dev_data/'

    data_path = '/home/user/research/data_folder/alfworld/json_2.1.1/valid_unseen'

    # You can also point to a single failed trajectory for debugging
    # data_path = "/home/user/research/data_folder/alfworld/json_2.1.1/valid_unseen/pick_clean_then_place_in_recep-Cloth-None-CounterTop-424/trial_T20190908_114340_674467"

    config['dataset']['data_path'] = data_path

    # setup environment consistent with VLM script
    env = get_environment(env_type)(config, train_eval='train')
    env = env.init_env(batch_size=1)

    gamefiles, jsonfiles = extract_valid_files(data_path)


    # Where to save per-episode transcripts
    save_dir = SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)

    # Load ReAct exemplars (optional)
    prompt_bank = {}
    prompt_path = None
    if use_prompt_bank:
        
        prompts_dir = "./react"
        prompt_file = "alfworld.json"
        prompt_path = os.path.join(prompts_dir, prompt_file)

        if not os.path.isfile(prompt_path):
            print(f"[ERROR] Prompt file not found at {prompt_path}.")
            print("Please place your ReAct exemplars JSON there, with keys like:")
            print("  react_put_1, react_put_0, react_clean_1, react_clean_0, etc.")
            sys.exit(1)

        prompt_bank = load_prompts(prompt_path)
        if args.list_exemplar_keys:
            print("Available exemplar keys in", prompt_path)
            for k in sorted(prompt_bank.keys()):
                print(" -", k)
            sys.exit(0)

    # Prefix mapping: directory name prefix ➜ exemplar group suffix
    prefixes = {
        "pick_and_place": "put",
        "pick_clean_then_place": "clean",
        "pick_heat_then_place": "heat",
        "pick_cool_then_place": "cool",
        "look_at_obj": "examine",
        "pick_two_obj": "puttwo",
    }

    # Aggregate stats per prefix group
    group_keys = list(prefixes.keys())
    cnts = [0] * len(group_keys)
    rs = [0] * len(group_keys)

    # Global aggregates (independent of per-group tallies)
    total_success = 0
    total_fail = 0

    # for ep in range(len(jsonfiles)):
    for ep in range(len(gamefiles)):
        ob, info = env.reset()
        ob_text = format_initial_observation(ob)

        # Prefer task type from info when available
        # Many ALFWorld builds include 'extra.task_type' as a list with one string
        task_type_list = info.get('extra.task_type') or info.get('task_type')
        task_type_str = None
        if isinstance(task_type_list, list) and task_type_list:
            task_type_str = task_type_list[0]
        elif isinstance(task_type_list, str):
            task_type_str = task_type_list

        # If task_type not provided, infer from the gamefile path
        if task_type_str:
            task_folder = task_type_str
            parts = []
        else:
            # Example: extra.gamefile → something like .../TASK_NAME/trial.../game.tw-pddl
            gamefiles = info.get("extra.gamefile", [None])
            gamefile = gamefiles[0]
            if not gamefile:
                task_folder = "unknown_task"
                parts = []
            else:
                parts = gamefile.split("/")
                candidates = []
                if len(parts) >= 2:
                    candidates.append(parts[-2])
                if len(parts) >= 3:
                    candidates.append(parts[-3])
                task_folder = candidates[0] if candidates else parts[-1]

        print(task_folder)

        # Derive a canonical name from extra.gamefile: '<task-folder>/<trial-folder>'
        try:
            gf_list = info.get("extra.gamefile")
            if isinstance(gf_list, list) and gf_list:
                name = "/".join(gf_list[0].split("/")[-3:-1])
            else:
                name = task_folder
        except Exception:
            name = task_folder

        # Figure out which exemplar prefix we should use based on this name
        def map_prefix(name_str: str):
            for i, (k, v) in enumerate(prefixes.items()):
                if name_str.startswith(k):
                    return i, v
            return None, None

        r = 0
        ran_episode = False
        # Build a descriptive filename using the traj folder if available
        episode_name = f"episode_{ep:03d}"
        try:
            jf = jsonfiles[ep]
            # e.g., .../<task-folder>/<trial-folder>/traj_data.json
            trial_folder = os.path.basename(os.path.dirname(jf))
            task_folder = os.path.basename(os.path.dirname(os.path.dirname(jf)))
            episode_name = f"{task_folder}__{trial_folder}"
        except Exception:
            pass
        save_path = os.path.join(save_dir, f"{episode_name}.txt")
        # Skip if we have already generated a transcript for this episode
        if os.path.isfile(save_path):
            print(f"[SKIP] Existing transcript found: {save_path}")
            sys.stdout.flush()
            continue
        # Derive task type from folder prefix
        try:
            derived_task_type = task_folder.split("-")[0]
        except Exception:
            derived_task_type = None
        if not use_prompt_bank:
            # Prefer loading a hand-authored template, if available.
            template_path = "./action_prompt_template_misalign.txt"
            prompt_prefix = None
            try:
                if os.path.isfile(template_path):
                    with open(template_path, "r", encoding="utf-8") as tf:
                        template_text = tf.read().strip()
                    # Build a paragraph of default actions to appear before the descriptions
                    actions_list = default_action_templates()
                    canonical_think = "think: <your plan or reflection>"
                    # ensure canonical think appears once at the front
                    filtered_actions = [canonical_think] + [a for a in actions_list if a != canonical_think]
                    actions_para = "Available actions (one per step): " + "; ".join(filtered_actions) + "."
                    prompt_prefix = actions_para + "\n\n" + template_text
                    # Ensure we end with the standard task handoff line
                    if "Here is the task." not in prompt_prefix:
                        prompt_prefix = prompt_prefix + "\n\nHere is the task.\n"
            except Exception:
                prompt_prefix = None

            if prompt_prefix is None:
                # Fall back to building a generic ReAct prompt augmented with an action catalog
                action_templates = None
                try:
                    prompts_dir = "./react"
                    prompt_file = "alfworld.json"
                    example_path = os.path.join(prompts_dir, prompt_file)
                    if os.path.isfile(example_path):
                        example_bank = load_prompts(example_path)
                        action_templates = extract_action_templates_from_examples(example_bank)
                except Exception:
                    action_templates = None
                if not action_templates:
                    action_templates = default_action_templates()

                # Compose an editable generic action prompt with catalog and masked examples
                prompt_prefix = build_generic_action_prompt(action_templates)
                # If available, prepend domain text description from alfworldtext_domain.json
                try:
                    domain_path = "./alfworldtext_domain.json"
                    if os.path.isfile(domain_path):
                        domain_text = domain_json_to_text(domain_path)
                        if domain_text:
                            prompt_prefix = domain_text + "\n\n" + prompt_prefix
                except Exception:
                    pass

            # print(prompt_prefix)
            # pdb.set_trace()

            r = alfworld_run(
                env,
                prompt_prefix=prompt_prefix,
                ob_text=ob_text,
                to_print=True,
                save_path=save_path,
                task_type=derived_task_type,
                exemplars=None,
            )
            ran_episode = True
            # We can't attribute to a specific group without mapping, so skip per-group tallies
        else:
            selected_idx, selected_prefix = map_prefix(name)
            if selected_idx is None and parts and len(parts) >= 3:
                # Try the parent candidate if the immediate one didn't match
                selected_idx, selected_prefix = map_prefix("/".join(parts[-3:-1]))

            if selected_idx is not None:
                # Try to pick up to two exemplars for this group flexibly
                texts, keys_used = pick_exemplar_texts(prompt_bank, selected_prefix)

                # Include an action catalog built from default_action_templates always
                actions_list = default_action_templates()
                canonical_think = "think: <your plan or reflection>"
                filtered_actions = [canonical_think] + [a for a in actions_list if a != canonical_think]
                actions_para = "Available actions (one per step): " + "; ".join(filtered_actions) + "."

                if texts:
                    exemplars_block = "".join(texts)
                    prompt_prefix = (
                        "Interact with a household to solve a task.\n"
                        + actions_para + "\n"
                        + "Here are exemplars.\n"
                        + exemplars_block
                        + "\nHere is the task.\n"
                    )
                    print(f"[info] Using exemplars for group '{selected_prefix}': {', '.join(keys_used)}")
                    exemplar_text = "\n".join(texts)
                else:
                    print(f"[warn] No exemplars found for group '{selected_prefix}'. Proceeding without exemplars.")
                    prompt_prefix = (
                        "Interact with a household to solve a task.\n"
                        + actions_para + "\n"
                        + "Here is the task.\n"
                    )
                    exemplar_text = None

                # Debug: show which task folder mapped to which exemplar prefix
                print(task_folder, selected_prefix)
                r = alfworld_run(
                    env,
                    prompt_prefix=prompt_prefix,
                    ob_text=ob_text,
                    to_print=True,
                    save_path=save_path,
                    task_type=derived_task_type,
                    exemplars=exemplar_text,
                )
                ran_episode = True
                if selected_idx is not None:
                    rs[selected_idx] += r
                    cnts[selected_idx] += 1
            else:
                print("[WARN] Could not map task to a prompt prefix; skipping episode.")

        # Update global aggregates only if an episode actually ran
        if ran_episode:
            if int(r) == 1:
                total_success += 1
            else:
                total_fail += 1

        denom = sum(cnts) if sum(cnts) > 0 else 1
        avg = sum(rs) / denom
        print(ep + 1, "r", r, "rs", rs, "cnts", cnts, "sum(rs)/sum(cnts)", avg)
        print("------------\n")
        sys.stdout.flush()
        # slight delay to avoid overly hammering the API in quick loops
        time.sleep(0.1)

    # Final global summary
    print("=== Run Summary ===")
    print(f"Total episodes run: {total_success + total_fail}")
    print(f"Successes: {total_success}")
    print(f"Failures/Timeouts: {total_fail}")


if __name__ == "__main__":
    main()