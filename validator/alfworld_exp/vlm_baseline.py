import base64
import json
import os
import time
import cv2
import numpy as np
from google import genai
from google.genai import types
from openai import OpenAI
import yaml
import logging

from tqdm import tqdm

from alfworld.agents.environment import get_environment

log_dirname = f'log_{time.time()}'
os.makedirs(log_dirname)
logging.basicConfig(
    filename=os.path.join(log_dirname, 'exp.log'),              # log file name
    level=logging.INFO,              # minimum level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s [%(levelname)s] %(message)s",  # log format
    datefmt="%Y-%m-%d %H:%M:%S"      # time format
)

json_schema_openai = {
    "name": "ThoughtActionResult",
        "schema": {
            "type": "object",
            "properties": {
                "thought": {
                  "type": "string",
                  "description": "Reasoning or explanation of the step."
                },
                "action": {
                  "type": "string",
                  "description": "Action to take based on the thought."
                }
              },
            "required": ["thought", "action"],
            "additionalProperties": False
        }
    }

json_schema_gemini = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ThoughtActionResult",
    "type": "object",
    "properties": {
        "thought": {
          "type": "string",
          "description": "Reasoning or explanation of the step."
        },
        "action": {
          "type": "string",
          "description": "Action to take based on the thought."
        }
      },
    "required": ["thought", "action"],
    "additionalProperties": False
}


default_action_templates = [
    "go to <obj> <num>",
    "open <obj> <num>",
    "close <obj> <num>",
    "take <obj> <num> from <obj> <num>",
    "put <obj> <num> in/on <obj> <num>",
    "clean <obj> <num> with <obj> <num>",
    "heat <obj> <num> with <obj> <num>",
    "cool <obj> <num> with <obj> <num>",
    "turn on <obj> <num>",
    "turn off <obj> <num>",
]


openai_client = OpenAI()
gemini_client = genai.Client()
qwen_client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

# Optional hand-authored action prompt template to include in per-step task prompts
# Override path via env var ACTION_PROMPT_TEMPLATE_PATH
ACTION_PROMPT_TEMPLATE_PATH = os.environ.get("ACTION_PROMPT_TEMPLATE_PATH", "./action_prompt_template_misalign.txt")
try:
    with open(ACTION_PROMPT_TEMPLATE_PATH, "r", encoding="utf-8") as _tf:
        ACTION_PROMPT_TEMPLATE = _tf.read().strip()
except Exception:
    ACTION_PROMPT_TEMPLATE = ""

system_prompt = f"""You are an AI agent that solves tasks using the ReAct (Reasoning + Acting) pattern. 
You interact with a household environment and can observe images to help solve tasks.

Always follow this strict format for each step:

Thought: <your reasoning about the task based on current observations>
Action: <action_name>

The valid actions are {', '.join(default_action_templates)} where <obj> is an object that is presented in the image and <num> is an integer.
Valid <obj> are AlarmClock, Apple, AppleSliced, BaseballBat, BasketBall, Book, Bowl, Box, Bread, BreadSliced, ButterKnife, 
CD, Candle, CellPhone, Cloth, CreditCard, Cup, DeskLamp, DishSponge, Egg, Faucet, FloorLamp, Fork, Glassbottle, HandTowel, HousePlant, 
Kettle, KeyChain, Knife, Ladle, Laptop, LaundryHamperLid, Lettuce, LettuceSliced, LightSwitch, Mug, Newspaper, Pan, PaperTowel, 
PaperTowelRoll, Pen, Pencil, PepperShaker, Pillow, Plate, Plunger, Pot, Potato, PotatoSliced, RemoteControl, SaltShaker, ScrubBrush, 
ShowerDoor, SoapBar, SoapBottle, Spatula, Spoon, SprayBottle, Statue, StoveKnob, TeddyBear, Television, TennisRacket, TissueBox, ToiletPaper, 
ToiletPaperRoll, Tomato, TomatoSliced, Towel, Vase, Watch, WateringCan, WineBottle,

Rules:
1. Only produce one Thought and one Action per step.
2. Do NOT provide Observations or Final Answers yet.
3. Use clear, concise reasoning in Thought.
4. Use valid action names and provide inputs appropriate for the action.
5. Wait for the environment to return the Observation before continuing to the next step.
6. MUST ensure <obj> and <num> are valid
"""

system_prompt = system_prompt.lower()

def process_ob(ob: str) -> str:
    """Clean up ALFWorld obs prefix for readability (as in your snippet)."""
    if ob.startswith("You arrive at loc "):
        ob = ob[ob.find(". ") + 2 :]
    return ob

def numpy_to_base64(img: np.ndarray, format: str = ".png") -> str:
    # Encode NumPy array to image buffer in memory
    success, buffer = cv2.imencode(format, img)
    if not success:
        raise ValueError("Could not encode image")

    # Convert to base64 string
    base64_str = base64.b64encode(buffer).decode("utf-8")
    return base64_str

def numpy_to_bytes(image_array, format=".png"):
    success, buffer = cv2.imencode(format, image_array)
    if not success:
        raise ValueError("Could not encode image")
    return buffer.tobytes()

def llm_gpt(text_prompt, image_prompt):
    messages = [{"role": "system", "content": system_prompt}]
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": text_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{numpy_to_base64(image_prompt)}"}}
        ],
    })
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",   # replace with desired model
        messages=messages,
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        response_format={"type": "json_schema", "json_schema": json_schema_openai},
    )
    return json.loads(response.choices[0].message.content)

def llm_qwen(text_prompt, image_prompt):
    messages = [{"role": "system", "content": system_prompt}]
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": text_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{numpy_to_base64(image_prompt)}"}}
        ],
    })
    response = qwen_client.chat.completions.create(
        model="qwen2.5-vl-72b-instruct",   # replace with desired model
        messages=messages,
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        response_format={"type": "json_schema", "json_schema": json_schema_openai},
    )
    return json.loads(response.choices[0].message.content)

def llm_gemini(text_prompt, image_prompt):
    response = gemini_client.models.generate_content(
        # model='gemini-2.5-pro',
        model='gemini-2.0-flash',
        contents=[
            system_prompt,
            text_prompt,
            types.Part.from_bytes(data=numpy_to_bytes(image_prompt), mime_type='image/png'),
        ],
        config={
            'response_mime_type': 'application/json',
            'response_json_schema': json_schema_gemini,
            'temperature': 0.
        },
    )
    logging.info(str(response.parsed))
    return response.parsed

def llm_gemini_text(text_prompt):
    """Text-only variant returning parsed JSON according to json_schema_gemini."""
    response = gemini_client.models.generate_content(
        model='gemini-2.0-flash',
        contents=[
            system_prompt,
            text_prompt,
        ],
        config={
            'response_mime_type': 'application/json',
            'response_json_schema': json_schema_gemini,
            'temperature': 0.
        },
    )
    logging.info(str(response.parsed))
    return response.parsed

with open('configs/base_config.yaml') as reader:
    config = yaml.safe_load(reader)

env_type = 'AlfredThorEnv'
#env_type = 'AlfredTWEnv'
# env_type = 'AlfredHybrid'

config['dataset']['data_path'] = '/home/user/research/data_folder/alfworld/json_2.1.1/valid_unseen'

# config['dataset']['data_path'] = 'dev_data'

# Explicitly disable expert policy/advice to prevent crashes in expert handlers
try:
    config.setdefault('expert', {})
    config['expert']['use'] = False
    # Some builds use nested subgoal flags
    if isinstance(config['expert'], dict):
        config['expert'].setdefault('subgoal', {})
        if isinstance(config['expert']['subgoal'], dict):
            config['expert']['subgoal']['use'] = False
    # Some builds gate advice via model/advice flags
    config.setdefault('model', {})
    if isinstance(config['model'], dict):
        config['model'].setdefault('advice', {})
        if isinstance(config['model']['advice'], dict):
            config['model']['advice']['use'] = False
except Exception:
    pass

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

env = get_environment(env_type)(config, train_eval='train')
env = env.init_env(batch_size=1)

# Use the same dataset path configured for the environment
data_path = config['dataset']['data_path']
gamefiles, jsonfiles = extract_valid_files(data_path)

# Best-effort: also disable any live expert hooks on the environment object
for attr in ('expert', 'use_expert', 'enable_expert'):
    if hasattr(env, attr):
        try:
            if attr == 'expert':
                setattr(env, attr, None)
            else:
                setattr(env, attr, False)
        except Exception:
            pass

# Figure out which exemplar prefix we should use based on the task type/folder
def map_prefix(folder: str):
    for i, (k, v) in enumerate(prefixes.items()):
        if folder.startswith(k):
            return i, v
    return None, None


# --- New helpers: visible object parsing and action refinement ---
def parse_visible_objects_from_text(ob_text: str) -> list:
    """Extract a list like ["fridge 1", "cabinet 1", ...] from the textual observation.

    Heuristic: find the sentence fragment after 'you see' up to the next period, then split by commas/and.
    Returns all tokens in lowercase.
    """
    text = ob_text.strip()
    lower = text.lower()
    start_key = "you see "
    if start_key not in lower:
        return []
    start = lower.find(start_key) + len(start_key)
    # take until the next period after the section begins
    end = lower.find(".", start)
    if end == -1:
        end = len(lower)
    segment = lower[start:end].strip()

    # Split by commas and ' and '
    parts = []
    for chunk in segment.split(","):
        chunk = chunk.strip()
        # further split ' and '
        if " and " in chunk:
            parts.extend([p.strip() for p in chunk.split(" and ") if p.strip()])
        else:
            if chunk:
                parts.append(chunk)

    # Remove leading articles and keep the object name with index
    objects = []
    for p in parts:
        # remove common leading articles
        if p.startswith("and "):
            p = p[4:]
        if p.startswith("a "):
            p = p[2:]
        elif p.startswith("an "):
            p = p[3:]
        elif p.startswith("the "):
            p = p[4:]
        p = p.strip()
        # keep only tokens that look like 'name number'
        # allow alphanum names with digits at end (e.g., 'drawer 10')
        if p and any(ch.isdigit() for ch in p.split()[-1:][0]):
            objects.append(p)
    # de-dup preserving order
    seen = set()
    out = []
    for o in objects:
        if o not in seen:
            seen.add(o)
            out.append(o)
    return out


def refine_action_with_visible_objects(previous_action: str, visible_objects: list, task_prompt: str,
                                       initial_ob_text: str, actions_taken: list) -> str:
    """
    Re-prompt the LLM with the last proposed action and constrain object choices to the provided visible list.
    Returns a new action string (lowercased) that should select only from visible_objects.
    """
    # Build a strict instruction emphasizing object selection from the list only.
    visible_block = "\n".join(f"- {o}" for o in visible_objects) if visible_objects else "- (none detected)"
    actions_list_block = "\n".join(f"- {a}" for a in actions_taken) if actions_taken else "- None yet"
    refine_prompt = (
        "You previously proposed an action that may reference objects not currently visible.\n"
        "You must choose objects ONLY from the visible list below. Keep the same intent as the previous action.\n"
        "CRITICAL: Keep the action name and structure IDENTICAL to the previous action (same verb and prepositions).\n"
        "Only replace the object name-number pairs with ones from the visible list.\n\n"
        # f"Task:\n{task_prompt}\n\n"
        # f"Initial observation:\n{initial_ob_text}\n\n"
        # f"Actions taken so far:\n{actions_list_block}\n\n"
        f"Visible objects now:\n{visible_block}\n\n"
        f"Previous action: {previous_action}\n\n"
        "Respond strictly in the required JSON format with a Thought plus a corrected Action using only the visible objects, "
        "keeping the exact same action name and prepositions."
    )
    try:
        result = llm_gemini_text(refine_prompt)
        new_action = result.get("action", previous_action).lower()
    except Exception:
        # If the refinement fails (API, network), fall back to previous action
        print("Refinement failed, falling back to previous action")
        new_action = previous_action.lower()
    return new_action

def alfworld_run(ep, task_prompt, ob_image, initial_ob_text, to_print=True):
    text_prompt = task_prompt

    # Track and show initial observation and the actions taken
    actions_taken = []
    if to_print:
        print("INITIAL OBSERVATION:\n" + initial_ob_text)
        print("\n--- Begin Episode Actions ---\n")
    logging.info("INITIAL OBSERVATION:\n" + initial_ob_text)

    # Extract navigable places from the initial observation once and surface to the model every step
    initial_places = parse_visible_objects_from_text(initial_ob_text)
    places_block = ("\n".join(f"- {p}" for p in initial_places)) if initial_places else "- (none parsed)"

    last_ob_text = initial_ob_text
    for i in range(1, 50):
        # cv2.imwrite(f'{log_dirname}/ep_{ep}_frame_{i}.png', ob_image)
        # Compose a richer text prompt including initial observation and actions taken so far
        actions_list_block = ("\n".join(f"- {a}" for a in actions_taken)) if actions_taken else "- None yet"
        composed_prompt_core = (
            f"Task:\n{task_prompt}\n\n"
            f"Initial observation:\n{initial_ob_text}\n\n"
            f"Places you can go (from initial observation) in the format of <obj> <num>:\n{places_block}\n\n"
            f"Actions taken so far:\n{actions_list_block}\n\n"
            "Provide the next step strictly as Thought and Action per the required format. "
            "If you choose a navigation action ('go to <obj> <num>'), you MUST choose <obj> <num> only from the Places list above."
        )
        composed_prompt = ("Here is a description of how your action impact the enviorment:\n" + ACTION_PROMPT_TEMPLATE + "\n\n" + composed_prompt_core) if ACTION_PROMPT_TEMPLATE else composed_prompt_core
        proposed_action = llm_gpt(composed_prompt, ob_image)["action"].lower()
        # Visible list from the most recent textual observation
        visible_now = parse_visible_objects_from_text(last_ob_text)
        # Only refine non-navigation actions; keep 'go to ...' as-is
        if proposed_action.startswith("go to "):
            action = proposed_action
        else:
            action = refine_action_with_visible_objects(
                previous_action=proposed_action,
                visible_objects=visible_now,
                task_prompt=task_prompt,
                initial_ob_text=initial_ob_text,
                actions_taken=actions_taken,
            )

        print('proposed_action', proposed_action)
        print('last_ob_text', last_ob_text)
        print('visible_now', visible_now)
        print('action', action)
        print('-------------------------')
        observation, reward, done, info = env.step([action])
        # print(action)
        # print(observation)
        # print('-------------------------')
        actions_taken.append(action)
        obs_str = process_ob(observation[0])
        success_bool = int(info.get("won", [0])[0])  # 0/1
        done_bool = bool(done[0])
        if to_print:
            logging.info(f"Admissible Act: {info['admissible_commands']}")
            logging.info(f"Act {i}: {action} Obs {i}: {obs_str}")
        if done_bool:
            # At episode end, summarize the actions taken
            summary = "\n".join(f"{idx+1}. {a}" for idx, a in enumerate(actions_taken))
            if to_print:
                print("\n--- Actions Taken ---")
                print(summary)
                print("--- End Episode ---\n")
            logging.info("\nACTIONS TAKEN:\n" + summary)
            return success_bool
        # Update for next step
        last_ob_text = observation[0]
        ob_image = env.get_frames()[0]
    # Timeout: also summarize
    summary = "\n".join(f"{idx+1}. {a}" for idx, a in enumerate(actions_taken))
    if to_print:
        print("\n--- Actions Taken (timeout) ---")
        print(summary)
        print("--- End Episode ---\n")
    logging.info("\nACTIONS TAKEN (timeout):\n" + summary)
    return 0

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

for ep in range(len(jsonfiles)):
    ob, info = env.reset()
    initial_ob = ob[0]
    # Best-effort: log the gamefile path if available
    try:
        json_file = (info.get('extra.gamefile') or [None])[0]
        if json_file:
            logging.info(f"Task gamefile: {json_file}")
    except Exception:
        pass

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

    selected_idx, selected_prefix = map_prefix(task_folder)
    if selected_idx is None and parts and len(parts) >= 3:
        # Try the parent candidate if the immediate one didn't match
        selected_idx, selected_prefix = map_prefix(parts[-3])
    # Default result value in case episode is skipped
    r = 0
    if selected_idx is not None:
        # Debug: show which task folder mapped to which exemplar prefix
        logging.info(f"Task folder: {task_folder}")
        logging.info(f"Selected prefix: {selected_prefix}")
        task_prompt = '\n'.join(ob[0].split('\n\n')[2:])
        logging.info(f"Task prompt: {task_prompt}")
        frame = env.get_frames()[0]
        try:
            r = alfworld_run(ep, task_prompt, frame, initial_ob)
            rs[selected_idx] += r
            cnts[selected_idx] += 1
        except Exception as e:
            logging.exception(f"[EP {ep}] Episode failed with an exception: {e}")
            r = 0
    else:
        logging.info("[WARN] Could not map task to a prompt prefix; skipping episode.")

    denom = sum(cnts)
    avg = (sum(rs) / denom) if denom > 0 else 0.0
    logging.info(f"{ep + 1}, r, {r}, rs, {rs}, cnts, {cnts}, sum(rs)/sum(cnts), {avg}")
    logging.info("------------")