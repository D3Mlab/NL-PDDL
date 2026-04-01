from typing import Any, Dict, List, Set, Tuple, Optional
from pddl.logic import Predicate, variables

from pathlib import Path

from openai import OpenAI

import re

from PIL import Image
import io
import base64

import json
import random

import re
import numpy as np

import ast

import pdb

class ALFWorldProblemParser:
    """
    Parses a structured 'goal' dict into:
      - PDDL variables (self.vars)
      - positive literals (self.pos_set)
      - negative literals (self.neg_set)
    and can render them to PDDL text.
    """

    def __init__(self, save_path: str = None):
 
        if save_path is None:
            self.save_path = "./"

    def parse_nl_pddl_problem(self, goal: dict, task_type: str, save_name=None) -> str:

        if save_name is None:
            save_name = "generated_alfworld_problem.pddl"

        goal_object = goal["object_target"].lower()
        goal_receptacle  = goal["parent_target"].lower()
        toggle_obj = goal.get("toggle_target") .lower()

        # If you later want to rename these, update this map:
        var_names = {
            "goal_obj":    "goal_obj",
            "goal_recep":  "goal_recep",
            "goal_obj_2":  "goal_obj_2",
            "goal_recep_2":"goal_recep_2",
            "toggle_obj":  "toggle_obj"
        }

        NL_goal = []

        hand_empty_goal = ["the agent's hand is empty", {}]

        goal_obj_type = [
            f"{var_names['goal_obj']} is a {goal_object}",
            {var_names["goal_obj"]: "object"}
        ]

        goal_receptacle_type = [
            f"{var_names['goal_recep']} is a {goal_receptacle}",
            {var_names["goal_recep"]: "object"}
        ]

        toggle_obj_type = [
            f"{var_names['toggle_obj']} is a {toggle_obj}",
            {var_names["toggle_obj"]: "object"}
        ]

        inside_predicate = [
            f"{var_names['goal_obj']} is inside {var_names['goal_recep']}",
            {var_names["goal_obj"]: "object", var_names["goal_recep"]: "object"}
        ]

        turn_on_predicate = [
            f"{var_names['toggle_obj']} is turned on",
            {var_names["toggle_obj"]: "object"}
        ]

        light_predicate = [
            f"{var_names['goal_obj']} is under light {var_names['toggle_obj']}",
            {var_names["goal_obj"]: "object", var_names["toggle_obj"]: "object"}
        ]

        heat_predicate = [
            f"{var_names['goal_obj']} is hot",
            {var_names["goal_obj"]: "object"}
        ]

        clean_predicate = [
            f"{var_names['goal_obj']} is clean",
            {var_names["goal_obj"]: "object"}
        ]

        cool_predicate = [
            f"{var_names['goal_obj']} is cool",
            {var_names["goal_obj"]: "object"}
        ]
        holding_predicate = [
            f"the agent is holding {var_names['goal_obj']}",
            {var_names["goal_obj"]: "object"}
        ]


        

        if task_type == "pick_and_place_simple" or task_type == "pick_two_obj_and_place":
            NL_goal.append(hand_empty_goal)
            NL_goal.append(inside_predicate)
            NL_goal.append(goal_obj_type)
            NL_goal.append(goal_receptacle_type)
        elif task_type == "pick_heat_then_place_in_recep":
            NL_goal.append(hand_empty_goal)
            NL_goal.append(inside_predicate)
            NL_goal.append(goal_obj_type)
            NL_goal.append(goal_receptacle_type)
            NL_goal.append(heat_predicate)
        elif task_type == "pick_clean_then_place_in_recep":
            NL_goal.append(hand_empty_goal)
            NL_goal.append(inside_predicate)
            NL_goal.append(goal_obj_type)
            NL_goal.append(goal_receptacle_type)
            NL_goal.append(clean_predicate)
        elif task_type == "pick_cool_then_place_in_recep":
            NL_goal.append(hand_empty_goal)
            NL_goal.append(inside_predicate)
            NL_goal.append(goal_obj_type)
            NL_goal.append(goal_receptacle_type)
            NL_goal.append(cool_predicate)
        elif task_type == "pick_light_then_place_in_recep" :
            NL_goal.append(inside_predicate)
            NL_goal.append(goal_obj_type)
            NL_goal.append(goal_receptacle_type)
            NL_goal.append(light_predicate)
        elif task_type == "look_at_obj_in_light":
            NL_goal.append(goal_obj_type)
            NL_goal.append(light_predicate)
            NL_goal.append(toggle_obj_type)
        else:
            raise ValueError(f"Unsupported task_type: {task_type!r}")

        return NL_goal

    def parse_pddl_problem(self, goal: dict, task_type: str, save_name=None) -> str:

        if save_name is None:
            save_name = "generated_alfworld_problem.pddl"

        obj        = goal["object_target"]
        parent     = goal["parent_target"]
        toggle_obj = goal.get("toggle_target")

        # If you later want to rename these, update this map:
        var_names = {
            "goal_obj":    "goal_obj",
            "goal_recep":  "goal_recep",
            "goal_obj_2":  "goal_obj_2",
            "goal_recep_2":"goal_recep_2",
            "toggle_obj":  "toggle_obj"
        }

        # Define your goal templates here:
        GOAL_TEMPLATES = {
            "pick_and_place_simple": {
                "pos": [
                    "(handEmpty)",
                    f"(is{obj} {var_names['goal_obj']})",
                    f"(is{parent} {var_names['goal_recep']})",
                    f"(inReceptacle {var_names['goal_obj']} {var_names['goal_recep']})",
                ],
                "neg": []
            },
            "pick_two_obj_and_place": {
                "pos": [
                    "(handEmpty)",
                    f"(is{obj} {var_names['goal_obj']})",
                    f"(is{obj} {var_names['goal_obj_2']})",
                    f"(is{parent} {var_names['goal_recep']})",
                    f"(is{parent} {var_names['goal_recep_2']})",
                    f"(inReceptacle {var_names['goal_obj']} {var_names['goal_recep']})",
                ],
                "neg": []
            },
            "pick_heat_then_place_in_recep": {
                "pos": [
                    "(handEmpty)",
                    f"(is{obj} {var_names['goal_obj']})",
                    f"(is{parent} {var_names['goal_recep']})",
                    f"(isHot {var_names['goal_obj']})",
                    f"(inReceptacle {var_names['goal_obj']} {var_names['goal_recep']})",
                ],
                "neg": [] 
            },
            "pick_cool_then_place_in_recep": {
                "pos": [
                    "(handEmpty)",
                    f"(is{obj} {var_names['goal_obj']})",
                    f"(is{parent} {var_names['goal_recep']})",
                    f"(isCool {var_names['goal_obj']})",
                    f"(inReceptacle {var_names['goal_obj']} {var_names['goal_recep']})",
                ],
                "neg": []
            },
            "pick_clean_then_place_in_recep": {
                "pos": [
                    "(handEmpty)",
                    f"(is{obj} {var_names['goal_obj']})",
                    f"(is{parent} {var_names['goal_recep']})",
                    f"(isClean {var_names['goal_obj']})",
                    f"(inReceptacle {var_names['goal_obj']} {var_names['goal_recep']})",
                ],
                "neg": []
            },
            "look_at_obj_in_light": {
                "pos": [
                    f"(holds {var_names['goal_obj']})",
                    "(handEmpty)",
                    f"(isLightOn {var_names['toggle_obj']})",
                ],
                "neg": []
            },
            # … add more task_types here …
        }

        spec = GOAL_TEMPLATES.get(task_type)
        if spec is None:
            raise ValueError(f"Unsupported task_type: {task_type!r}")

        # Build the conjunction body
        lines: List[str] = []
        for atom in spec["pos"]:
            lines.append(atom)
        for atom in spec["neg"]:
            lines.append(f"(not {atom})")

        body = "\n    ".join(lines)
        pddl_goal = f"(:goal (and\n    {body}\n))"

        objs_block = ""
        init_block = ""
        problem_name = "alfworld_task"
        domain_name = "alfworld_task"

        problem = f"(define (problem {problem_name}) (:domain {domain_name}) (:objects{objs_block}) (:init {init_block}) \n {pddl_goal})"

        base_dir = Path(self.save_path)
        file_path = base_dir / save_name
        with open(file_path, "w") as f:
            f.write(problem)

        return problem, str(file_path)


    def parse_goal_llm(
        self,
        task_type: str,
        goal_key: Optional[str] = None,
        examples_path: str = "nl_goal_examples.json",
        model: str = "gpt-4o",
        temperature: float = 0.2,
        max_tokens: int = 600,
        synonyms: Dict[str, str] | None = None,
    ) -> List[Any]:
        """
        Use an LLM to generate an NL goal JSON for the given task_type and (optionally) goal_key (task description),
        guided by up to two randomly sampled examples from the same task_type in nl_goal_examples.json.

        Returns:
            A Python object parsed from the JSON the model returns (expected to be a list of [str, {var_typing}] entries).
        """
        # Load examples
        ex_path = Path(examples_path)
        if not ex_path.exists():
            raise FileNotFoundError(f"Examples file not found: {examples_path}")
        with ex_path.open("r") as f:
            all_examples = json.load(f)

        bucket = all_examples.get(task_type)
        if not isinstance(bucket, dict) or not bucket:
            raise ValueError(f"No examples found for task_type '{task_type}' in {examples_path}")

        # Randomly select up to two different examples (exclude the current goal_key if present)
        candidate_keys = [k for k in bucket.keys() if (goal_key is None or k != goal_key)]
        if not candidate_keys:
            # fallback: allow using the only available key
            candidate_keys = list(bucket.keys())
        k = min(2, len(candidate_keys))
        shot_keys = random.sample(candidate_keys, k=k)

        # Build few-shot prompt
        shots_txt = []
        for i, sk in enumerate(shot_keys, start=1):
            try:
                nl_goal_ex = bucket[sk]
            except Exception:
                nl_goal_ex = []
            shots_txt.append(
                (
                    f"Example {i}:\n"
                    f"Task type: {task_type}\n"
                    f"Task description: {sk}\n"
                    f"NL goal JSON: {json.dumps(nl_goal_ex, ensure_ascii=False)}\n"
                )
            )

        system_msg = (
            "You convert ALFWorld task descriptions into a normalized NL goal JSON format. "
            "Output must be strictly valid JSON (no markdown, no code fences). "
            "The JSON must be a list. Each element must be a two-item list: "
            "[predicate_string, {variable_typing}]. Use the same variable names as examples: "
            "goal_obj, goal_recep, goal_obj_2, goal_recep_2, toggle_obj. Use typing 'object'."
        )

        user_msg = f"Task type: {task_type}\n"
        if shots_txt:
            user_msg += "\n".join(shots_txt) + "\n"
        if goal_key:
            user_msg += (
                "Now produce ONLY the NL goal JSON for this new task description.\n"
                f"Task description: {goal_key}\n"
            )
        else:
            user_msg += (
                "Now produce ONLY the canonical NL goal JSON that matches tasks of this type, "
                "based on the examples above.\n"
            )
        user_msg += "Respond with JSON only."

        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )

        content = resp.choices[0].message.content.strip()

        # Best-effort strip common code fences if any slipped through
        if content.startswith("```"):
            content = content.strip("`\n ")
            # remove possible language hint
            if "\n" in content:
                content = content.split("\n", 1)[1]
            content = content.strip("`\n ")

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            # Try to extract JSON substring heuristically
            start = content.find("[")
            end = content.rfind("]")
            if start != -1 and end != -1 and end > start:
                sub = content[start:end+1]
                parsed = json.loads(sub)
            else:
                raise ValueError(f"LLM returned non-JSON content: {content}") from e

        # Default synonyms mapping if none provided
        if synonyms is None:
            synonyms = {
                "sink": "sinkbasin",
                "refrigerator": "fridge",
                "refridgerator": "fridge",
                "lamp": "desklamp",   # default; can refine downstream
                "table lamp": "tablelamp",
                "light switch": "lightswitch",
            }

        parsed = self._apply_synonyms_to_nl_goal(parsed, synonyms)
        return parsed

    def _apply_synonyms_to_nl_goal(self, nl_goal: List[Any], synonyms: Dict[str, str]) -> List[Any]:
        """
        Apply a word-level synonym normalization to predicate strings in the NL goal list.
        Only predicate strings (index 0 of each [predicate, typing] pair) are transformed.
        """
        if not isinstance(nl_goal, list) or not synonyms:
            return nl_goal

        # Build a case-insensitive regex for the synonyms' keys
        keys = list(synonyms.keys())
        if not keys:
            return nl_goal
        pat = re.compile(r"\b(" + "|".join(map(re.escape, keys)) + r")\b", flags=re.IGNORECASE)

        out: List[Any] = []
        for item in nl_goal:
            if isinstance(item, list) and len(item) == 2 and isinstance(item[0], str):
                pred = item[0]
                # Substitute case-insensitively but map to canonical lowercase target
                def _repl(m):
                    return synonyms.get(m.group(0).lower(), m.group(0))
                pred_norm = pat.sub(_repl, pred)
                out.append([pred_norm, item[1]])
            else:
                out.append(item)
        return out

    def parse_pddl_problem(self, goal: dict, task_type: str, save_name=None) -> str:

        if save_name is None:
            save_name = "generated_alfworld_problem.pddl"

        obj        = goal["object_target"]
        parent     = goal["parent_target"]
        toggle_obj = goal.get("toggle_target")

        # If you later want to rename these, update this map:
        var_names = {
            "goal_obj":    "goal_obj",
            "goal_recep":  "goal_recep",
            "goal_obj_2":  "goal_obj_2",
            "goal_recep_2":"goal_recep_2",
            "toggle_obj":  "toggle_obj"
        }

        # Define your goal templates here:
        GOAL_TEMPLATES = {
            "pick_and_place_simple": {
                "pos": [
                    "(handEmpty)",
                    f"(is{obj} {var_names['goal_obj']})",
                    f"(is{parent} {var_names['goal_recep']})",
                    f"(inReceptacle {var_names['goal_obj']} {var_names['goal_recep']})",
                ],
                "neg": []
            },
            "pick_two_obj_and_place": {
                "pos": [
                    "(handEmpty)",
                    f"(is{obj} {var_names['goal_obj']})",
                    f"(is{obj} {var_names['goal_obj_2']})",
                    f"(is{parent} {var_names['goal_recep']})",
                    f"(is{parent} {var_names['goal_recep_2']})",
                    f"(inReceptacle {var_names['goal_obj']} {var_names['goal_recep']})",
                ],
                "neg": []
            },
            "pick_heat_then_place_in_recep": {
                "pos": [
                    "(handEmpty)",
                    f"(is{obj} {var_names['goal_obj']})",
                    f"(is{parent} {var_names['goal_recep']})",
                    f"(isHot {var_names['goal_obj']})",
                    f"(inReceptacle {var_names['goal_obj']} {var_names['goal_recep']})",
                ],
                "neg": [] 
            },
            "pick_cool_then_place_in_recep": {
                "pos": [
                    "(handEmpty)",
                    f"(is{obj} {var_names['goal_obj']})",
                    f"(is{parent} {var_names['goal_recep']})",
                    f"(isCool {var_names['goal_obj']})",
                    f"(inReceptacle {var_names['goal_obj']} {var_names['goal_recep']})",
                ],
                "neg": []
            },
            "pick_clean_then_place_in_recep": {
                "pos": [
                    "(handEmpty)",
                    f"(is{obj} {var_names['goal_obj']})",
                    f"(is{parent} {var_names['goal_recep']})",
                    f"(isClean {var_names['goal_obj']})",
                    f"(inReceptacle {var_names['goal_obj']} {var_names['goal_recep']})",
                ],
                "neg": []
            },
            "look_at_obj_in_light": {
                "pos": [
                    f"(holds {var_names['goal_obj']})",
                    "(handEmpty)",
                    f"(isLightOn {var_names['toggle_obj']})",
                ],
                "neg": []
            },
            # … add more task_types here …
        }

        spec = GOAL_TEMPLATES.get(task_type)
        if spec is None:
            raise ValueError(f"Unsupported task_type: {task_type!r}")

        # Build the conjunction body
        lines: List[str] = []
        for atom in spec["pos"]:
            lines.append(atom)
        for atom in spec["neg"]:
            lines.append(f"(not {atom})")

        body = "\n    ".join(lines)
        pddl_goal = f"(:goal (and\n    {body}\n))"

        objs_block = ""
        init_block = ""
        problem_name = "alfworld_task"
        domain_name = "alfworld_task"

        problem = f"(define (problem {problem_name}) (:domain {domain_name}) (:objects{objs_block}) (:init {init_block}) \n {pddl_goal})"

        base_dir = Path(self.save_path)
        file_path = base_dir / save_name
        with open(file_path, "w") as f:
            f.write(problem)

        return problem, str(file_path)


class ALFWorldTextObsParser:

    def __init__(self, obj_pattern=r'\b\w+\s\d+\b'):
        self.obj_pattern = obj_pattern

    def extract_objects_text(self, obs, location):

        obs = obs.lower().strip()
        obj_pattern = self.obj_pattern
        obj_list = []

        if obs.startswith('you are in the middle of a room') or \
           obs.startswith('you arrive at') or\
           obs.startswith('you open') or \
           'is empty' in obs or \
           obs.strip() == "":
            # raw_objs = re.findall(obj_pattern, obs)
            # objs = ["_".join(i.split(' ')) for i in raw_objs if 'loc ' not in i]
            return obj_list
        elif 'looking quickly around you' in obs:
            obs = ','.join(obs.split(',')[1:])
            raw_objs = re.findall(obj_pattern, obs)
            objs = [("_".join(i.split(' ')),'receptacle', None, "_".join(i.split(' '))) for i in raw_objs if 'loc ' not in i]
            obj_list = obj_list + objs
        elif "is open" in obs:
            raw_objs = re.findall(obj_pattern, obs)
            objs = ["_".join(i.split(' ')) for i in raw_objs if 'loc ' not in i]
            obj_list.append((objs[0], 'receptacle', 'open', objs[0]))
        elif "is closed" in obs:
            raw_objs = re.findall(obj_pattern, obs)
            objs = ["_".join(i.split(' ')) for i in raw_objs if 'loc ' not in i]
            obj_list.append((objs[0], 'receptacle', 'close', objs[0]))
        elif obs.startswith('in it'):
            obs = ','.join(obs.split(',')[1:])
            raw_objs = re.findall(obj_pattern, obs)
            objs = [("_".join(i.split(' ')),'object', 'inside', location) for i in raw_objs if 'loc ' not in i]
            obj_list = obj_list + objs
        elif obs.startswith('on the') or obs.startswith('on it'):
            obs = ','.join(obs.split(',')[1:])
            raw_objs = re.findall(obj_pattern, obs)
            objs = [("_".join(i.split(' ')),'object', 'outside', location) for i in raw_objs if 'loc ' not in i]
            obj_list = obj_list + objs

        elif "is empty" in obs:
            raw_objs = re.findall(obj_pattern, obs)
            objs = [("_".join(i.split(' ')),'object', 'inside', location) for i in raw_objs if 'loc ' not in i]
            obj_list = obj_list + objs

        else:
            raise NotImplementedError(str(obs) + ' is unseen, check text extraction')



        return obj_list
    
    def parse_text_observation(self, obs_str, location=None):
        obj_pattern = self.obj_pattern
        all_objects = []

        sentences = obs_str.split('.')

        obs_str = obs_str.lower()
        
        if  obs_str.startswith('you are in the middle of a room'):
            location = None
        elif obs_str.startswith('you arrive at') or obs_str.startswith('you open') :
            raw_objs = re.findall(obj_pattern, obs_str)
            objs = ["_".join(i.split(' ')) for i in raw_objs if 'loc ' not in i]
            location = objs[0]

        for s in sentences:
            objs = self.extract_objects_text(s, location)
            all_objects = all_objects + objs
        return all_objects

    def extract_objects_from_list(self, obs_list, location):

        obj_list = []
        for i in obs_list:
            obj_list.append((i, 'object', 'outside', location))

        return obj_list

class ALFWorldVLMObsParser:
    def __init__(
        self,
        model: str = "gpt-4o",
        max_tokens: int = 5,
    ):
        self.client = OpenAI()
        self.model = model
        self.max_tokens = max_tokens

        # self.VOI_PROMPT = """
        
        # In the image, do you see {object}?
        # Answer this quesiton with ['yes', 'no'] only.

        # """

        # self.CHOICE_PROMPT = """List all objects in the image that is a {object}.
        # Answer with a list of objects, separated by commas as ['object1', 'object2', ...].
        # """

    def _call_model_vlm(self, img_arr, prompt):
        img = Image.fromarray(img_arr, mode="RGB")
        img.save("output_test.png")

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        encoded_image = base64.b64encode(buffer.read()).decode('utf-8')

        client = OpenAI()

        # if question_type == "bool":
        #     prompt = self.BOOL_PROMPT
        # elif question_type == "choice":
        #     prompt = self.CHOICE_PROMPT

        

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}",
                            },
                        },
                    ],
                }
            ], 
            )
        
        text_response = response.choices[0].message.content.strip().lower()

        # Return 'yes' or 'no' if found in the response
        if "yes" in text_response:
            return "yes"
        else:
            return "no"
        
    def _call_model_vll(self, prompt):




        client = OpenAI()



        response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
                ]
            )
        
        text_response = response.choices[0].message.content.strip().lower()
        return text_response
        
    
    def parse_image(self, img_arr, object: str) -> List[Tuple[str, str]]:
        response = self._call_model_vlm(img_arr, object, question_type="bool")
        detected_objects = json.loads(response.choices[0].message.content)

        print(detected_objects)

        raise ValueError

    def parse_vlm_observation(self, observation, img_arr, voi_events):

        object_list = []
        
        for e in voi_events:
            description = e.description

            prompt = (
            "You are a vision-language model.\n"
            "Only reply with one word: either 'yes' or 'no'.\n"
            "Question: In the image, do you see an {object}?".format(object=description)
                )

            # prompt = self.VOI_PROMPT.format(object=description)

            response = self._call_model_vlm(img_arr, observation)
            print(prompt)
            print(response)
            print("===")

            if response == "yes" or response == "no":
                prompt = (
                    "{observation}\n"
                    "Based on your observation, do you see {object}?\n"
                    "Please give your answer in this format: ['object 1', 'object 2', ...]"
                    "If you do not see any objects, please return 'none'"
                ).format(observation=observation, object=description)

                response = self._call_model_vll(prompt)

                if 'none' not in response:
                    raw_items = response.strip("[]").split(',')
                    objs = [item.strip().strip("'") for item in raw_items]
                    # objs = ast.literal_eval(response)
                    object_list = object_list + objs
        
        object_list = list(set(object_list))

        return object_list







import re
import numpy as np
from collections import defaultdict, Counter

import re
import numpy as np
from collections import defaultdict, Counter

class ALFWorldTextVisionBridge:
    """
    Bridge between THOR pixels/masks and ALFWorld text names like 'fridge_1', 'apple_1',
    with names aligned to the text environment's indices (multi-word classes supported).

    Now prefers the Oracle's own naming (perfect match to text actions); falls back to
    metadata-based alignment if Oracle isn't available.
    """

    # ----- lifecycle ---------------------------------------------------------
    def __init__(self, env_wrapper):
        self.envw = env_wrapper
        self._thor = self._get_thor_env()
        self._feedback = ""
        self._admissible = []
        self._meta = None

        # mappings
        self._id2text = {}       # THOR objectId -> 'class k' (spaced)
        self._text2id = {}       # 'class k' -> THOR objectId
        self._global_order = defaultdict(list)  # cls_key -> [objectIds] (sorted, stable)
        self._class_display = {}  # cls_key -> preferred display string with spaces (from text)

        self.refresh_state()

    def refresh_state(self):
        env0 = getattr(self.envw, "envs", [None])[0]
        if env0 is None:
            raise RuntimeError("Expected a batched env (batch_size=1).")

        # Text side: feedback + admissible commands
        if hasattr(env0, "get_results"):
            res = env0.get_results()
            self._feedback = res[0]
            self._admissible = self._extract_admissible(res[2])
        else:
            self._feedback = getattr(env0, "_feedback", "")
            res = getattr(env0, "_res", [[], [], []])
            self._admissible = self._extract_admissible(res[2] if len(res) > 2 else [])

        # THOR side: metadata and alignment
        self._thor = self._get_thor_env()
        if self._thor is not None and self._thor.last_event is not None:
            self._meta = self._thor.last_event.metadata

            # 1) Prefer EXACT mapping from the Oracle (the text controller)
            if not self._rebuild_from_oracle():
                # 2) Fallback: metadata-based global order + text-driven pinning
                self._build_global_order()
                self._rebuild_text_id_index_global()  # global, deterministic numbering
                self._align_names_to_text()           # pin indices mentioned by text/admissibles
        else:
            self._meta = None
            self._id2text, self._text2id = {}, {}
            self._global_order.clear()
            self._class_display.clear()


    def _is_within_visibility(self, object_id, eps=1e-6):
        """
        Return True iff the object is within the configured visibility distance of the agent.
        Falls back to the simulator's 'visible' flag if we can't resolve a distance threshold.
        """
        if self._meta is None:
            return False

        # look up object + agent pose
        obj = next((o for o in self._meta.get("objects", []) if o.get("objectId") == object_id), None)
        if obj is None:
            return False

        agent_pos = self._meta.get("agent", {}).get("position")
        obj_pos   = obj.get("position")
        if agent_pos is None or obj_pos is None:
            # as a last resort, trust THOR's visibility computation
            return bool(obj.get("visible", False))

        # euclidean distance in (x, z, y) space (THOR uses x, y, z; y is height)
        ax, ay, az = float(agent_pos["x"]), float(agent_pos["y"]), float(agent_pos["z"])
        ox, oy, oz = float(obj_pos["x"]),   float(obj_pos["y"]),   float(obj_pos["z"])
        dist = ((ax - ox)**2 + (ay - oy)**2 + (az - oz)**2) ** 0.5

        # figure out the intended visibility radius
        vis_dist = getattr(self._thor, "visibility_distance", None)
        if vis_dist is None:
            try:
                import alfworld.gen.constants as _c
                vis_dist = getattr(_c, "VISIBILITY_DISTANCE", None)
            except Exception:
                vis_dist = None

        if vis_dist is None:
            # fallback: rely on THOR's own 'visible' boolean
            return bool(obj.get("visible", False))

        return dist <= (float(vis_dist) + eps)

    # ----- public queries ----------------------------------------------------
    def list_text_objects(self):
        objs = self._parse_visible_text_objects(self._feedback)
        return [o.replace(" ", "_") for o in objs]

    def mask_to_text_name(self, mask):
        if self._thor is None or self._thor.last_event is None:
            raise RuntimeError("THOR not active (no frames yet).")
        obj_id = self._mask_to_object_id(mask)
        if obj_id is None:
            return None

        # NEW: distance/visibility guard
        if not self._is_within_visibility(obj_id):
            return None

        # Prefer Oracle/aligned mapping
        if obj_id in self._id2text:
            return self._id2text[obj_id].replace(" ", "_")
        # Fallback to global order
        cls_key = self._cls_key(self._object_type_from_id(obj_id))
        ids = self._global_order.get(cls_key, [])
        k = ids.index(obj_id) + 1 if obj_id in ids else 1
        display = self._class_display.get(cls_key, cls_key)
        return f"{display.replace(' ', '_')}_{k}"
        

    def bbox_to_text_name(self, bbox, conf_threshold=0.0, image_size=None):
        """
        Map a bounding box (ymin, xmin, ymax, xmax) in normalized coords [0,1]
        to the closest THOR object and return its text_name.
        """
        if self._thor is None or self._thor.last_event is None:
            raise RuntimeError("THOR not active (no frames yet).")

        ev = self._thor.last_event
        if image_size is None:
            H, W = ev.frame.shape[:2]
        else:
            H, W = image_size

        ymin, xmin, ymax, xmax = bbox
        x1, y1 = int(xmin * W), int(ymin * H)
        x2, y2 = int(xmax * W), int(ymax * H)

        mask = np.zeros((H, W), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 1

        obj_id = self._mask_to_object_id(mask)
        if obj_id is None:
            return None

        # NEW: distance/visibility guard
        if not self._is_within_visibility(obj_id):
            return None

        if obj_id in self._id2text:
            return self._id2text[obj_id].replace(" ", "_")

        cls_key = self._cls_key(self._object_type_from_id(obj_id))
        ids = self._global_order.get(cls_key, [])
        k = ids.index(obj_id) + 1 if obj_id in ids else 1
        display = self._class_display.get(cls_key, cls_key)
        return f"{display.replace(' ', '_')}_{k}"

    def object_id_to_text(self, object_id):
        name = self._id2text.get(object_id)
        return name.replace(" ", "_") if name else None

    def text_to_object_id(self, text_name):
        key = text_name.replace("_", " ")
        return self._text2id.get(key, None)

    def list_visible_objects_with_masks(self, only_interactable=True, return_bbox=False):
        """
        Return a list like:
          [{'text_name': 'fridge_1', 'object_id': '<THOR-ID>', 'mask': np.ndarray(bool,H,W), 'bbox': (x1,y1,x2,y2)?}, ...]
        """
        if self._thor is None or self._thor.last_event is None:
            return []

        masks = self._get_instance_masks()
        if not masks:
            return []

        # ensure (aligned) text mapping exists
        if not self._id2text and self._meta is not None:
            if not self._rebuild_from_oracle():
                self._build_global_order()
                self._rebuild_text_id_index_global()
                self._align_names_to_text()

        # filter ids by visibility (and optionally interactability)
        visible_ids = set()
        for o in self._meta.get("objects", []):
            if not o.get("visible", False):
                continue
            if (not only_interactable) or (o.get("pickupable") or o.get("receptacle") or o.get("openable") or o.get("toggleable") or o.get("sliceable")):
                visible_ids.add(o["objectId"])

        out = []
        for oid, m in masks.items():
            if oid not in visible_ids or not np.any(m):
                continue
            # use Oracle/aligned name if available; else global fallback with display
            tname = self.object_id_to_text(oid)
            if tname is None:
                cls_key = self._cls_key(self._object_type_from_id(oid))
                ids = self._global_order.get(cls_key, [])
                k = ids.index(oid) + 1 if oid in ids else 1
                display = self._class_display.get(cls_key, cls_key)
                tname = f"{display.replace(' ', '_')}_{k}"

            item = {
                "object_id": oid,
                "text_name": tname,
                "mask": m.astype(bool)
            }
            if return_bbox:
                ys, xs = np.where(item["mask"])
                if ys.size and xs.size:
                    x1, y1 = int(xs.min()), int(ys.min())
                    x2, y2 = int(xs.max()), int(ys.max())
                    item["bbox"] = (x1, y1, x2, y2)
            out.append(item)

        out.sort(key=lambda d: d["text_name"])
        return out

    # ----- internals ---------------------------------------------------------
    def _get_thor_env(self):
        env0s = getattr(self.envw, "envs", [])
        if env0s:
            env0 = env0s[0]
            if hasattr(env0, "env") and getattr(env0, "env") is not None:
                return env0.env
        if hasattr(self.envw, "thor_env"):
            return getattr(self.envw, "thor_env")
        if hasattr(self.envw, "thor"):
            return getattr(self.envw, "thor")
        return None

    # ---------- Oracle integration (exact naming) ----------
    def _get_oracle_agent(self):
        """
        Best-effort to find the Oracle controller that owns the text 'class k' names.
        We check common attributes on the per-env wrapper.
        """
        env0s = getattr(self.envw, "envs", [])
        if not env0s:
            return None
        env0 = env0s[0]
        for attr in ("oracle", "oracle_agent", "agent", "controller"):
            agent = getattr(env0, attr, None)
            # must have at least one of these maps
            if agent is not None and (hasattr(agent, "objects") or hasattr(agent, "receptacles")):
                return agent
        return None

    def _rebuild_from_oracle(self):
        """
        Mirror the Oracle's own text↔THOR mapping, if available.
        Returns True if we found anything; otherwise False (caller will fallback).
        """
        agent = self._get_oracle_agent()
        if agent is None:
            return False

        id2text, text2id = {}, {}

        def harvest(mapping):
            # mapping can be dict (values are records) or list of records
            if mapping is None:
                return
            if isinstance(mapping, dict):
                it = mapping.values()
            elif isinstance(mapping, (list, tuple)):
                it = mapping
            else:
                return
            for rec in it:
                if not isinstance(rec, dict):
                    continue
                # common key variants
                oid = rec.get("objectId") or rec.get("object_id") or rec.get("id")
                name = rec.get("num_id") or rec.get("name") or rec.get("text_id")
                if not name:
                    # best-effort compose from type + count
                    typ = rec.get("objectType") or rec.get("type") or rec.get("class") or rec.get("cls")
                    num = rec.get("num") or rec.get("idx") or rec.get("index")
                    if typ and num is not None:
                        name = f"{str(typ).lower()} {int(num)}"
                if oid and name:
                    # normalize spacing/case
                    name = str(name).strip().lower()
                    id2text[oid] = name
                    text2id[name] = oid

                    # remember display string per class for fallbacks
                    parts = name.rsplit(" ", 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        cls_text = parts[0]
                        self._class_display.setdefault(self._cls_key(cls_text), cls_text)

        # try both receptacles and objects
        harvest(getattr(agent, "receptacles", None))
        harvest(getattr(agent, "objects", None))

        if id2text:
            self._id2text = id2text
            self._text2id = text2id
            return True
        return False

    def _extract_admissible(self, acs):
        # acs may be: list[str], or list[list[str]] (batched). Normalize to flat list[str].
        if isinstance(acs, (list, tuple)):
            if len(acs) and isinstance(acs[0], (list, tuple)):
                return [str(x) for x in acs[0]]
            return [str(x) for x in acs]
        return []

    # --- TEXT parsing (multi-word classes + a/an/the) ---
    def _parse_visible_text_objects(self, feedback):
        objs = []
        if not feedback or "you see" not in feedback.lower():
            return objs
        try:
            _, after = re.split(r"you see", feedback, flags=re.IGNORECASE, maxsplit=1)
            object_list = after.split(".", 1)[0]
            object_list = object_list.replace(", and ", ", ").replace(" and ", ", ")
            parts = [p.strip() for p in object_list.split(",") if p.strip()]
            for p in parts:
                # allow 'a|an|the' and multi-word classes ending with an index
                m = re.search(r"(?:a|an|the)\s+([a-z ]+?\s+\d+)\b", p, flags=re.IGNORECASE)
                if m:
                    objs.append(m.group(1).lower().strip())
        except Exception:
            pass
        return sorted(set(objs))

    def _visible_interactable_ids(self):
        if self._meta is None:
            return []
        out = []
        for o in self._meta.get("objects", []):
            if not o.get("visible", False):
                continue
            if o.get("pickupable") or o.get("receptacle") or o.get("openable") or o.get("toggleable") or o.get("sliceable"):
                out.append(o["objectId"])
        return out

    def _build_global_order(self):
        """Per-class global order of objectIds (stable: sorted by objectId)."""
        self._global_order.clear()
        if self._meta is None:
            return
        for o in self._meta.get("objects", []):
            cls_key = self._cls_key(o["objectType"])  # collapse spaces/underscores, lowercase
            self._global_order[cls_key].append(o["objectId"])
        for cls_key in self._global_order:
            self._global_order[cls_key].sort()

    def _rebuild_text_id_index_global(self):
        """
        Global mapping from all objects using global order (no text alignment yet).
        Uses a best-effort display string; alignment will overwrite for classes seen in text.
        """
        self._id2text, self._text2id = {}, {}
        for cls_key, ids in self._global_order.items():
            display = self._class_display.get(cls_key, cls_key)  # may be overwritten by alignment
            for i, oid in enumerate(ids, start=1):
                name = f"{display} {i}"
                self._id2text[oid] = name
                self._text2id[name] = oid

    def _align_names_to_text(self):
        """
        Pin indices that appear in the text (observation + admissibles) to the same
        k-th object (in global order) for that class. Supports multi-word classes.
        """
        if self._meta is None:
            return

        # harvest tokens like '<class> <k>' from observation
        text_tokens = set(self._parse_visible_text_objects(self._feedback))

        # plus from admissible commands (allow multi-word classes)
        for ac in self._admissible:
            for m in re.finditer(r"\b([a-z ]+?)\s+(\d+)\b", ac.lower()):
                text_tokens.add(f"{m.group(1).strip()} {m.group(2)}")

        want_by_cls = defaultdict(set)   # cls_key -> {k,...}
        for t in text_tokens:
            m = re.match(r"^([a-z ]+?)\s+(\d+)$", t)
            if not m:
                continue
            cls_text = m.group(1).strip()
            k = int(m.group(2))
            cls_key = self._cls_key(cls_text)
            want_by_cls[cls_key].add(k)
            # prefer the text's display string (with spaces) for this class
            self._class_display.setdefault(cls_key, cls_text)

        # start from global mapping and overwrite requested indices
        id2text_aligned = dict(self._id2text)
        text2id_aligned = dict(self._text2id)

        # remove per-class names first (we will reassign deterministically)
        for cls_key, ids in self._global_order.items():
            for oid in ids:
                nm = id2text_aligned.get(oid)
                if nm and nm.startswith((self._class_display.get(cls_key, cls_key) + " ")):
                    text2id_aligned.pop(nm, None)

        # assign requested indices (using the text display for that class)
        for cls_key, want_idxs in want_by_cls.items():
            ids = self._global_order.get(cls_key, [])
            if not ids:
                continue
            display = self._class_display.get(cls_key, cls_key)
            for k in sorted(want_idxs):
                pos = k - 1
                if 0 <= pos < len(ids):
                    oid = ids[pos]
                    name = f"{display} {k}"
                    id2text_aligned[oid] = name
                    text2id_aligned[name] = oid

        # fill remaining slots with global positions (avoid clobbering pinned names)
        for cls_key, ids in self._global_order.items():
            display = self._class_display.get(cls_key, cls_key)
            for i, oid in enumerate(ids, start=1):
                name = f"{display} {i}"
                if name not in text2id_aligned:
                    text2id_aligned[name] = oid
                id2text_aligned.setdefault(oid, name)

        self._id2text = id2text_aligned
        self._text2id = text2id_aligned

    @staticmethod
    def _object_type_from_id(object_id):
        return object_id.split("|", 1)[0]

    @staticmethod
    def _cls_key(s):
        """Canonical class key: lowercase, strip spaces/underscores (e.g., 'PaperTowelRoll' -> 'papertowelroll', 'paper towel roll' -> 'papertowelroll')."""
        return re.sub(r"[\s_]+", "", s.lower())

    def _mask_to_object_id(self, mask):
        if mask is None or self._thor is None or self._thor.last_event is None:
            return None
        mask = (mask > 0).astype(np.uint8)
        ev = self._thor.last_event
        seg = np.array(ev.instance_segmentation_frame)
        color_map = ev.color_to_object_id
        nzr, nzc = np.nonzero(mask)
        if len(nzr) == 0:
            return None
        counts = Counter()
        for r, c in zip(nzr, nzc):
            counts[tuple(seg[r, c])] += 1
        mask_bool = mask.astype(bool)
        best_color, best_iou = None, -1.0
        for col, inter in counts.most_common():
            inst_bool = np.all(seg == col, axis=2)
            union = np.logical_or(inst_bool, mask_bool).sum()
            iou = (inter / float(union)) if union else 0.0
            if iou > best_iou:
                best_color, best_iou = col, iou
        if best_color is None:
            return None
        return color_map.get(best_color)

    def _get_instance_masks(self):
        """
        Prefer THOR's per-instance masks; else build them from the instance seg image.
        Returns: dict {objectId: bool(H,W)} for all objects present in the frame.
        """
        ev = self._thor.last_event
        inst = getattr(ev, "instance_masks", None)
        if isinstance(inst, dict) and len(inst):
            return {oid: m.astype(bool) for oid, m in inst.items()}

        seg = np.asarray(ev.instance_segmentation_frame)  # HxWx3
        color2id = ev.color_to_object_id
        colors = np.unique(seg.reshape(-1, 3), axis=0)
        out = {}
        for color in colors:
            color_t = tuple(int(v) for v in color.tolist())
            oid = color2id.get(color_t)
            if not oid:
                continue
            mask = ((seg[:, :, 0] == color[0]) &
                    (seg[:, :, 1] == color[1]) &
                    (seg[:, :, 2] == color[2]))
            if mask.any():
                out[oid] = mask
        return out


    
    




if __name__ == "__main__":
    estimator = LLMTruthProbEstimator()

    headlines = [
        "There exists an object p1 such that: p1 is hot, and p1 is pickupable object, and p1 is potato",
        "There exists an object p1 such that: p1 is pickupable object, and p1 is potato",
        "There exists an object r1 such that: r1 receptacle object, and r1 is table",
        "There exists two objects ?V47 and p1 such that: ?V47 can heat p1; ?V47 is heating object; p1 is pickupable object, and p1 is potato",
    ]

    for h in headlines:
        p_true = estimator.get_probability(h)
        print(f"{h!r} → P(True) = {p_true:.2f}")