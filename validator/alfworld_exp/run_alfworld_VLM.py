import yaml
import numpy as np
from PIL import Image, ImageDraw

import os
import json
import io
import re
import traceback

from tqdm import tqdm
from pprint import pprint
import pdb

from alfworld.agents.environment import get_environment
import alfworld.agents.modules.generic as generic

from parser import ALFWorldProblemParser
from vlm import detect_object_types  # we draw locally to stay RGB

import cv2
import shutil
import sys


# ------------------------------------------------------------
# File discovery
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Output helpers
# ------------------------------------------------------------

def _sanitize_for_path(s) -> str:
    # ensure string, then keep only [A-Za-z0-9_-], collapse others to "_"
    return re.sub(r'[^A-Za-z0-9_\-]+', '_', str(s))


def build_episode_outdir(json_path: str, data_path: str, *, reset: bool = True) -> str:
    """
    Build ./output_vlm/<rel_to_data_path_of_json_dir>. If reset=True and the
    directory exists, delete it first so we save from scratch.
    """
    json_dir = os.path.dirname(json_path)
    try:
        rel = os.path.relpath(json_dir, start=os.path.abspath(data_path))
    except Exception:
        rel = _sanitize_for_path(json_dir)

    # Build absolute, normalized paths and guard against traversal
    base_out = os.path.abspath("./output_vlm")
    out_dir = os.path.abspath(os.path.join(base_out, rel))
    if not (out_dir == base_out or out_dir.startswith(base_out + os.sep)):
        raise ValueError(f"Refusing to write outside ./output_vlm (got: {out_dir})")

    # Reset if requested
    if reset and os.path.isdir(out_dir):
        shutil.rmtree(out_dir)

    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def safe_action_name(action) -> str:
    # accept any object, stringify, sanitize, and trim
    return _sanitize_for_path(action)[:120]

def rgb_to_png_bytes(frame_rgb: np.ndarray) -> bytes:
    """Encode an RGB numpy array to PNG bytes (no color conversion)."""
    im = Image.fromarray(frame_rgb.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()

def save_bbox_overlay_rgb(frame_rgb: np.ndarray, detections, out_path: str):
    """Draw bboxes on an RGB array and save as true RGB PNG."""
    # ensure frame is RGB, not BGR
    if frame_rgb.shape[-1] == 3:
        frame_rgb = frame_rgb[..., ::-1]  # convert BGR→RGB if needed

    image = Image.fromarray(frame_rgb.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    w, h = image.size
    for det in detections:
        ymin, xmin, ymax, xmax = det["bbox"]  # normalized [0,1]
        x0 = int(xmin * w); y0 = int(ymin * h)
        x1 = int(xmax * w); y1 = int(ymax * h)
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
        label = f"{det['object']} {det['confidence']:.2f}"
        draw.text((x0 + 4, y0 + 4), label, fill=(255, 255, 0))
    
    image.save("frame.png")
    # pdb.set_trace()

    image.save(out_path, format="PNG")  # guaranteed RGB


# ------------------------------------------------------------
# Failure logging
# ------------------------------------------------------------

FAILED_LOG = "failed_tasks_vlm.txt"

def log_failed(task_path: str, reason: str = ""):
    """Append a failed task (absolute path + optional reason) to failed_tasks_vlm.txt."""
    try:
        abs_path = os.path.abspath(task_path)
        line = abs_path if not reason else f"{abs_path}\t{reason}"
        with open(FAILED_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        # last-resort: print if file logging somehow fails
        print(f"[WARN] Failed to log task: {task_path} ({reason})")


# ------------------------------------------------------------
# Persistent progress (for resume)
# ------------------------------------------------------------

PROGRESS_FILE = "progress_vlm.json"

def load_progress():
    """Load progress file if present; else return empty progress dict."""
    try:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # ensure sets/lists exist
        return {
            "processed": set(data.get("processed", [])),
            "success": set(data.get("success", [])),
            "failed": set(data.get("failed", [])),
        }
    except Exception:
        return {"processed": set(), "success": set(), "failed": set()}

def save_progress(progress):
    """Persist current progress to disk."""
    try:
        data = {
            "processed": sorted(list(progress.get("processed", set()))),
            "success": sorted(list(progress.get("success", set()))),
            "failed": sorted(list(progress.get("failed", set()))),
        }
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to save progress: {e}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

from regression_agent.agent_VLM import RegressionAgentVLM

# help functions
with open('./configs/base_config.yaml') as reader:
    config = yaml.safe_load(reader)

env_type = config['env']['type'] # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'
env_type = 'AlfredThorEnv'
# env_type = 'AlfredTWEnv'
# env_type = 'AlfredHybrid'

# data_path = './dev_data_single/'
# data_path = './dev_data/'

data_path = '/home/user/research/data_folder/alfworld/json_2.1.1/valid_unseen'

## failed
# data_path = "/home/user/research/data_folder/alfworld/json_2.1.1/valid_unseen/pick_clean_then_place_in_recep-Cloth-None-CounterTop-424/trial_T20190908_114340_674467"

config['dataset']['data_path'] = data_path

# setup environment
env = get_environment(env_type)(config, train_eval='train')
env = env.init_env(batch_size=1)

gamefiles, jsonfiles = extract_valid_files(data_path)

# Optional: run only failed episodes listed in a progress file
# Minimal argv parsing (no argparse dependency)
use_failed = False
progress_file_path = PROGRESS_FILE
argv = sys.argv[1:]
if "--use-failed" in argv:
    use_failed = True
    # support optional "--progress-file <path>"
    if "--progress-file" in argv:
        try:
            idx = argv.index("--progress-file")
            progress_file_path = argv[idx + 1]
        except Exception:
            print("[WARN] --progress-file provided without a path; falling back to default progress_vlm.json")

if use_failed:
    # Load failed list from the progress file instead of failed_tasks_vlm.txt
    try:
        with open(progress_file_path, "r", encoding="utf-8") as pf:
            pdata = json.load(pf)
        failed_list = [os.path.abspath(p) for p in pdata.get("failed", [])]
        # Filter to existing files only
        failed_list = [p for p in failed_list if os.path.isfile(p)]
        if failed_list:
            print(f"[INFO] Using {len(failed_list)} failed episodes from {progress_file_path}")
            jsonfiles = failed_list
        else:
            print(f"[WARN] No failed episodes found in {progress_file_path}; proceeding with discovered set ({len(jsonfiles)}).")
    except Exception as e:
        print(f"[WARN] Could not read failed episodes from {progress_file_path}: {e}. Proceeding with discovered set ({len(jsonfiles)}).")

print("Game files: ", gamefiles)
print("Json files: ", jsonfiles)

success_tasks = []
failed_tasks = []

# Load prior progress for resume
progress = load_progress()
processed_set = set(os.path.abspath(p) for p in progress.get("processed", set()))
success_set   = set(os.path.abspath(p) for p in progress.get("success", set()))
failed_set    = set(os.path.abspath(p) for p in progress.get("failed", set()))

# Seed summary lists from prior progress (optional)
success_tasks.extend(sorted(success_set))
failed_tasks.extend(sorted(failed_set))

prior_processed = len(processed_set)

# Determine how many remain to run for this invocation
remaining_jsons = [j for j in jsonfiles if os.path.abspath(j) not in processed_set]
total = prior_processed + len(remaining_jsons)

failed_tasks_file = "failed_tasks_vlm.txt"

processed_this_run = 0
for episode_json in jsonfiles:
    abs_json = os.path.abspath(episode_json)
    # Skip policy differs by mode:
    # - Normal: skip anything in processed_set (resume behavior)
    # - --use-failed: rerun failed episodes, skip ONLY those already marked success
    if not 'use_failed' in globals() or not use_failed:
        if abs_json in processed_set:
            print(f"[Resume] Skipping already processed: {episode_json}")
            continue
    else:
        if abs_json in success_set:
            print(f"[Skip success] Already succeeded: {episode_json}")
            continue
    processed_this_run += 1
    try:
        reg_agent = RegressionAgentVLM(
            domain_path='/home/user/research/NL-PDDL/NL_ALFWorld/alfworldtext_domain.json',
            env=env,
            llm_model="gpt-4o"
        )
        env.json_file_list = [episode_json]  # <-- drive env by this episode explicitly

        # Do not reset output when resuming
        out_dir = build_episode_outdir(episode_json, data_path, reset=False)
        print(f"[Output] Saving episode artifacts to: {out_dir}")

        obs, info = env.reset()

        print(obs[0])

        NL_goal = reg_agent.parse_goal_llm(info, obs_text=obs[0])
        # NL_goal = reg_agent.parse_goal(info)
        reg_agent.set_solver()
        reg_agent.generate_policy(max_plan_length=5)

        reg_agent.update_initial_observation_vlm(obs[0], gen_type="llm")

        done = False
        # Safeguard: cap iterations to avoid infinite loops when no progress is possible
        iter_count = 0
        MAX_ITERS = 50

        from parser import ALFWorldTextVisionBridge
        bridge = ALFWorldTextVisionBridge(env)

        while reg_agent.step_count < 50 and not done:
            iter_count += 1
            if iter_count > MAX_ITERS:
                print(f"[LoopGuard] Reached MAX_ITERS={MAX_ITERS} without finishing. Stopping episode.")
                break
            reg_agent.generate_plan_query()
            action_type, plan_list = reg_agent.gen_kb_action()

            # for p in plan_list:
            #     print(p)

            

            if action_type == "explore":
                env_action = reg_agent.gen_env_explore_action(criteria="random")
                success, feedback, failed_action, env_action, done = reg_agent.step(env_action)
                bridge.refresh_state()
                frame = env.get_frames()[0]
                relevant_objects = reg_agent.get_objects_list()
                _, buffer = cv2.imencode(".png", frame)
                img_bytes = buffer.tobytes()
                found_objects = detect_object_types(img_bytes, relevant_objects)
                object_names = []
                for obj in found_objects:
                    bbox = obj['bbox']
                    obj_name = bridge.bbox_to_text_name(bbox)
                    if obj_name is not None:
                        object_names.append(obj_name)
                if success:
                    reg_agent.update_object_list(object_names, gen_type="llm")
                
                print(found_objects)
                print(object_names)
                print('----------------')

            elif action_type == "gen_relationships":
                # for p in plan_list: print(p)
                reg_agent.generate_relationships(plan_list, gen_type="llm")

            elif action_type == "plan_action":
                
                plan = reg_agent.choose_excutable_plan(plan_list)
                print(plan)
                # Defensive: if no executable plan, stop the episode to avoid spinning
                if plan is None or not getattr(plan, 'actions', None):
                    print("[Plan] No executable plan returned. Stopping episode.")
                    break
                query_str = plan.get_query_str()
                query_vars = plan.extract_unique_objects_query(query_str)
                query_answer = plan.get_grounding(query_str)[0]

                # print(query_str)
                # print(query_vars)
                # print(query_answer)
                # print('----------------')

                var_mapping = {query_vars[i]: query_answer[i] for i in range(len(query_vars))}
                plan_success = True
                for action in plan.actions:
                    # Plan each env action and attempt it
                    env_action = reg_agent.plan_action2env_action(action, var_mapping)
                    if env_action is None:
                        # Could not form a valid action; mark failure but continue to try remaining actions
                        plan_success = False
                        continue

                    success, feedback, failed_action, env_action, done = reg_agent.step(env_action)

                    if success:
                        # record and save frame for successful action
                        reg_agent.success_env_actions.add(env_action)
                        frame = env.get_frames()[0]
                        step_tag = f"{reg_agent.step_count:03d}"
                        out_name = f"step{step_tag}_{safe_action_name(env_action)}.png"
                        out_path = os.path.join(out_dir, out_name)
                        cv2.imwrite(out_path, frame)
                    else:
                        # Try an alternative object/receptacle for this action
                        reg_agent.failed_env_actions.add(env_action)


                # If we executed all actions successfully, consider the plan (and task) done
                if plan_success and not done:
                    done = True
                if done:
                    break
                # If plan failed overall, stop the episode to avoid infinite cycles on the same failing plan
                if not plan_success:
                    done = False
            else:
                raise NotImplementedError(action_type + " not implemented")

        # episode-level accounting (don’t depend on info['extra.gamefile'])
        abs_json = os.path.abspath(episode_json)
        if done:
            success_tasks.append(os.path.dirname(abs_json))
            success_set.add(abs_json)
        else:
            failed_tasks.append(os.path.dirname(abs_json))
            failed_set.add(abs_json)
            log_failed(abs_json, reason="episode finished without success")

    except Exception as e:
        # Suppress full traceback to avoid noisy console output
        print(f"[ERROR] Episode error: {e.__class__.__name__}: {e}")
        abs_json = os.path.abspath(episode_json)
        log_failed(abs_json, reason=f"exception: {e.__class__.__name__}: {e}")
        failed_tasks.append(os.path.dirname(abs_json))
        failed_set.add(abs_json)
    finally:
        # ---- This ALWAYS runs once per (attempted) episode ----
        # Update persistent progress and print summary with prior processed included.
        abs_json = os.path.abspath(episode_json)
        processed_set.add(abs_json)
        progress = {"processed": processed_set, "success": success_set, "failed": failed_set}
        save_progress(progress)

        num_success = len(success_set)
        num_failed = len(failed_set)
        so_far = len(processed_set)
        success_rate = (num_success / so_far * 100) if so_far else 0.0
        print(f"[Progress] {so_far}/{total} tasks done | "
              f"Success: {num_success} | Failed: {num_failed} | "
              f"Success rate: {success_rate:.2f}%")
