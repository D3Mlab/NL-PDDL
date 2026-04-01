import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

import yaml
from alfworld.agents.environment import get_environment

# Local import
from parser import ALFWorldProblemParser


def find_traj_files(data_path: Path) -> List[Path]:
    """Recursively find all traj_data.json files under data_path."""
    return list(data_path.rglob("traj_data.json"))


def load_traj(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def extract_task_descs_from_obs(obs_text: str) -> List[str]:
    """
    Extract task description(s) from ALFWorld observation text.
    Typical pattern: "... Your task is to: <desc>".
    Returns a list with a single description if found; else empty list.
    """
    try:
        cleaned = (obs_text or "").replace("-= Welcome to TextWorld, ALFRED! =-", "").strip()
        if "Your task is to:" in cleaned:
            parts = cleaned.split("Your task is to:", 1)
            desc = parts[1].strip()
            # normalize whitespace
            desc = " ".join(desc.split())
            if desc:
                return [desc]
    except Exception:
        pass
    return []


def build_llm_examples_with_env(
    data_path: Path,
    examples_path: str,
    model: str,
    env_type: str,
    config_path: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Build LLM-generated NL goal examples grouped by task_type using parse_goal_llm,
    while driving the ALFWorld environment to obtain obs/info per trajectory.

    Output format:
    {
      task_type: {
        task_desc: NL_goal_list
      }
    }
    """
    # Load base config and set dataset path
    with open(config_path, "r") as reader:
        config = yaml.safe_load(reader)
    config['env']['type'] = env_type
    config['dataset']['data_path'] = str(data_path)

    # Initialize environment
    env = get_environment(env_type)(config, train_eval='train')
    env = env.init_env(batch_size=1)

    # warm-up reset to print an initial obs
    obs, info = env.reset()
    try:
        print("[Env Reset] Sample obs:")
        print(obs[0])
    except Exception:
        pass

    parser = ALFWorldProblemParser()
    grouped: Dict[str, Dict[str, Any]] = {}

    files = find_traj_files(data_path)
    for fp in files:
        # Drive env to this specific episode
        try:
            env.json_file_list = [str(fp)]
            obs, info = env.reset()
        except Exception as e:
            print(f"[WARN] Env reset failed for {fp}: {e}", file=sys.stderr)
            continue

        # Extract task description from obs
        obs_text = ""
        try:
            obs_text = obs[0] if isinstance(obs, (list, tuple)) and obs else str(obs)
        except Exception:
            obs_text = str(obs)
        descs = extract_task_descs_from_obs(obs_text)

        # Fallback: no desc in obs -> use JSON annotations
        try:
            traj = load_traj(fp)
        except Exception as e:
            print(f"[WARN] Failed to load {fp}: {e}", file=sys.stderr)
            continue

        task_type = traj.get("task_type")
        if not task_type:
            print(f"[WARN] Missing task_type in {fp}", file=sys.stderr)
            continue

        if not descs:
            ta = traj.get("turk_annotations", {})
            anns = ta.get("anns", [])
            for ann in anns:
                td = ann.get("task_desc")
                if isinstance(td, str) and td.strip():
                    descs.append(td.strip())
            if not descs:
                root_desc = traj.get("task_desc") or traj.get("task_description")
                if isinstance(root_desc, str) and root_desc.strip():
                    descs = [root_desc.strip()]

        if not descs:
            print(f"[INFO] No task_desc found for {fp}; skipping.")
            continue

        bucket = grouped.setdefault(task_type, {})

        for d in descs:
            if d in bucket:
                # Already generated for this description; skip duplicate
                continue
            try:
                nl_goal = parser.parse_goal_llm(
                    task_type,
                    goal_key=d,
                    examples_path=examples_path,
                    model=model,
                )
            except Exception as e:
                print(f"[WARN] parse_goal_llm failed for {fp} (task_type={task_type}): {e}", file=sys.stderr)
                continue
            bucket[d] = nl_goal

            # Print out the generated entry with context
            print("\n[Generated]")
            print(f"Episode: {fp}")
            print(f"Task type: {task_type}")
            print(f"Task description: {d}")
            try:
                print(json.dumps(nl_goal, indent=2))
            except Exception:
                print(str(nl_goal))

    return grouped


def main():
    ap = argparse.ArgumentParser(description="Generate LLM-based NL goal examples grouped by task_type using env.")
    ap.add_argument("--data-path", type=str, default="dev_data/",
                    help="Root directory containing ALFWorld traj JSON hierarchy (default: %(default)s)")
    ap.add_argument("--examples", type=str, default="nl_goal_examples.json",
                    help="Path to existing examples file used for few-shot (default: %(default)s)")
    ap.add_argument("--out", type=str, default="nl_goal_llm_test.json",
                    help="Output JSON path (default: %(default)s)")
    ap.add_argument("--model", type=str, default="gpt-4o",
                    help="LLM model for goal generation (default: %(default)s)")
    ap.add_argument("--env-type", type=str, default="AlfredTWEnv",
                    help="Environment type: AlfredTWEnv | AlfredThorEnv | AlfredHybrid (default: %(default)s)")
    ap.add_argument("--config", type=str, default="./configs/base_config.yaml",
                    help="Path to base YAML config (default: %(default)s)")
    args = ap.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"[ERROR] data_path does not exist: {data_path}", file=sys.stderr)
        sys.exit(1)

    grouped = build_llm_examples_with_env(
        data_path=data_path,
        examples_path=args.examples,
        model=args.model,
        env_type=args.env_type,
        config_path=args.config,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(grouped, f, indent=2)

    # brief summary to stdout
    counts = {k: len(v) for k, v in grouped.items()}
    print("\nWrote:", out_path)
    print("Task type counts:")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v} examples")


if __name__ == "__main__":
    main()