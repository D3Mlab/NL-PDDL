import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

# Local import from this repository
from parser import ALFWorldProblemParser


def find_traj_files(data_path: Path) -> List[Path]:
    """
    Recursively find all traj_data.json files under data_path.
    """
    return list(data_path.rglob("traj_data.json"))


def load_traj(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def extract_task_descs(traj: Dict[str, Any]) -> List[str]:
    descs: List[str] = []
    ta = traj.get("turk_annotations", {})
    anns = ta.get("anns", [])
    for ann in anns:
        td = ann.get("task_desc")
        if isinstance(td, str) and td.strip():
            descs.append(td.strip())
    return descs


def extract_goal_params(traj: Dict[str, Any]) -> Dict[str, Any]:
    params = traj.get("pddl_params", {}) or {}
    # Ensure keys exist to avoid AttributeError in parser
    params.setdefault("object_target", "")
    params.setdefault("parent_target", "")
    params.setdefault("toggle_target", "")
    return params


def build_examples(data_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Build examples grouped by task_type.

    Returns a dict:
    {
      task_type: {
        task_desc: NL_goal_list
      }
    }
    """
    parser = ALFWorldProblemParser()
    grouped: Dict[str, Dict[str, Any]] = {}

    files = find_traj_files(data_path)
    for fp in files:
        try:
            traj = load_traj(fp)
        except Exception as e:
            print(f"[WARN] Failed to load {fp}: {e}", file=sys.stderr)
            continue

        task_type = traj.get("task_type")
        if not task_type:
            print(f"[WARN] Missing task_type in {fp}", file=sys.stderr)
            continue

        goal_params = extract_goal_params(traj)
        try:
            nl_goal = parser.parse_nl_pddl_problem(goal_params, task_type)
        except Exception as e:
            print(f"[WARN] parse_nl_pddl_problem failed for {fp} (task_type={task_type}): {e}", file=sys.stderr)
            continue

        descs = extract_task_descs(traj)
        if not descs:
            # If no annotated descriptions, use the high-level description if present
            hd = traj.get("task_desc") or traj.get("task_description")
            if isinstance(hd, str) and hd.strip():
                descs = [hd.strip()]

        if not descs:
            # As a last resort, synthesize a description from params
            obj = goal_params.get("object_target", "").lower()
            parent = goal_params.get("parent_target", "").lower()
            descs = [f"{task_type} with object={obj}, parent={parent}"]

        bucket = grouped.setdefault(task_type, {})
        for d in descs:
            # Last writer wins if duplicate keys occur across files
            bucket[d] = nl_goal

    return grouped


def main():
    default_data = "/home/user/research/data_folder/alfworld/json_2.1.1/valid_unseen"

    ap = argparse.ArgumentParser(description="Extract NL-PDDL goal examples grouped by task_type.")
    ap.add_argument("--data-path", type=str, default=default_data,
                    help="Root directory containing ALFWorld traj JSON hierarchy (default: %(default)s)")
    ap.add_argument("--out", type=str, default="nl_goal_llm.json",
                    help="Output JSON path (default: %(default)s)")
    args = ap.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"[ERROR] data_path does not exist: {data_path}", file=sys.stderr)
        sys.exit(1)

    grouped = build_examples(data_path)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(grouped, f, indent=2)

    # brief summary to stdout
    counts = {k: len(v) for k, v in grouped.items()}
    print("Wrote:", out_path)
    print("Task type counts:")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v} examples")


if __name__ == "__main__":
    main()
