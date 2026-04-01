#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def load_files_by_pattern(stats_dir: Path, pattern: str) -> Iterable[Tuple[Path, Any]]:
    for p in sorted(stats_dir.glob(pattern)):
        try:
            with p.open() as f:
                data = json.load(f)
            yield p, data
        except Exception as e:
            print(f"Warning: failed to read {p}: {e}")


def load_text_files_by_pattern(stats_dir: Path, pattern: str) -> Iterable[Tuple[Path, str]]:
    for p in sorted(stats_dir.glob(pattern)):
        try:
            text = p.read_text(errors="ignore")
            yield p, text
        except Exception as e:
            print(f"Warning: failed to read {p}: {e}")


def load_stats_files(stats_dir: Path) -> Iterable[Tuple[Path, Dict[str, Any]]]:
    return load_files_by_pattern(stats_dir, "*_stats.json")  # non-recursive


def load_val_summary_files(stats_dir: Path) -> Iterable[Tuple[Path, Dict[str, Any]]]:
    return load_files_by_pattern(stats_dir, "*_val_summary.json")


def load_main_json_files(stats_dir: Path) -> Iterable[Tuple[Path, Dict[str, Any]]]:
    # Exclude *_stats.json and *_val_summary.json
    for p in sorted(stats_dir.glob("*.json")):
        if p.name.endswith("_stats.json") or p.name.endswith("_val_summary.json"):
            continue
        try:
            with p.open() as f:
                data = json.load(f)
            yield p, data
        except Exception as e:
            print(f"Warning: failed to read {p}: {e}")


def load_initial_state_txt_files(stats_dir: Path) -> Iterable[Tuple[Path, str]]:
    return load_text_files_by_pattern(stats_dir, "*_initial_state.txt")


def _boolish(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        if v.lower() in {"true", "yes", "1"}:
            return True
        if v.lower() in {"false", "no", "0"}:
            return False
    return None


action_line_patterns = [
    re.compile(r"^\s*\(.*\)\s*$", re.IGNORECASE),  # (action ...)
    re.compile(r"^\s*\d+\s*:\s*\(.*\)\s*$", re.IGNORECASE),  # 1: (action ...)
    re.compile(r"^\s*step\s*\d*\s*:\s*\(.*\)\s*$", re.IGNORECASE),  # step: (action ...)
    re.compile(r"^\s*action\b.*", re.IGNORECASE),  # lines starting with 'action'
    # YAML-style bullet list: - action(args) or * action(args)
    re.compile(r"^\s*[-*]\s*[^()\n]+\([^)]*\)\s*$", re.IGNORECASE),
]


def _extract_from_initial_state_text(text: str) -> int:
    """Return length of the first action block after the first 'Actions:' line.

    - Find the first line matching 'Actions:' (case-insensitive).
    - Starting from the next line, count contiguous action lines (as matched by
      action_line_patterns). Stop at the first non-matching line.
    - Return that count (0 if there is no 'Actions:' or no following action lines).
    """
    lines = [ln.rstrip("\n") for ln in text.splitlines()]

    # Find the first 'Actions:' header
    start_idx: Optional[int] = None
    for i, ln in enumerate(lines):
        if re.match(r"^\s*actions\s*:\s*$", ln, flags=re.IGNORECASE):
            start_idx = i
            break

    if start_idx is None:
        return 0

    # Count only the first contiguous block after Actions:
    count = 0
    for ln in lines[start_idx + 1 : ]:
        if not ln.strip():
            # blank line ends the first block if we've started counting
            if count > 0:
                break
            else:
                # ignore leading blank lines
                continue
        if any(pat.match(ln) for pat in action_line_patterns):
            count += 1
        else:
            # stop at first non-action line once we've begun counting
            if count > 0:
                break
            # if we haven't started counting yet, keep scanning for the very first action line
            # but as soon as we encounter a non-empty non-action line before any action, we assume
            # there is no action list immediately under Actions:
            else:
                # No immediate action item under Actions:, treat as zero-length
                break

    return count


def _extract_plan_length(plan_obj: Any) -> Optional[int]:
    if plan_obj is None:
        return None

    actions = None
    if isinstance(plan_obj, dict):
        if isinstance(plan_obj.get("actions"), list):
            actions = plan_obj["actions"]
        elif isinstance(plan_obj.get("plan"), list):
            actions = plan_obj["plan"]
        elif isinstance(plan_obj.get("action_sequence"), list):
            actions = plan_obj["action_sequence"]
        elif isinstance(plan_obj.get("steps"), list):
            actions = plan_obj["steps"]
        if actions is not None:
            return len(actions)

        for k in ("plan_length", "length", "num_actions", "n_actions", "horizon"):
            v = plan_obj.get(k)
            if isinstance(v, int):
                return v
            if isinstance(v, str) and v.isdigit():
                return int(v)

        for k in ("plan", "actions", "action_sequence"):
            v = plan_obj.get(k)
            if isinstance(v, str):
                lines = [ln for ln in v.strip().splitlines() if ln.strip()]
                if lines:
                    return len(lines)

    if isinstance(plan_obj, list):
        return len(plan_obj)

    if isinstance(plan_obj, str):
        lines = [ln for ln in plan_obj.strip().splitlines() if ln.strip()]
        if lines:
            return len(lines)

    return None


def _plan_is_init_satisfiable(plan_obj: Any) -> Optional[bool]:
    if plan_obj is None:
        return None

    if isinstance(plan_obj, dict):
        for k in (
            "initial_state_satisfied",
            "init_satisfied",
            "satisfiable_initial_state",
            "initial_satisfied",
            "init_sat",
            "initial_sat",
            "is_init_satisfied",
            # sometimes a generic flag
            "satisfied",
        ):
            if k in plan_obj:
                b = _boolish(plan_obj[k])
                if b is not None:
                    return b

        checks = plan_obj.get("checks")
        if isinstance(checks, dict):
            init = checks.get("initial_state")
            if isinstance(init, dict):
                sat = init.get("satisfied")
                b = _boolish(sat)
                if b is not None:
                    return b

        result = plan_obj.get("result")
        if isinstance(result, dict):
            for k in ("initial_state_satisfied", "init_satisfied"):
                if k in result:
                    b = _boolish(result[k])
                    if b is not None:
                        return b

    return None


def _iter_plans_from_stats(stats: Dict[str, Any]) -> Iterable[Any]:
    for key in ("plans", "plan_candidates", "validated_plans", "all_plans", "results"):
        v = stats.get(key)
        if isinstance(v, list):
            for item in v:
                yield item
            return
    for key in ("best_plan", "selected_plan", "plan"):
        v = stats.get(key)
        if v is not None:
            yield v
            return
    return


def _iter_plans_from_val_summary(obj: Any) -> Iterable[Any]:
    # Many val_summary JSONs are arrays or dicts with a list under some key
    if isinstance(obj, list):
        for item in obj:
            yield item
    elif isinstance(obj, dict):
        # common containers
        for key in ("plans", "validated", "candidates", "results", "entries", "items"):
            v = obj.get(key)
            if isinstance(v, list):
                for item in v:
                    yield item
                return
        # if unknown, fall back to recursive search below
        for plan in _recursive_plan_candidates(obj):
            yield plan


def _recursive_plan_candidates(obj: Any) -> Iterable[Any]:
    # Generic recursive search for dicts that look like plan objects
    # Heuristic: a dict that has at least one of the plan-ish keys OR
    # has an init-satisfiable flag.
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            keys = set(cur.keys())
            if keys & {"plan", "actions", "action_sequence", "steps", "plan_length", "num_actions"}:
                yield cur
            elif keys & {"initial_state_satisfied", "init_satisfied", "satisfiable_initial_state", "initial_sat", "init_sat"}:
                yield cur
            for v in cur.values():
                stack.append(v)
        elif isinstance(cur, list):
            stack.extend(cur)


def _iter_plans_from_main_json(obj: Any) -> Iterable[Any]:
    # Try common containers first
    if isinstance(obj, dict):
        for key in ("plans", "plan_candidates", "results", "trajectories", "samples"):
            v = obj.get(key)
            if isinstance(v, list):
                for item in v:
                    yield item
                return
    # Fallback to recursive scan
    for plan in _recursive_plan_candidates(obj):
        yield plan


def _file_id_from_name(p: Path, data: Dict[str, Any]) -> str:
    return data.get("file", p.stem + (".txt" if p.name.endswith("_stats.json") else ""))


def _aggregate_from_iterable(files_iter: Iterable[Tuple[Path, Dict[str, Any]]], plan_iter_func, count_unknown_as_sat: bool) -> Tuple[Dict[int, int], Dict[int, int], List[Dict[str, Any]], int]:
    plans_by_length: Dict[int, int] = {}
    files_by_length: Dict[int, int] = {}
    file_rows: List[Dict[str, Any]] = []
    files_count = 0

    for p, s in files_iter:
        files_count += 1
        file_id = _file_id_from_name(p, s)
        sat_plans_lengths: List[int] = []

        for plan in plan_iter_func(s):
            sat = _plan_is_init_satisfiable(plan)
            if sat is False:
                continue
            if sat is None and not count_unknown_as_sat:
                continue
            length = _extract_plan_length(plan)
            if length is None:
                continue
            sat_plans_lengths.append(length)

        # Only count the shortest plan per file
        if sat_plans_lengths:
            min_len = min(sat_plans_lengths)
            plans_by_length[min_len] = plans_by_length.get(min_len, 0) + 1
            files_by_length[min_len] = files_by_length.get(min_len, 0) + 1
            file_rows.append({
                "file": file_id,
                "min_satisfiable_plan_length": min_len,
                "all_satisfiable_plan_lengths": sorted(sat_plans_lengths),
            })
        else:
            file_rows.append({
                "file": file_id,
                "min_satisfiable_plan_length": None,
                "all_satisfiable_plan_lengths": [],
            })

    return plans_by_length, files_by_length, file_rows, files_count


def _aggregate_from_initial_state(files_iter: Iterable[Tuple[Path, str]]) -> Tuple[Dict[int, int], Dict[int, int], List[Dict[str, Any]], int]:
    plans_by_length: Dict[int, int] = {}
    files_by_length: Dict[int, int] = {}
    file_rows: List[Dict[str, Any]] = []
    files_count = 0

    for p, text in files_iter:
        files_count += 1
        file_id = p.name.replace("_initial_state.txt", ".txt")
        length = _extract_from_initial_state_text(text)

        # Always record the first-block length (possibly 0) and include it in the distribution
        plans_by_length[length] = plans_by_length.get(length, 0) + 1
        files_by_length[length] = files_by_length.get(length, 0) + 1
        file_rows.append({
            "file": file_id,
            "first_actions_block_length": length,
        })

    return plans_by_length, files_by_length, file_rows, files_count


def summarize_plan_lengths(stats_dir: Path, source: str = "auto", count_unknown_as_sat: bool = False) -> Dict[str, Any]:
    # Try by source preference. Prefer initial_state first since user requested it.
    attempts_any = []  # type: List[Tuple[str, Any, Any]]
    if source in ("auto", "initial_state"):
        attempts_any.append(("initial_state", load_initial_state_txt_files(stats_dir), _aggregate_from_initial_state))
    if source in ("auto", "stats"):
        attempts_any.append(("stats", load_stats_files(stats_dir), lambda it: _aggregate_from_iterable(it, _iter_plans_from_stats, count_unknown_as_sat)))
    if source in ("auto", "val_summary"):
        attempts_any.append(("val_summary", load_val_summary_files(stats_dir), lambda it: _aggregate_from_iterable(it, _iter_plans_from_val_summary, count_unknown_as_sat)))
    if source in ("auto", "json"):
        attempts_any.append(("json", load_main_json_files(stats_dir), lambda it: _aggregate_from_iterable(it, _iter_plans_from_main_json, count_unknown_as_sat)))

    aggregate = None
    chosen_source = None
    files_count = 0
    file_rows: List[Dict[str, Any]] = []
    plans_by_length: Dict[int, int] = {}
    files_by_length: Dict[int, int] = {}

    for label, files_iter, aggregator in attempts_any:
        if label == "initial_state":
            plans_by_length, files_by_length, file_rows, files_count = aggregator(files_iter)
        else:
            plans_by_length, files_by_length, file_rows, files_count = aggregator(files_iter)
        if plans_by_length:
            aggregate = (plans_by_length, files_by_length, file_rows, files_count)
            chosen_source = label
            break
        if source != "auto" and label == source:
            aggregate = (plans_by_length, files_by_length, file_rows, files_count)
            chosen_source = label
            break

    if aggregate is None:
        return {"error": f"No candidate files found under {stats_dir}"}

    plans_by_length, files_by_length, file_rows, files_count = aggregate

    if files_count == 0:
        return {"error": f"No files found under {stats_dir}"}

    plans_hist = {int(k): plans_by_length[k] for k in sorted(plans_by_length)}
    files_hist = {int(k): files_by_length[k] for k in sorted(files_by_length)}

    return {
        "directory": str(stats_dir),
        "files_count": files_count,
        "source": chosen_source,
        "plans_satisfiable_by_length": plans_hist,
        "files_with_any_satisfiable_plan_of_length": files_hist,
        "files": file_rows,
    }


ESSENTIAL_FIELDS = [
    "source",
    "plans_satisfiable_by_length",
    "files_with_any_satisfiable_plan_of_length",
]


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Summarize plan lengths (action-sequence length) for plans that are "
            "satisfiable by the initial condition. Supports *_initial_state.txt, *_stats.json, *_val_summary.json, and main *.json."
        )
    )
    ap.add_argument(
        "--dir",
        dest="stats_dir",
        default=str(Path(__file__).parent / "out_pipeline/results_10_sep23"),
        help="Directory containing result files (default: ./out_pipeline/results_10_sep23)",
    )
    ap.add_argument(
        "--source",
        dest="source",
        choices=["auto", "initial_state", "stats", "val_summary", "json"],
        default="auto",
        help="Which files to read: auto (initial_state → stats → val_summary → json), or a specific source",
    )
    ap.add_argument(
        "--count-unknown-as-sat",
        dest="count_unknown_as_sat",
        action="store_true",
        help="Count plans without an explicit init-satisfiable flag as satisfiable (for JSON sources)",
    )
    ap.add_argument(
        "--write-json",
        dest="write_json",
        action="store_true",
        help="Also write an aggregate JSON summary next to the stats directory",
    )
    args = ap.parse_args()

    stats_dir = Path(args.stats_dir)
    summary = summarize_plan_lengths(stats_dir, source=args.source, count_unknown_as_sat=args.count_unknown_as_sat)

    if "error" in summary:
        print(summary["error"])
        return

    print("\n=== Plan Length Summary (Init-Satisfiable) ===")
    for k in ESSENTIAL_FIELDS:
        print(f"{k}: {summary[k]}")

    if args.write_json:
        out_path = stats_dir / "aggregate_plan_length_summary.json"
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"\nWrote aggregate plan-length summary -> {out_path}")


if __name__ == "__main__":
    main()
