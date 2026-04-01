#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Any


def load_stats(stats_dir: Path):
    for p in sorted(stats_dir.glob("*_stats.json")):
        try:
            with p.open() as f:
                data = json.load(f)
            yield p, data
        except Exception as e:
            print(f"Warning: failed to read {p}: {e}")


def summarize(stats_dir: Path) -> Dict[str, Any]:
    count_files = 0

    sum_val_valid = 0
    sum_val_invalid = 0
    sum_val_total = 0

    sum_init_satisfied = 0
    sum_init_total = 0

    files_with_any_init_satisfied = 0
    files_with_none_init_satisfied = 0

    # Plan-level tallies (across all files)
    plans_satisfied = 0
    plans_unsatisfied = 0

    file_rows = []

    for p, s in load_stats(stats_dir):
        count_files += 1

        val_valid = int(s.get("val_valid", 0) or 0)
        val_invalid = int(s.get("val_invalid", 0) or 0)
        val_total = int(s.get("val_total", val_valid + val_invalid) or 0)

        init_sat = int(s.get("initial_state_satisfied", 0) or 0)
        init_tot = int(s.get("initial_state_total", 0) or 0)

        sum_val_valid += val_valid
        sum_val_invalid += val_invalid
        sum_val_total += val_total

        sum_init_satisfied += init_sat
        sum_init_total += init_tot

        plans_satisfied += init_sat
        plans_unsatisfied += max(init_tot - init_sat, 0)

        if init_sat > 0:
            files_with_any_init_satisfied += 1
        else:
            files_with_none_init_satisfied += 1

        file_rows.append({
            "file": s.get("file", p.name.replace("_stats.json", ".txt")),
            "val_valid": val_valid,
            "val_invalid": val_invalid,
            "val_total": val_total,
            "initial_state_satisfied": init_sat,
            "initial_state_total": init_tot,
        })

    if count_files == 0:
        return {
            "error": f"No *_stats.json files found under {stats_dir}",
        }

    avg_val_valid = sum_val_valid / count_files
    avg_val_invalid = sum_val_invalid / count_files
    avg_init_satisfied = sum_init_satisfied / count_files

    out = {
        "directory": str(stats_dir),
        "files_count": count_files,

        # Averages per file
        "avg_val_valid_per_file": avg_val_valid,
        "avg_val_invalid_per_file": avg_val_invalid,
        "avg_initial_state_satisfied_per_file": avg_init_satisfied,

        # Totals across all files
        "total_val_valid": sum_val_valid,
        "total_val_invalid": sum_val_invalid,
        "total_val_plans": sum_val_total,

        "total_initial_state_satisfied": sum_init_satisfied,
        "total_initial_state_plans": sum_init_total,

        # File counts by satisfiable subgoal presence
        "files_with_any_satisfiable_subgoal": files_with_any_init_satisfied,
        "files_with_no_satisfiable_subgoal": files_with_none_init_satisfied,

        # Plan-level satisfied vs unsatisfied (across all files)
        "plans_satisfied": plans_satisfied,
        "plans_unsatisfied": plans_unsatisfied,

        # Optional per-file rows if needed for CSV-like inspection
        "files": file_rows,
    }
    return out


essential_fields = [
    "avg_val_valid_per_file",
    "avg_val_invalid_per_file",
    "avg_initial_state_satisfied_per_file",
    "files_with_any_satisfiable_subgoal",
    "files_with_no_satisfiable_subgoal",
    "plans_satisfied",
    "plans_unsatisfied",
]


def main():
    ap = argparse.ArgumentParser(description="Summarize *_stats.json files in out_pipeline")
    ap.add_argument("--dir", dest="stats_dir", default=str(Path(__file__).parent / "out_pipeline/results_10_sep23"),
                    help="Directory containing *_stats.json files (default: ./out_pipeline)")
    ap.add_argument("--write-json", dest="write_json", action="store_true",
                    help="Also write an aggregate JSON summary next to the stats")
    args = ap.parse_args()

    stats_dir = Path(args.stats_dir)
    summary = summarize(stats_dir)

    if "error" in summary:
        print(summary["error"])
        return

    print("\n=== Aggregate Summary ===")
    for k in essential_fields:
        print(f"{k}: {summary[k]}")

    if args.write_json:
        out_path = stats_dir / "aggregate_summary.json"
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"\nWrote aggregate summary -> {out_path}")


if __name__ == "__main__":
    main()
