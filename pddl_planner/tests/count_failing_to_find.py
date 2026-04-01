import argparse
import os
import re


def infer_dataset_and_depth(logs_dir: str) -> tuple[str, int]:
    """
    Infer dataset prefix and depth from a logs directory name like:
    .../logs/<dataset_prefix>_results_depth<depth>
    """
    base = os.path.basename(logs_dir.rstrip("/"))
    # Expect pattern like "<prefix>_results_depth<depth>"
    m = re.match(r"(.+)_results_depth(\d+)$", base)
    if not m:
        raise ValueError(f"Cannot infer dataset/depth from directory name: {base}")
    dataset = m.group(1)
    depth = int(m.group(2))
    return dataset, depth


def count_failing_lines_per_file(logs_dir: str) -> list[tuple[int, int, str]]:
    """
    Count lines containing 'Failing to find' per file in logs_dir.
    Returns a list of tuples: (problem_index, count, filename).
    """
    results: list[tuple[int, int, str]] = []
    file_regex = re.compile(r".*_(\d+)\.txt$")
    for entry in sorted(os.listdir(logs_dir)):
        path = os.path.join(logs_dir, entry)
        if not os.path.isfile(path):
            continue
        # Only consider text-like files
        if not entry.endswith(".txt"):
            continue
        m = file_regex.match(entry)
        if not m:
            continue
        idx = int(m.group(1))
        count = 0
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "Failing to find" in line:
                        count += 1
        except Exception:
            # Skip unreadable files
            continue
        results.append((idx, count, entry))
    # sort by problem index for stable output
    results.sort(key=lambda x: x[0])
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Count 'Failing to find' lines in planner logs and append to llm_calls.txt")
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="pddl_planner/tests/logs/misalignment_blockworld_results_depth4",
        help="Path to the logs directory for a specific dataset/depth",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="pddl_planner/tests/results",
        help="Root directory containing per-dataset result folders",
    )
    args = parser.parse_args()

    dataset, depth = infer_dataset_and_depth(args.logs_dir)
    per_file = count_failing_lines_per_file(args.logs_dir)
    # Print summary to stdout
    for idx, cnt, name in per_file:
        print(f"{name}: {cnt}")

    results_dir = os.path.join(args.results_root, f"{dataset}_results_depth{depth}")
    os.makedirs(results_dir, exist_ok=True)
    calls_file = os.path.join(results_dir, "llm_calls.txt")

    with open(calls_file, "a") as f:
        # dataset, depth, problem_index, count
        for idx, cnt, _ in per_file:
            f.write(f"{dataset},{depth},{idx},{cnt}\n")


if __name__ == "__main__":
    main()


