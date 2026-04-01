import argparse
import json
import logging
import os
import time
from typing import Any, List, Tuple

from dotenv import load_dotenv

from pddl_planner.planner.nl_planner import NLFOLRegressionPlanner
from pddl_planner.logic.nl_parser import NLParser
from pddl_planner.logic.formula import DisjunctiveFormula

logger = logging.getLogger("pddl_planner.cli")


def ensure_cache_file(cache_path: str) -> None:
    if cache_path is None:
        return
    if not os.path.exists(cache_path):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump({}, f)


def write_initial_state(save_file_path: str, init_state: Any) -> None:
    with open(save_file_path, 'w') as f:
        if init_state is not None:
            parser = NLParser()
            type_tags = {}
            for pred in init_state:
                type_tags.update(pred[1])
            init_formula = parser.parse_formula(init_state, term_type_dict=type_tags)
            init_formula = DisjunctiveFormula(init_formula).distribute_and_over_or()
            logger.info("Initial state: %s", init_formula)
            f.write("Initial State:\n")
            f.write(str(init_formula) + '\n')
            f.write(str([]) + '\n')
            f.write(str({}) + '\n')
            f.write("--------------------\n")
            f.write("Regressed goals:\n")
        else:
            f.write("Regressed goals:\n")


def infer_result_prefix(model_path: str) -> str:
    # Try dataset directory name (e.g., files/blockworld/blockworld_model.json -> blockworld)
    parent = os.path.basename(os.path.dirname(model_path))
    if parent:
        return parent
    # Fallback to filename stem
    stem = os.path.splitext(os.path.basename(model_path))[0]
    return stem or "nlpddl"


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run NL-PDDL FOL regression planner on NL JSON model & goals")
    parser.add_argument("--model", type=str, default="files/blockworld/blockworld_model.json", help="Path to NL model JSON file")
    parser.add_argument("--goals", type=str, default="files/blockworld/blockworld_goal.json", help="Path to NL goals JSON file")
    parser.add_argument("--max_depth", type=int, default=10, help="Maximum regression depth")
    parser.add_argument("--time_limit", type=int, default=None, help="Time limit per problem in seconds")
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--llm_api_key", type=str, default=None, help="LLM API key (overrides env if provided)")
    parser.add_argument("--cache_path", type=str, default=None, help="Path to LLM cache JSON (will be created if missing)")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to write results files")
    parser.add_argument("--log_dir", type=str, default=None, help="Directory to write log files")
    parser.add_argument("--llm_verbose", action="store_true", help="Enable LLM entailment logs (cache, substitutions, responses)")
    parser.add_argument("--quiet", action="store_true", help="Suppress planner progress logs (only show warnings/errors)")
    parser.add_argument("--limit", type=int, default=1, help="Number of problems to run (from index 0)")
    args = parser.parse_args()

    # Configure root logging for CLI usage with colored output
    from pddl_planner import make_colored_handler
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(make_colored_handler())

    # Load model and goals
    with open(args.model, 'r') as f:
        domain = json.load(f)
    with open(args.goals, 'r') as f:
        all_goals = json.load(f)

    # Defaults for output/log/cache derived from model path
    result_prefix = infer_result_prefix(args.model)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    max_depth = args.max_depth
    time_limit = args.time_limit

    output_dir = args.output_dir or os.path.join(script_dir, f"results/{result_prefix}_results_depth{max_depth}")
    log_dir = args.log_dir or os.path.join(script_dir, f"logs/{result_prefix}_results_depth{max_depth}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    cache_path = args.cache_path or os.path.join(script_dir, f"llm_cache/{result_prefix}.json")
    ensure_cache_file(cache_path)

    # Iterate problems
    limit = max(0, int(args.limit))
    subset = all_goals[0:limit] if limit > 0 else []

    for i, problem in enumerate(subset):
        # Support both (init_state, goal) tuples and goal-only entries
        if isinstance(problem, list) and len(problem) == 2 and isinstance(problem[0], list):
            init_state, goal = problem  # Blockworld-style
        else:
            init_state, goal = None, problem  # Goal-only style

        save_file_path = os.path.join(output_dir, f"{result_prefix}_results_{i}.txt")
        log_file_path = os.path.join(log_dir, f"{result_prefix}_results_depth{max_depth}_{i}.txt")

        write_initial_state(save_file_path, init_state)

        logger.info("Problem %d/%d: %s", i + 1, len(subset), goal)
        planner = NLFOLRegressionPlanner(
            domain.copy(),
            goal.copy(),
            None,
            max_depth=max_depth,
            llm_model=args.llm_model,
            llm_api_key=args.llm_api_key,
            verbose=not args.quiet,
            llm_verbose=args.llm_verbose,
            log_path=log_file_path,
            time_limit=time_limit,
            cache_path=cache_path,
        )

        start_time = time.time()
        _ = planner.regress_plan(save_file_path=save_file_path)
        elapsed = time.time() - start_time
        logger.info("Problem %d completed in %.2fs", i + 1, elapsed)

        # Append missing predicate count to a per-depth file inside results folder
        calls_file_path = os.path.join(output_dir, "llm_calls.txt")
        missing_count = getattr(planner, "_missing_name_count", 0)
        with open(calls_file_path, "a") as cf:
            # dataset(prefix inferred from model), depth, problem_index, missing_predicate_count
            cf.write(f"{result_prefix},{max_depth-1},{i},{missing_count}\n")


if __name__ == "__main__":
    main()
