import argparse
import time
import json
import os
from dotenv import load_dotenv

from pddl_planner.planner.nl_planner import NLFOLRegressionPlanner
from pddl_planner.logic.nl_parser import NLParser
from pddl_planner.logic.formula import DisjunctiveFormula


def ensure_cache_file(cache_path: str) -> None:
    if not os.path.exists(cache_path):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump({}, f)


def write_initial_state_if_any(save_file_path: str, init_state) -> None:
    with open(save_file_path, 'w') as f:
        if init_state is not None:
            parser = NLParser()
            type_tags = {}
            for pred in init_state:
                type_tags.update(pred[1])
            init_formula = parser.parse_formula(init_state, term_type_dict=type_tags)
            init_formula = DisjunctiveFormula(init_formula).distribute_and_over_or()
            print("Initial State:")
            print(init_formula)
            print("Actions:", [])
            print("Substitution:", {})
            print("--------------------")
            f.write("Initial State:\n")
            f.write(str(init_formula) + '\n')
            f.write(str([]) + '\n')
            f.write(str({}) + '\n')
            f.write("--------------------\n")
            f.write("Regressed goals:\n")
        else:
            f.write("Regressed goals:\n")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Unified runner for NL FOL regression planner")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "alfworld_text",
            "alfworld_text_with_misalignment",
            "blockworld",
            "misalignment_blockworld",
            "mystery_blockworld",
            "randomized_blockworld",
        ],
        help="Dataset to run",
    )
    parser.add_argument("--max_depth", type=int, default=10, help="Maximum regression depth")
    parser.add_argument("--max_time", type=int, default=None, help="Maximum time (seconds)")
    parser.add_argument("--limit", type=int, default=1, help="Number of problems to run from the dataset")
    args = parser.parse_args()

    # Map dataset to file locations and behaviors
    dataset_key = args.dataset

    dataset_map = {
        "alfworld_text": {
            "domain": "files/alfworld_text/alfworld_text_model.json",
            "goals": "files/alfworld_text/alfworld_text_goal.json",
            "result_prefix": "alfworld_text",
            "has_init": False,
            "cache": "alfworld_text.json",
        },
        "alfworld_text_with_misalignment": {
            "domain": "files/alfworld_text_with_misalignment/alfworld_text_with_misalignment_model.json",
            "goals": "files/alfworld_text_with_misalignment/alfworld_text_with_misalignment_goal.json",
            "result_prefix": "alfworld_text_with_misalignment",
            "has_init": False,
            "cache": "alfworld_text_with_misalignment.json",
        },
        "blockworld": {
            "domain": "files/blockworld/blockworld_model.json",
            "goals": "files/blockworld/blockworld_goal.json",
            "result_prefix": "blockworld",
            "has_init": True,
            "cache": "blockworld.json",
        },
        "misalignment_blockworld": {
            "domain": "files/misalignment_blockworld/misalignment_blockworld_model.json",
            "goals": "files/blockworld/blockworld_goal.json",
            "result_prefix": "misalignment_blockworld",
            "has_init": True,
            "cache": "misalignment_blockworld.json",
        },
        "mystery_blockworld": {
            "domain": "files/mystery_blocksworld/domain.json",
            "goals": "files/mystery_blocksworld/goal.json",
            "result_prefix": "mystery_blockworld",
            "has_init": True,
            "cache": "blockworld.json",
        },
        "randomized_blockworld": {
            "domain": "files/randomized_blockworld/domain.json",
            "goals": "files/randomized_blockworld/goal.json",
            "result_prefix": "randomized_blockworld",
            "has_init": True,
            "cache": "blockworld.json",
        },
    }

    config = dataset_map[dataset_key]

    # Load domain and goals
    with open(config["domain"], "r") as f:
        domain = json.load(f)
    with open(config["goals"], "r") as f:
        all_goals = json.load(f)

    # Setup folders
    current_dir = os.path.dirname(os.path.abspath(__file__))
    max_depth = args.max_depth
    time_limit = args.max_time

    save_folder_path = os.path.join(current_dir, f"results/{config['result_prefix']}_results_depth{max_depth}")
    log_folder_path = os.path.join(current_dir, f"logs/{config['result_prefix']}_results_depth{max_depth}")
    os.makedirs(save_folder_path, exist_ok=True)
    os.makedirs(log_folder_path, exist_ok=True)

    # LLM cache
    llm_cache_path = os.path.join(current_dir, f"llm_cache/{config['cache']}")
    ensure_cache_file(llm_cache_path)

    # Iterate over requested subset of problems
    num_to_run = max(0, int(args.limit))
    subset = all_goals[0:num_to_run] if num_to_run > 0 else []

    for i, problem in enumerate(subset):
        if config["has_init"]:
            init_state, goal = problem
        else:
            init_state, goal = None, problem

        save_file_path = os.path.join(save_folder_path, f"{config['result_prefix']}_results_{i}.txt")
        log_file_path = os.path.join(log_folder_path, f"{config['result_prefix']}_results_depth{max_depth}_{i}.txt")

        write_initial_state_if_any(save_file_path, init_state)

        print(f"Problem {goal} =========================================")
        planner = NLFOLRegressionPlanner(
            domain.copy(),
            goal.copy(),
            None,
            max_depth=max_depth,
            log_path=log_file_path,
            time_limit=time_limit,
            cache_path=llm_cache_path,
        )

        start_time = time.time()
        _ = planner.regress_plan(save_file_path=save_file_path)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")

        # Append missing predicate count to a per-depth file inside results folder
        calls_file_path = os.path.join(save_folder_path, "llm_calls.txt")
        missing_count = getattr(planner, "_missing_name_count", 0)
        with open(calls_file_path, "a") as cf:
            # dataset, depth, problem_index, missing_predicate_count
            cf.write(f"{config['result_prefix']},{max_depth-1},{i},{missing_count}\n")


if __name__ == "__main__":
    main()


