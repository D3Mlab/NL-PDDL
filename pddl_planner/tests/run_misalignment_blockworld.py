import time
import json
import os
from dotenv import load_dotenv

from pddl_planner.planner.nl_planner import NLFOLRegressionPlanner
from pddl_planner.logic.nl_parser import NLParser
from pddl_planner.logic.formula import DisjunctiveFormula

if __name__ == "__main__":
    # load the domain from the file
    with open('files/misalignment_blockworld/misalignment_blockworld_model.json', 'r') as f:
        domain = json.load(f)

    # load the goals from the file (note that the goal are the same as the blockworld goal)
    with open('files/blockworld/blockworld_goal.json', 'r') as f:
        all_goals = json.load(f)

    # create a directory to store the results
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # initialize the planner
    max_depth = 10

    # save path
    save_folder_path = os.path.join(current_dir, f'results/misalignment_blockworld_results_depth{max_depth}')
    log_folder_path = os.path.join(current_dir, f'logs/misalignment_blockworld_results_depth{max_depth}')

    # llm cache path
    llm_cache_path = os.path.join(current_dir, f'llm_cache/misalignment_blockworld.json')
    #check if the cache json file exists
    if not os.path.exists(llm_cache_path):
        os.makedirs(os.path.dirname(llm_cache_path), exist_ok=True)
        with open(llm_cache_path, 'w') as f:
            json.dump({}, f)

    os.makedirs(save_folder_path, exist_ok=True)
    os.makedirs(log_folder_path, exist_ok=True)

    for i, problem in enumerate(all_goals[0:1]):
        init_state, goal = problem
        save_file_path = os.path.join(save_folder_path, f'misalignment_blockworld_results_{i}.txt')
        log_file_path = os.path.join(log_folder_path, f'misalignment_blockworld_results_depth{max_depth}_{i}.txt')


        # save the initial state
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

        print(f'Problem {goal} =========================================')
        planner = NLFOLRegressionPlanner(domain.copy(), goal.copy(), None, max_depth=max_depth, log_path=log_file_path, time_limit=None, cache_path=llm_cache_path)
        start_time = time.time()
        regressed_plans = planner.regress_plan(save_file_path=save_file_path)
        end_time = time.time()
        print(f'Time taken: {end_time - start_time} seconds')