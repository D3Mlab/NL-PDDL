import yaml
import numpy as np
from PIL import Image

import os
import json

import pdb

from alfworld.agents.environment import get_environment
import alfworld.agents.modules.generic as generic

from parser import ALFWorldProblemParser
# from planner import RegressionPlanningAgent

from pprint import pprint

import time 

from tqdm import tqdm


def extract_valid_files(data_path):

    TASK_TYPES = {1: "pick_and_place_simple",
              2: "look_at_obj_in_light",
              3: "pick_clean_then_place_in_recep",
              4: "pick_heat_then_place_in_recep",
              5: "pick_cool_then_place_in_recep",
              6: "pick_two_obj_and_place"}

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
            if not traj_data['task_type'] in TASK_TYPES.values():
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


with open('./configs/base_config.yaml') as reader:
    config = yaml.safe_load(reader)

from regression_agent.agent_NL import RegressionAgentNL

# help functions

env_type = config['env']['type'] # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'
# env_type = 'AlfredThorEnv'
env_type = 'AlfredTWEnv'
# env_type = 'AlfredHybrid'

# data_path = './dev_data/'
# data_path = './dev_data_single/'
# data_path = '$ALFWORLD_DATA/json_2.1.1/valid_unseen'
data_path = '/home/user/research/data_folder/alfworld/json_2.1.1/valid_unseen'

# inifinite loop
# data_path = '/home/user/research/data_folder/alfworld/json_2.1.1/valid_unseen/pick_clean_then_place_in_recep-Spatula-None-Drawer-10'

# light bug
# data_path = '/home/user/research/data_folder/alfworld/json_2.1.1/valid_unseen/look_at_obj_in_light-Mug-None-DeskLamp-308'

# pick two bug
# data_path = '/home/user/research/data_folder/alfworld/json_2.1.1/valid_unseen/pick_two_obj_and_place-SoapBar-None-Cabinet-424'


config['dataset']['data_path'] = data_path

# setup environment
env = get_environment(env_type)(config, train_eval='train')
env = env.init_env(batch_size=1)


gamefiles, jsonfiles = extract_valid_files(data_path)

print("Game files: ", gamefiles)
print("Json files: ", jsonfiles)    


success_tasks = []
failed_tasks = []

# env.json_file_list = [jsonfiles[0]]

# pdb.set_trace()


# for i in range(10):

for idx in range(len(jsonfiles)):
# for idx in range(10):

    # try:

        domain_path = '/home/user/research/NL-PDDL/NL_ALFWorld/alfworldtext_domain.json'
        # domain_path = './alfworldtext_domain_before_light_fix.json'

        reg_agent = RegressionAgentNL(domain_path=domain_path,\
                                        env=env, llm_model="gpt-4o-mini")

        # if env_type == 'AlfredThorEnv':
        env.json_file_list = jsonfiles

        print('-------------------------current json file: ', jsonfiles[idx])

       

        # interact
        obs, info = env.reset()

        

        NL_goal = reg_agent.parse_goal(info)



        

        reg_agent.set_solver()

        

        reg_agent.generate_policy(max_plan_length=3)

        # reg_agent.calculate_plan_prob(n_response=1)
        # for p in reg_agent.kb_policy.plans:
        #      print("Plan: ", p)
        #      print("Plan Prob: ", p.llmprob)
        #      for e in p.events:
        #           print("\t Predicate: ", e.predicates, "probs: ", e.llmprob)
        #           print("\t Event: ", e.description)
        #           print("\t VOI: ", reg_agent.calculate_voi(e))
        #      print("----------------------")

        print(obs[0])

        # pdb.set_trace()

        reg_agent.update_initial_observation(obs[0])

        # for i in reg_agent.kb_policy.plans:
        #     print(i)
        #     print(i.regression_plan.root_subgoal.formula)
        #     print("Subgoals: ", i.subgoal_pos.keys())
        #     print("Subgoal Queries: ", i.subgoal_query)
        #     print("---------------------")

        done = False

        while reg_agent.step_count < 50 and not done:

            reg_agent.generate_plan_query()
            action_type, plan_list = reg_agent.gen_kb_action()

            if action_type == "explore":
                env_action = reg_agent.gen_env_explore_action(criteria="random")
                success, feedback, failed_action, env_action, done = reg_agent.step(env_action)

                # reg_agent.get_frame()
                
                # print("Env Action: ", env_action)
                # # print("Feedback: ", feedback)
                # # print("After Location", reg_agent.current_location)
                # print("---------------------")

                if success:
                    reg_agent.update_observation(feedback, gen_type="gt")

            elif action_type == "gen_relationships":

                # for p in plan_list:
                #     print("relationship plan: ", p)

                reg_agent.generate_relationships(plan_list)

            elif action_type == "plan_action":

                plan = reg_agent.choose_excutable_plan(plan_list)

                # for i in plan_list:
                #     print(i)

                query_str = plan.get_query_str()
                query_vars = plan.extract_unique_objects_query(query_str)
                query_answer = plan.get_grounding(query_str)[0]
                var_mapping = {query_vars[i]: query_answer[i] for i in range(len(query_vars))}

                # print("*****plan action*****")
                # print("Plan: ", plan)
                # for f in reg_agent.kb.obs_facts:
                #     if 'is_a' not in f and 'can_contain' not in f:
                #         print(f)

                for action in plan.actions:

             
                    env_action = reg_agent.plan_action2env_action(action, var_mapping)

                    success, feedback, failed_action, env_action, done = reg_agent.step(env_action)

                    print("Env Action: ", env_action)
                    print("Feedback: ", feedback)
                    print('---------------------')

                    # pdb.set_trace()

                    # print(env_action, feedback, success)
                    # print(feedback, success)

                    if success:
                        reg_agent.success_env_actions.add(env_action)
                        # reg_agent.update_plan_action()
                    else:
                        print('Failed Action: ', env_action)
                        reg_agent.failed_env_actions.add(env_action)
                        reg_agent.update_failed_action(failed_action)
                        # print(env_action, "failed")
                        ## need to remove affordance and other actions
                        break

                    if done:
                        break

                    # if reg_agent.step_count > 50:
                    #     done = False
                    #     break
            
            else:
                raise NotImplementedError(action_type, " not implemented")

            if done == True:
                print('Success: ', done)
                print('eps length: ', reg_agent.step_count)

                gamefile = info['extra.gamefile'][0]
                traj_dir = os.path.dirname(gamefile)
                success_tasks.append(traj_dir)

                # progress after success
                total = len(success_tasks) + len(failed_tasks)
                pct = (len(success_tasks) / total * 100.0) if total > 0 else 0.0
                print(f"Progress: {len(success_tasks)} success / {len(failed_tasks)} fail ({pct:.1f}% success)")

                break

        if done == False:
            print('Success: ', done)
            print('eps length: ',  reg_agent.step_count)

            gamefile = info['extra.gamefile'][0]
            traj_dir = os.path.dirname(gamefile)
            failed_tasks.append(traj_dir)

            # progress after failure
            total = len(success_tasks) + len(failed_tasks)
            pct = (len(success_tasks) / total * 100.0) if total > 0 else 0.0
            print(f"Progress: {len(success_tasks)} success / {len(failed_tasks)} fail ({pct:.1f}% success)")

            print('-------------------------current json file: ', jsonfiles[idx])
            pdb.set_trace()

        # pdb.set_trace()

        # break

        # break

    # except Exception as e:
    #     # Log error and count as a failure for this index
    #     print("Error during execution:", repr(e))
    #     # Try to recover the directory that failed
    #     try:
    #         gamefile = info['extra.gamefile'][0]
    #         traj_dir = os.path.dirname(gamefile)
    #     except Exception:
    #         # Fallback to jsonfiles/gamefiles indexing if info isn't available
    #         if idx < len(gamefiles):
    #             traj_dir = os.path.dirname(gamefiles[idx])
    #         elif idx < len(jsonfiles):
    #             traj_dir = os.path.dirname(jsonfiles[idx])
    #         else:
    #             traj_dir = f"<unknown-{idx}>"
    #     failed_tasks.append(traj_dir)

    #     # progress after exception
    #     total = len(success_tasks) + len(failed_tasks)
    #     pct = (len(success_tasks) / total * 100.0) if total > 0 else 0.0
    #     print(f"Progress: {len(success_tasks)} success / {len(failed_tasks)} fail ({pct:.1f}% success)")

# --- After loop: save all failed task directories ---
with open("failed_tasks.txt", "w") as f:
    for traj_dir in failed_tasks:
        f.write(traj_dir + "\n")
print(f"Saved {len(failed_tasks)} failed tasks to failed_tasks.txt")
