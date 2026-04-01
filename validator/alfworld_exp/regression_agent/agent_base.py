# from .regression.solver import RegressionSolver
# from .regression.domain import RegressionDomain
# from .regression.state import RegressionState
# from .regression.action import RegressionAction
# from .regression.solver import RegressionSolver

# from .plan.plan import ThorKBPlan, ThorKBPolicy

from kb import ThorKB, ThorKBFact, ThorKBObject, ThorKBPolicy

# from pddl.logic import Predicate, constants, variables

import re
import io

from pprint import pprint

import pdb

import random

import os

import json

from actions import ALFWorldTextAction, EnvExploreAction, EnvInteractAction  

from parser import ALFWorldProblemParser, ALFWorldTextObsParser, ALFWorldVLMObsParser
from regression import ALFWorldRegressionAgent
from llm import LLMExplorer

import numpy as np
from PIL import Image

from openai import OpenAI
import base64


class RegressionAgentBase:
    def __init__(self, 
                 domain_path, 
                 env,
                 max_plan_length: int = 5,
                 llm_model: str = "gpt-4o-mini"):
        
    

        self.domain_path = domain_path
        self.problem_path = None

        self.planner = None

        self.llm_model = llm_model

        self.max_plan_length = max_plan_length

        self.llm_explore = LLMExplorer(model=llm_model)


        self.current_location = 'start'

        self.holding = None

        self.kb = ThorKB()

        self.ingore_preds = ["holdsany", "iscontained", "handempty", "holds"]

        self.object_types = ["cooling", "heating", "cleaning", "lighting"]

        self.affordance_preds = ["canheat", "cancool", "canclean", "canpickup", "canlight", "cancontain"]

        self.action2state_dict = {"heat":"ishot", "cool":"iscool", "clean":"isclean", "light":"islight"}
        self.action2affordance_dict = {"heat":"canheat", "cool":"cancool", "clean":"canclean", "light":"canlight"}

        self.kb_policy = None

        self.goal_parser = ALFWorldProblemParser()

        self.all_explore_actions = set()

        self.success_explore_actions = set()
        self.success_env_actions = set()
        self.failed_explore_actions = set()
        self.failed_env_actions = set()

        self.current_plan = None

        self.current_grounded_plan = None
        
        self.current_action = None
        self.current_query_model = None

        self.all_objects = {}
        self.objects = {}
        self.receptacles = {}
        self.openable_receptacles = {}

        self.all_affordance_facts = set()
        self.current_affordance_facts = set()
        self.failed_affordance_facts = set()
        self.success_affordance_facts = set()

        self.text_obs_parser = ALFWorldTextObsParser()
        self.visual_obs_parser = ALFWorldVLMObsParser()

        self.env = env

        self.task_type = None

        self.step_count = 0

        self.goal_text = None

        self.goal_objects = None

        self.failed_relationships = set()

        self.all_facts = set()
        

    def update_plan_action(self):
        self.current_plan.take_action()


    def parse_goal(self, info):
        gamefile = info['extra.gamefile'][0]

        

        traj_dir = os.path.dirname(gamefile)
        json_file = os.path.join(traj_dir, 'traj_data.json')
        # json_file = gamefile+'/traj_data.json'

        # pdb.set_trace()

        with open(json_file, 'r') as f:
            traj_data = json.load(f)
        goal_params = traj_data['pddl_params']
        task_type = traj_data["task_type"]
        self.task_type = task_type

        goal_objects = [i for i in goal_params.values() if i not in [None,True,False, ""]]
        goal_objects = [i.lower() for i in goal_objects]

        self.goal_objects = goal_objects
        
        problem_text, problem_path = self.goal_parser.parse_pddl_problem(goal_params, task_type)

        self.problem_path = problem_path



    def set_solver(self):
        self.planner = ALFWorldRegressionAgent(self.domain_path, self.problem_path, 
                                               max_plan_length=self.max_plan_length,
                                               llm_model=self.llm_model)


    def generate_policy(self, max_plan_length=5, max_visited_states=1000):

        self.planner.generate_plans(max_plan_length=max_plan_length)

        regression_policy = self.planner.policy


        policy = ThorKBPolicy(regression_policy, self.kb)

        self.kb_policy = policy

    def calculate_plan_prob(self, n_response=1):
        self.planner.extract_and_score_events(n_responses=n_response)
        for plan in self.kb_policy.plans:
            for event in plan.events:
                event.llmprob = event.regression_event.llmprob
         
        for plan in self.kb_policy.plans:
            regression_plan = plan.regression_plan
            plan.llmprob = self.kb_policy.regression_policy.calculate_plan_probability(regression_plan)
            
        return self.planner.scored_events
    
    def calculate_voi(self, kb_event):
        return self.planner.policy.voi_score(kb_event.regression_event)


    def update_initial_observation(self, obs):

        cleaned_text = obs.replace("-= Welcome to TextWorld, ALFRED! =-", "").strip()
        parts = cleaned_text.split("Your task is to:", 1)

        room_description = parts[0].strip()
        task_description = parts[1].strip()

        self.task_text = task_description


        initial_observation = room_description

        initial_objs = self.text_obs_parser.parse_text_observation(initial_observation, self.current_location)

        for name, typ, state, loc in initial_objs:
            obj = self._update_object(name, typ, state, loc, gen_type='gt')
            self.all_explore_actions.add(EnvExploreAction('go to', obj))

        
        self.generate_plan_query()

    def check_object_property_gt(self, obj_name):
        """
        Check if the object is of a specific type.
        """
        object_properties = []
        if 'fridge' in obj_name:
            object_properties = ['cooling']
        elif 'microwave' in obj_name:
            object_properties = ['heating']
        elif 'sink' in obj_name:
            object_properties = ['cleaning']
        elif 'lamp' in obj_name:
            object_properties = ['lighting']
        
        return object_properties

    
    def check_object_present(self, obj_name):
        object_properties = ['']

    

    def update_observation(self, observation: str, gen_type: str = "gt"):
        """
        Parse a text observation into (name, type, state, location) tuples,
        then update or create the corresponding KB objects and facts.
        """
        if gen_type == "gt":

            objs = self.text_obs_parser.parse_text_observation(observation, self.current_location)

        elif gen_type == "vlm":
            image = self.get_frame()
            voi_events = self.get_voi_objects()

            vlm_objects = self.visual_obs_parser.parse_vlm_observation(observation, image, voi_events)

            for o in vlm_objects:
                obj = self._update_object(o, 'object', 'outside', self.current_location, gen_type)


        else:
            raise ValueError(f"Unknown gen_type '{gen_type}")


        # for name, typ, state, loc in objs:
        #     obj = self._update_object(name, typ, state, loc, gen_type)

        #     if state in ("open", "close"):
        #         self.openable_receptacles[name] = obj
        #         if state == "close":
        #             self.all_explore_actions.add(EnvExploreAction('open', obj))
    
    def get_voi_objects(self):
        # init_objects = self.goal_objects
        plan_objects = []
        best_plan_prob = 0
        best_plan = None
        for plan in self.kb_policy.plans:
            if plan.llmprob > best_plan_prob:
                best_plan = plan
                best_plan_prob = plan.llmprob
        plan_objects = [e for e in best_plan.events if e.type == 'type']
        pos_voi_objects = []
        for p in self.kb_policy.plans:
            for e in p.events:
                if e.type == 'type':
                    voi = self.calculate_voi(e)
                    if voi > 0 and e not in pos_voi_objects:
                        pos_voi_objects.append(e)
                        # print(e)
        
        all_objects = list(set(plan_objects + pos_voi_objects))

        return all_objects

    
        

    
    def get_frame(self):        

        img_arr = self.env.get_frames()[0]
        img_arr = img_arr.astype(np.uint8)
        img_arr = img_arr[..., ::-1]


        return img_arr
    


        # img = Image.fromarray(img_arr, mode="RGB")
        # img.save("output.png")

        # pdb.set_trace()

            


    def _update_object(self, name: str, type: str, state: str, loc: str, gen_type: str):
        # get or create
        if name in self.objects:
            obj = self.objects[name]
            obj.obj_state = state
        else:
            obj = ThorKBObject(
                name=name,
                location=loc,
                obj_type=type,
                obj_state=state,
                sense_type=type
            )
            self.objects[name]    = obj
            self.all_objects[name] = obj

            self.kb.add_object(obj)

            if type == 'object':
                self.objects[name] = obj
            elif type == 'receptacle':
                self.receptacles[name] = obj
            self.all_objects[name] = obj

            observed_fact = ThorKBFact('is'+name.split('_')[0], [obj], sense_type='observation')
            isobject_fact = ThorKBFact('isobject', [obj], sense_type='observation')

            contain_fact = ThorKBFact('cancontained', [self.all_objects[obj.location], obj], sense_type='observation')

            

            self.kb.add_fact(observed_fact)
            self.kb.add_fact(isobject_fact)
            self.kb.add_fact(contain_fact)

            self.all_facts.add(observed_fact)
            self.all_facts.add(isobject_fact)
            self.all_facts.add(contain_fact)

            

            if gen_type == 'gt':
                object_properties = self.check_object_property_gt(obj.name)
                for p in object_properties:
                    obj.obj_property.append(p)
                for i in object_properties:
                    object_fact = ThorKBFact('is'+i+'object', [obj], sense_type='property')
                    self.kb.add_fact(object_fact)
                    self.all_facts.add(object_fact)
            else:
                raise ValueError(f"Unknown gen_type '{gen_type}' in observation")

        return obj




       

    def extract_unique_objects_query(self, input_string):

        objects = re.findall(r'\(([^)]+)\)', input_string)
        split_objects = []
        for obj in objects:
            split_objects.extend(obj.replace(',', '').split())
        
        unique_objects = list(dict.fromkeys(split_objects))
    
        return unique_objects


    def generate_plan_query(self):

        for plan in self.kb_policy.plans:

            # all_event_sat = True
            type_event_sat = True
            relation_event_sat = True   
            

            for event in plan.events:
                event.query_str =  " & ".join([str(pred) for pred in event.predicates])
                event.query_vars = self.extract_unique_objects_query(event.query_str)                
                event.query_answers = self.kb.query_pred(event.query_str).answers

                for i in event.query_answers:
                    if 'None' not in i:
                        event.is_sat = True
                        
                if not event.is_sat and event.type == 'relationship':
                    relation_event_sat = False

                if not event.is_sat and event.type == 'type':
                    type_event_sat = False
            
            plan_query_str = plan.get_query_str()
            plan_sat = 'None' not in plan.get_grounding(plan_query_str)[0]

            plan.plan_sat = plan_sat
            plan.type_event_sat = type_event_sat
            plan.relation_event_sat = relation_event_sat


    def gen_kb_action(self):

        complete_query_plans = []
        type_query_plans = []
        
        for plan in self.kb_policy.plans:
            if plan.plan_sat:
                complete_query_plans.append(plan)
            if plan.type_event_sat:
                type_query_plans.append(plan)
        
        if len(complete_query_plans) > 0:
            return "plan_action", complete_query_plans
        elif len(type_query_plans) > 0:
            return "gen_relationships", type_query_plans
        else:
            return "explore", None
            
            

    def rank_excutable_plans_by_length(self, excutable_plans):

        sorted_plan = sorted(excutable_plans, key=lambda x: len(x.actions), reverse=False)

        return sorted_plan
    
    def choose_query_answer_random(self, query_answers):
        feasible_groundings = []
        for i in query_answers:
            if 'None' not in i:
                feasible_groundings.append(i)
        if feasible_groundings == []:
            raise ValueError('No feasible grounding for {}'.format(query_answers))
        grounding = random.choice(feasible_groundings)
        return grounding
        

    def predstr_to_list(self, pname):
        matches = re.findall(r'\(([^)]+)\)', pname)
        result = matches[0].split(', ')
        return result



    def plan_action2env_action(self, action, var_mapping):
       
        action_str = action.name.lower()

        ## I need to change this with a full ordered variable list when doing regression later
        # if action_str == 'pickupobjectinreceptacle':
        if action_str == 'pickupobject':
            # answer = self.choose_query_answer_random(query_answers)
            # var_mapping = {v:g for v,g in zip(query_vars, answer)}
            obj_traget = None
            recep_target = None
            for pname in action.pos_preds:
                if 'isobject' in pname.lower(): 
                    pvar_list = self.predstr_to_list(pname)
                    try:                 
                        obj_target = self.all_objects[var_mapping[pvar_list[0]]]
                        recep_target = self.all_objects[obj_target.location]
                    except:
                        pdb.set_trace()
                    
                    break

            env_action = EnvInteractAction('take', obj_target, recep_target)
        elif action_str == 'pickupobjectinreceptacle':
            # answer = self.choose_query_answer_random(query_answers)
            # var_mapping = {v:g for v,g in zip(query_vars, answer)}
            obj_traget = None
            recep_target = None
            for pname in action.pos_preds:
                if 'inreceptacle' in pname.lower(): 
                    pvar_list = self.predstr_to_list(pname)
                    obj_target = self.all_objects[var_mapping[pvar_list[1]]]
                    recep_target = self.all_objects[var_mapping[pvar_list[0]]]
                    break

            env_action = EnvInteractAction('take', obj_target, recep_target)
            # action_str_list = env_action.gen_action_str_list()
        elif action_str == 'heatobject':
            # answer = self.choose_query_answer_random(query_answers)
            # var_mapping = {v:g for v,g in zip(query_vars, answer)}
            obj_traget = None
            recep_target = None
            for pname in action.pos_preds:
                if 'canheat' in pname.lower(): 
                    pvar_list = self.predstr_to_list(pname)
                    obj_target = self.all_objects[var_mapping[pvar_list[1]]]
                    recep_target = self.all_objects[var_mapping[pvar_list[0]]]
                    break
            env_action = EnvInteractAction('heat', obj_target, recep_target)
            # action_str_list = env_action.gen_action_str_list()
        elif action_str == 'cleanobject':
            # answer = self.choose_query_answer_random(query_answers)
            # var_mapping = {v:g for v,g in zip(query_vars, answer)}
            obj_traget = None
            recep_target = None
            for pname in action.pos_preds:
                if 'canclean' in pname.lower(): 
                    pvar_list = self.predstr_to_list(pname)
                    obj_target = self.all_objects[var_mapping[pvar_list[1]]]
                    recep_target = self.all_objects[var_mapping[pvar_list[0]]]
                    break
            env_action = EnvInteractAction('clean', obj_target, recep_target)
            # action_str_list = env_action.gen_action_str_list()
        elif action_str == 'coolobject':
            # answer = self.choose_query_answer_random(query_answers)
            # var_mapping = {v:g for v,g in zip(query_vars, answer)}
            obj_traget = None
            recep_target = None
            for pname in action.pos_preds:
                if 'cancool' in pname.lower(): 
                    pvar_list = self.predstr_to_list(pname)
                    obj_target = self.all_objects[var_mapping[pvar_list[1]]]
                    recep_target = self.all_objects[var_mapping[pvar_list[0]]]
                    break
            env_action = EnvInteractAction('cool', obj_target, recep_target)
            # action_str_list = env_action.gen_action_str_list()
        elif action_str == 'lightobject':
            # answer = self.choose_query_answer_random(query_answers)
            # var_mapping = {v:g for v,g in zip(query_vars, answer)}
            obj_traget = None
            recep_target = None
            for pname in action.pos_preds:
                if 'canlight' in pname.lower(): 
                    pvar_list = self.predstr_to_list(pname)
                    obj_target = self.all_objects[var_mapping[pvar_list[1]]]
                    recep_target = self.all_objects[var_mapping[pvar_list[0]]]
                    break
            env_action = EnvInteractAction('light', obj_target, recep_target)
            # action_str_list = env_action.gen_action_str_list()
        elif action_str == 'putobjectinreceptacle':
            

            # answer = self.choose_query_answer_random(query_answers)
            # var_mapping = {v:g for v,g in zip(query_vars, answer)}
            obj_traget = None
            recep_target = None
            for pname in action.add_effects:
                if 'inreceptacle' in pname.lower(): 
                    pvar_list = self.predstr_to_list(pname)
                    obj_target = self.all_objects[var_mapping[pvar_list[0]]]
                    recep_target = self.all_objects[var_mapping[pvar_list[1]]]

                    break
            env_action = EnvInteractAction('put', obj_target, recep_target)
            # action_str_list = env_action.gen_action_str_list()
        else:
            raise NotImplementedError(action_str, " not implemented.")
        

        return env_action            



    
    def generate_affordance_gt(self, question_inputs):
        
        affordance_facts = []
        for question in question_inputs:
            affordance_list = question['aff_fact'].split("&")

            for aff in affordance_list:
                aff = aff.strip()

                if 'heat' in aff:
                    aff_object = ['microwave']
                elif 'cool' in aff:
                    aff_object = ['fridge']
                elif 'clean' in aff:
                    aff_object = ['sinkbasin']
                elif 'light' in aff:
                    aff_object = ['desklamp', 'tablelamp']
                aff_obj_list = [k for k in self.all_objects if k.split('_')[0] in aff_object]

                if question['aff_vars'] == []:
                    
                    subbed_aff_fact = question['aff_fact']

                    pred_name, arguments = subbed_aff_fact.split('(')
                    arguments = arguments.rstrip(')').split(',')
                    arguments = [arg.strip() for arg in arguments]

                    aff_fact = ThorKBFact(pred_name, [self.all_objects[i] for i in arguments], sense_type='hyp_fact')

                    if aff_fact not in self.all_affordance_facts:
                        affordance_facts.append(aff_fact)
                    
                
                else:
                    for o in aff_obj_list:
                        for v in question['aff_vars']:
                            aff = self.replace_var_in_pred(aff, {v:o})
                        subbed_aff_fact = aff
                            # subbed_aff_fact = q['aff_fact'].replace(v, o)

                        pred_name, arguments = subbed_aff_fact.split('(')
                        arguments = arguments.rstrip(')').split(',')
                        arguments = [arg.strip() for arg in arguments]


                        aff_fact = ThorKBFact(pred_name, [self.all_objects[i] for i in arguments], sense_type='hyp_fact')

                        if aff_fact not in self.all_affordance_facts:
                            affordance_facts.append(aff_fact)

        return affordance_facts

    def replace_var_in_pred(self, pred_str, replace_dict):
        pattern = re.compile(r'\b(?:' + '|'.join(re.escape(key) for key in replace_dict.keys()) + r')\b')
    
        # Function to replace using regex
        def replace_match(match):
            return replace_dict[match.group(0)]
        
        # Perform the replacement
        result_string = pattern.sub(replace_match, pred_str)
        
        return result_string
    
    def _choose_plan(self, plan_list, criteria="length"):
        if criteria == "length":
            plan_list = sorted(plan_list, key=lambda x: len(x.actions), reverse=False)
        elif criteria == "prob":
            plan_list = sorted(plan_list, key=lambda x: x.llmprob, reverse=True)
        else:
            raise NotImplementedError(criteria, " not implemented")
        
        return plan_list[0]
    
    def generate_relationships(self, plan_list, gen_type="gt"):
        if gen_type == "gt":
            plan = self._choose_plan(plan_list, criteria="length")
            relationship_events = [e for e in plan.events if e.type == 'relationship']
            self._extract_grounded_facts(plan, relationship_events, gen_type=gen_type)
            # self.current_grounded_plan = plan

        else:
            raise NotImplementedError(gen_type, " relatinship not implemented")

    def _extract_grounded_facts(self, plan, relation_events, gen_type):

        relation_facts = []

        for event in relation_events:
            event_var_dict = {str(i):[] for i in event.query_vars}

            for v in event.query_vars:
                for e in plan.events:
                    e_name = str(e.vars[0])
                    if e.type == 'type' and e_name in event_var_dict:
                        ground_list = [i[0] for i in e.query_answers if i[0] != 'None']
                        event_var_dict[str(e.vars[0])] = ground_list

            arguement_list = []
            for v in event.query_vars:
                arguement_list.append(event_var_dict[str(v)])

            all_argument_pairs = [(self.all_objects[x], self.all_objects[y]) for x in arguement_list[0] for y in arguement_list[1]]
            pred_name = event.predicates[0].pname

            new_facts = []
            for arg in all_argument_pairs:
                fact = ThorKBFact(pred_name, arg, sense_type='hyp_fact')
                if fact not in self.failed_relationships:
                    new_facts.append(fact)

            if gen_type == "gt":
                choosen_fact = random.choice(new_facts)

                self.kb.add_fact(choosen_fact)

                self.all_facts.add(choosen_fact)
            
            else:
                raise NotImplementedError(f"{gen_type},  not implemented")



    def update_affordance_fact(self, plan_list):
        all_affordance_facts = []
        for plan in plan_list:
            query_model = plan.current_query_model

            obs_query = query_model['observation_query']['query']
            obs_query_vars = query_model['observation_query']['query_vars']
            
            affordance_query = query_model['affordance_query']['query']
            affordance_query_vars = query_model['affordance_query']['query_vars']
            affordance_only_vars = [i for i in affordance_query_vars if i not in obs_query_vars]
            
            obs_grounding_list = [i for i in query_model['observation_query']['query_answer'] if 'None' not in i]
            grounding_mappings = []
            question_inputs = []

            for g in obs_grounding_list:
                subst_dict = {v:o for v, o in zip(obs_query_vars, g)}
                grounding_mappings.append(subst_dict)

                obs_query_new = self.replace_var_in_pred(obs_query, subst_dict)
                affordance_query_new = self.replace_var_in_pred(affordance_query, subst_dict)
                question_inputs.append({'obs_fact':obs_query_new , 'aff_fact':affordance_query_new, 'aff_vars':affordance_only_vars })


            affordance_facts = self.generate_affordance_gt(question_inputs)

            for f in affordance_facts:
                self.all_affordance_facts.add(f)

            affordance_candidates = self.all_affordance_facts - self.failed_affordance_facts

            best_aff_facts = self.pick_random_affordance_fact(question_inputs, affordance_candidates)

            for f in best_aff_facts:
                self.kb.add_fact(f)
                self.all_facts.add(f)

        return best_aff_facts
    
    def pick_random_affordance_fact(self, question_inputs, aff_facts_set):

        random_affs = random.sample(aff_facts_set, 1)
        return random_affs
    
        
    def pick_random_explore_action(self):
        feasible_explore_actions = self.all_explore_actions - self.failed_explore_actions - self.success_explore_actions

        if len(feasible_explore_actions) == 0:
            go_to_actions = {i for i in self.all_explore_actions if i.action_type == 'go to'}
            feasible_explore_actions = go_to_actions
        
        return random.choice(list(feasible_explore_actions))
    
    def pick_llm_explore_action(self):
        feasible_explore_actions = self.all_explore_actions - self.failed_explore_actions - self.success_explore_actions
    
        if len(feasible_explore_actions) == 0:
            go_to_actions = {i for i in self.all_explore_actions if i.action_type == 'go to'}
            feasible_explore_actions = go_to_actions
        action_str = str(feasible_explore_actions)
        objects_str = str(self.goal_objects)

        
        explore_string = self.llm_explore.choose_exploration(objects_str, action_str)

        explore_action = random.choice(list(feasible_explore_actions))

        for i in list(feasible_explore_actions):
            if explore_string == str(i):
                explore_action = i
        return explore_action

    def gen_env_explore_action(self, criteria="random"):
        if criteria == "random":
            explore_action = self.pick_random_explore_action()
            return explore_action
        elif criteria == "llm":
            explore_action = self.pick_llm_explore_action()
            return explore_action
        else:
            raise NotImplementedError(criteria, " not implemented")
        
    def choose_excutable_plan(self, plan_list, criteria="random"):
        excuteable_plans = []

        for plan in plan_list:
            if plan.plan_sat:
                excuteable_plans.append(plan)
            # query_str = plan.get_query_str()
            # query_answer = plan.get_grounding(query_str)[0]
            # if 'None' not in query_answer:
            #     excuteable_plans.append(plan)
        
        if criteria == "random":
            plans = self.rank_excutable_plans_by_length(excuteable_plans)
        elif criteria == "llmprob":
            plans = sorted(excuteable_plans, key=lambda x: x.llmprob, reverse=False)
        
        try:
            plan = plans[0]
        except:
            pdb.set_trace()
        
        # self.current_plan = plan
        return plan
           
        
    def update_failed_action(self, failed_action):
        action_str = failed_action['action_str']
        env_action = failed_action['env_action']
        action_type = env_action.action_type
        self.failed_env_actions.add(env_action)

        if action_type in ["heat", "clean", "cool", "light"]:
            aff_name = self.action2affordance_dict[action_type]
            terms = [env_action.recep, env_action.obj]
            failed_affordance = ThorKBFact(aff_name, terms, sense_type='hyp_fact')
            self.kb.remove_fact(failed_affordance)
            self.failed_affordance_facts.add(failed_affordance)

        
    
    _STEP_HANDLERS = {
        'go to':    '_handle_explore',
        'open':     '_handle_explore',
        'take':     '_handle_take',
        'put':      '_handle_put',
        'heat':     '_handle_interact',
        'cool':     '_handle_interact',
        'clean':    '_handle_interact',
        'light':    '_handle_interact',
    }

    def step(self, env_action):
        """
        Execute an EnvAction by dispatching to the appropriate handler.
        Returns: (success, feedback, failed_action, env_action, done)
        """
        cmd_list = env_action.gen_action_str_list(self.current_location)
        if not cmd_list:
            return False, None, None, None, False

        handler_name = self._STEP_HANDLERS.get(env_action.action_type)
        if not handler_name:
            raise NotImplementedError(f"Unknown action_type '{env_action.action_type}'")

        handler = getattr(self, handler_name)
        return handler(env_action, cmd_list)


    #––– CORE UTILS –––#

    def _thor_step(self, cmd: str):
        """Run a single THOR command and return (feedback:str, done:bool)."""
        feedback, _, _, done = self.env.step([cmd])
        self.step_count += 1
        fb   = feedback[0].lower()
        done = done['won'][0]
        return fb, done

    def _build_failed(self, cmd, env_action, fb):
        return {'action_str': cmd, 'env_action': env_action, 'feedback': fb}


    #––– HANDLERS –––#

    def _handle_explore(self, env_action, cmds):
        failed = None
        for cmd in cmds:
            fb, done = self._thor_step(cmd)
            if "nothing happens" in fb:
                failed = self._build_failed(cmd, env_action, fb)
                print(f"{env_action} {cmds} {cmd} failed.")
                if cmd.startswith("open"):
                    self.current_location = env_action.obj.location
                self.failed_explore_actions.add(env_action)
                return False, fb, failed, env_action, done

        # success
        self.success_explore_actions.add(env_action)
        self.current_location = env_action.obj.location

        # last fb/done from loop
        return True, fb, failed or {}, env_action, done
    



    def _handle_take(self, env_action, cmds):
        failed = None
        for cmd in cmds:
            fb, done = self._thor_step(cmd)

            # if we had to go and door was closed, open it first
            if cmd.startswith("go to") and "close" in fb:
                open_cmd = f"open {env_action.recep.thor_name}"
                fb, done = self._thor_step(open_cmd)

            if "nothing happens" in fb and "open" not in cmd and "go to" not in cmd:
                bad_cmd = open_cmd if 'open_cmd' in locals() and "close" in fb else cmd
                failed = self._build_failed(bad_cmd, env_action, fb)
                print(f"{env_action} {cmds} {bad_cmd} failed.")
                self.failed_env_actions.add(env_action)
                # return False, fb, failed, env_action, done

        # success: update holding & KB
        self.current_location = env_action.recep.location
        env_action.obj.location = 'holding'
        self.holding = env_action.obj.name

        hold_f = ThorKBFact("holds", [env_action.obj])
        # in_f   = ThorKBFact("inreceptacle", [env_action.recep, env_action.obj])
        self.kb.add_fact(hold_f)
        self.all_facts.add(hold_f)
        # self.kb.remove_fact(in_f)

        return True, fb, failed or {}, env_action, done


    def _handle_put(self, env_action, cmds):
        failed = None

        for cmd in cmds:
            fb, done = self._thor_step(cmd)

            # maybe open if needed
            if cmd.startswith("go to") and "close" in fb:
                open_cmd = f"open {env_action.recep.thor_name}"
                fb, done = self._thor_step(open_cmd)

            if "nothing happens" in fb and "open" not in cmd and "go to" not in cmd:
                bad_cmd = open_cmd if 'open_cmd' in locals() and "close" in fb else cmd
                failed = self._build_failed(bad_cmd, env_action, fb)
                print(f"{env_action} {cmds} {bad_cmd} failed.")
                self.failed_env_actions.add(env_action)
                # return False, fb, failed, env_action, done

        # success: update KB and internal state
        env_action.obj.location = env_action.recep.location
        self.holding = None
        self.current_location = env_action.recep.location

        hold_f = ThorKBFact("holds", [env_action.obj])
        in_f   = ThorKBFact("inreceptacle", [env_action.recep, env_action.obj])
        self.kb.remove_fact(hold_f)
        self.kb.add_fact(in_f)
        self.all_facts.add(in_f)

        # clear stale facts
        self.kb.remove_fact(in_f)
        obs_f = ThorKBFact(f"is{env_action.obj.name.split('_')[0]}", [env_action.obj])
        self.kb.remove_fact(obs_f)

        # mark failed exploration for this receptacle (re-open next time)
        from_env = EnvExploreAction('go to', env_action.recep)
        open_env = EnvExploreAction('open', env_action.recep)
        self.failed_explore_actions.update({from_env, open_env})

        return True, fb, failed or {}, env_action, done


    def _handle_interact(self, env_action, cmds):
        failed = None
        for cmd in cmds:
            fb, done = self._thor_step(cmd)
            if "nothing happens" in fb and "open" not in cmd and "go to" not in cmd:
                failed = self._build_failed(cmd, env_action, fb)
                print(f"{env_action} {cmds} {cmd} failed.")
                if not cmd.startswith("go to"):
                    self.current_location = env_action.recep.location
                self.failed_env_actions.add(env_action)
                # return False, fb, failed, env_action, done

        # success: add state fact (heat/cool/clean/light)
        state = self.action2state_dict[env_action.action_type]
        self.current_location = env_action.recep.location
        fact = ThorKBFact(state, [env_action.obj])
        self.kb.add_fact(fact)
        self.all_facts.add(fact)

        return True, fb, failed or {}, env_action, done

    # def step(self, env_action):
    #     current_location = self.current_location
    #     thor_action_str_list = env_action.gen_action_str_list(current_location)

    #     if len(thor_action_str_list) == 0:
    #         return  False, None, None, None, False


    #     if env_action.action_type in ["go to", "open"]:
    #         failed_action = {'action_str':None, 'env_action':None, 'feedback':None}
            

    #         for a in thor_action_str_list:
                
    #             feedback, _, _, done = self.env.step([a])

    #             feedback = feedback[0]
    #             done = done['won'][0]
    #             self.step_count += 1

    #             feedback = feedback.lower()
    #             # feedbacks.append(feedback)

    #             if "nothing happens" in feedback:
    #                 failed_action['action_str'] = a
    #                 failed_action['env_action'] = env_action
    #                 failed_action['feedback'] = feedback
    #                 print(str(thor_action_str_list) + a + " failed.")
    #                 if "go to" not in a:
    #                     self.current_location = env_action.obj.location
    #                 self.failed_explore_actions.add(env_action)

    #                 return (False, feedback, failed_action, env_action, done)
                
    #         self.success_explore_actions.add(env_action)    
    #         self.current_location = env_action.obj.location

    #         return (True, feedback, failed_action, env_action, done)
    #     else:

    #         failed_action = {'action_str':None, 'env_action':None, 'feedback':None}
    #         if env_action.action_type == "take":
                
    #             for a in thor_action_str_list:
    #                 feedback, _, done, done = self.env.step([a])

    #                 feedback = feedback[0]
    #                 done = done['won'][0]

    #                 self.step_count += 1
    #                 feedback = feedback.lower()
    #                 # feedbacks.append(feedback)
    #                 if 'go to' in a:
    #                     if "close" in feedback:
    #                         open_str_action = "open {}".format(env_action.recep.thor_name)
    #                         feedback, _, done, done = self.env.step([open_str_action])

    #                         feedback = feedback[0]
    #                         done = done['won'][0]

    #                         self.step_count += 1
    #                         feedback = feedback.lower()
    #                         # feedbacks.append(feedback)
    #                     if "nothing happens" in feedback:
    #                         failed_action['action_str'] = open_str_action
    #                         failed_action['env_action'] = env_action
    #                         failed_action['feedback'] = feedback
    #                         print(str(thor_action_str_list) + a  + " failed.")
    #                         self.failed_env_actions.add(env_action)
    #                         return (False, feedback, failed_action, env_action, done)
    #                 else:
    #                     if "nothing happens" in feedback:
    #                         failed_action['action_str'] = a
    #                         failed_action['env_action'] = env_action
    #                         failed_action['feedback'] = feedback
    #                         print(str(thor_action_str_list) + a  + " failed.")
    #                         self.current_location = env_action.recep.location
    #                         self.failed_env_actions.add(env_action)
    #                         return (False, feedback, failed_action, env_action, done)

    #             self.current_location = env_action.recep.location
    #             env_action.obj.location = 'holding'
    #             self.holding = env_action.obj.name
          
    #             holding_fact = ThorKBFact("holds", [env_action.obj], sense_type='observation')
    #             inrecep_fact = ThorKBFact("inreceptacle", [env_action.recep, env_action.obj], sense_type='observation')

    #             self.kb.add_fact(holding_fact)
    #             self.kb.remove_fact(inrecep_fact)


    #         elif env_action.action_type == "put":

    #             for a in thor_action_str_list:
    #                 feedback, _, _, done = self.env.step([a])

    #                 feedback = feedback[0]
    #                 done = done['won'][0]


    #                 self.step_count += 1
    #                 feedback = feedback.lower()
    #                 # feedbacks.append(feedback)
    #                 if 'go to' in a:
    #                     if "close" in feedback:
    #                         open_str_action = "open {}".format(env_action.recep.thor_name)
    #                         feedback, _, _, done = self.env.step([open_str_action])

    #                         feedback = feedback[0]
    #                         done = done['won'][0]


    #                         self.step_count += 1
    #                         feedback = feedback.lower()
    #                         # feedbacks.append(feedback)
    #                     if "nothing happens" in feedback:
    #                         # feedbacks.append(feedback)
    #                         failed_action['action_str'] = open_str_action
    #                         failed_action['env_action'] = env_action
    #                         failed_action['feedback'] = feedback
    #                         print(str(thor_action_str_list) + a  + " failed.")
    #                         self.failed_env_actions.add(env_action)
    #                         return (False, feedback, failed_action, env_action, done)
    #                 else:
    #                     if "nothing happens" in feedback:
    #                         # feedbacks.append(feedback)
    #                         failed_action['action_str'] = a
    #                         failed_action['env_action'] = env_action
    #                         failed_action['feedback'] = feedback
    #                         print(str(thor_action_str_list) + a  + " failed.")
    #                         self.current_location = env_action.recep.location
    #                         self.failed_env_actions.add(env_action)
    #                         return (False, feedback, failed_action, env_action, done)

    #             env_action.obj.location = env_action.recep.location
    #             self.holding = None
    #             self.current_location = env_action.recep.location
                
    #             holding_fact = ThorKBFact("holds", [env_action.obj], sense_type='observation')
    #             inrecep_fact = ThorKBFact("inreceptacle", [env_action.recep, env_action.obj], sense_type='observation')

    #             self.kb.remove_fact(holding_fact)
    #             self.kb.add_fact(inrecep_fact)

    #             ## clear put items

    #             self.kb.remove_fact(inrecep_fact)
    #             observed_fact = ThorKBFact('is'+env_action.obj.name.split('_')[0], [env_action.obj], sense_type='observation')
    #             self.kb.remove_fact(observed_fact)
    #             self.failed_explore_actions.add(EnvExploreAction('go to', env_action.recep))
    #             self.failed_explore_actions.add(EnvExploreAction('open', env_action.recep))

            
    #         elif env_action.action_type in ["heat", "cool", "clean", "light"]:

    #             failed_action = {'action_str':None, 'env_action':None, 'feedback':None}
                
    #             for a in thor_action_str_list:
    #                 feedback, _, _, done = self.env.step([a])

                    

    #                 feedback = feedback[0]
    #                 done = done['won'][0]


    #                 self.step_count += 1

    #                 feedback = feedback.lower()
    #                 # feedbacks.append(feedback)
    #                 if "nothing happens" in feedback:
    #                     failed_action['action_str'] = a
    #                     failed_action['env_action'] = env_action
    #                     failed_action['feedback'] = feedback
    #                     if 'go to' not in a:
    #                         self.current_location = env_action.recep.location
    #                     print(str(thor_action_str_list) + a  + " failed.")
    #                     self.failed_env_actions.add(env_action)
    #                     return (False, feedback, failed_action, env_action, done)

                
    #             state_str = self.action2state_dict[env_action.action_type]

    #             self.current_location = env_action.recep.location

    #             state_fact = ThorKBFact(state_str, [env_action.obj], sense_type="observation")

    #             self.kb.add_fact(state_fact)

    #         self.success_env_actions.add(env_action)

    #         return (True, feedback, failed_action, env_action, done)



    

    
