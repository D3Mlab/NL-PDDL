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

from parser import ALFWorldProblemParser, ALFWorldTextObsParser, ALFWorldVLMObsParser, ALFWorldTextVisionBridge
from regression import ALFWorldRegressionAgentNL
from llm import LLMExplorer

import numpy as np
from PIL import Image

from openai import OpenAI
from contextlib import redirect_stderr
import sys
import base64


class RegressionAgentVLM:
    def __init__(self, 
                 domain_path, 
                 env,
                 max_plan_length: int = 5,
                 llm_model: str = "gpt-4o"):
        
    
        self.domain_path = domain_path
        with open(domain_path, 'r') as f:
            self.domain_NL = json.load(f)


        self.problem_path = None

        self.planner = None

        self.llm_model = llm_model

        self.max_plan_length = max_plan_length

        self.llm_explore = LLMExplorer(model=llm_model)

        # Token usage accounting (prompt/completion tokens)
        self.token_usage = {"prompt": 0, "completion": 0}
        # Register recorder with explorer to capture usage from its OpenAI calls
        try:
            self.llm_explore.set_usage_recorder(self._record_usage)
        except Exception:
            pass

        # Image usage accounting (counts and total bytes sent)
        self.image_usage = {"images_sent": 0, "image_bytes": 0}

        self.current_location = 'start'

        self.holding = None

        self.kb = ThorKB()

        self.ingore_preds = ["holdsany", "iscontained", "handempty", "holds"]

        self.object_types = ["cooling", "heating", "cleaning", "lighting"]

        self.affordance_preds = ["can_heat", "can_chill", "can_wash", "can_pickup", "is_turned_on", "can_contain"]

        self.action2state_dict = {"heat":"is_heated", "cool":"is_chilled", "clean":"is_washed", "light":"is_turned_on"}
        self.action2affordance_dict = {"heat":"can_heat", "cool":"can_chill", "clean":"can_wash", "light":"can_light"}

        self.openable_init_objects = ["cabinet", "drawer", "fridge", "microwave", "safe", "dishwasher", "garbagecan", "laundryhamper", "box"]

        self.kb_policy = None

        self.goal_parser = ALFWorldProblemParser()

        self.all_explore_actions = set()
        self.success_explore_actions = set()
        self.success_env_actions = set()
        self.failed_explore_actions = set()
        self.failed_env_actions = set()

        # Optional grouped action sets used by exploration helpers
        self.all_go_to_actions = set()
        self.all_open_actions = set()

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

        self.NL_goal = None

        # self.client = OpenAI()
        

    def update_plan_action(self):
        self.current_plan.take_action()

    def _record_usage(self, resp):
        """Accumulate token usage for OpenAI-like responses exposing .usage."""
        try:
            usage = getattr(resp, "usage", None)
            if usage:
                self.token_usage["prompt"] += int(getattr(usage, "prompt_tokens", 0) or 0)
                self.token_usage["completion"] += int(getattr(usage, "completion_tokens", 0) or 0)
        except Exception:
            pass

    def get_token_usage(self):
        return dict(self.token_usage)

    def _record_image_usage(self, img_array):
        """Increment image usage counters using an array (H,W,C) as source.
        We approximate payload by encoding to PNG in-memory to reflect typical VLM input.
        """
        try:
            import cv2
            import numpy as np
            if img_array is None:
                return
            arr = img_array
            if isinstance(arr, list):
                arr = np.asarray(arr)
            # Ensure uint8
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)
            # Encode to PNG bytes as a proxy for upload payload size
            ok, buf = cv2.imencode(".png", arr)
            if ok:
                self.image_usage["images_sent"] += 1
                self.image_usage["image_bytes"] += int(buf.size)
        except Exception:
            # Best-effort: still count the image even if encoding fails
            self.image_usage["images_sent"] += 1

    def get_image_usage(self):
        return dict(self.image_usage)

    def parse_goal(self, info):
        gamefile = info['extra.gamefile'][0]

        

        # traj_dir = os.path.dirname(gamefile)
        # json_file = os.path.join(traj_dir, 'traj_data.json')
        json_file = gamefile+'/traj_data.json'

        # pdb.set_trace()

        # pdb.set_trace()

        with open(json_file, 'r') as f:
            traj_data = json.load(f)
        goal_params = traj_data['pddl_params']
        task_type = traj_data["task_type"]
        self.task_type = task_type

        goal_objects = [i for i in goal_params.values() if i not in [None,True,False, ""]]
        goal_objects = [i.lower() for i in goal_objects]

        self.goal_objects = goal_objects
        
        NL_goal = self.goal_parser.parse_nl_pddl_problem(goal_params, task_type)

        self.NL_goal = NL_goal

        return NL_goal

    def parse_goal_llm(self, info, obs_text=None, generated_goal_json='nl_goal_llm.json'):
        gamefile = info['extra.gamefile'][0]

        json_file = gamefile+'/traj_data.json'

        with open(json_file, 'r') as f:
            traj_data = json.load(f)
        goal_params = traj_data['pddl_params']
        task_type = traj_data["task_type"]
        self.task_type = task_type

        goal_objects = [i for i in goal_params.values() if i not in [None,True,False, ""]]
        goal_objects = [i.lower() for i in goal_objects]

        self.goal_objects = goal_objects
        
        cleaned = (obs_text or "").replace("-= Welcome to TextWorld, ALFRED! =-", "").strip()
        if "Your task is to:" in cleaned:
            parts = cleaned.split("Your task is to:", 1)
            desc = parts[1].strip()
            desc = " ".join(desc.split())

        # NL_goal = self.goal_parser.parse_goal_llm(task_type, examples_path="nl_goal_examples.json", model=self.llm_model, obs_text=obs_text)
        with open(generated_goal_json, 'r') as f:
            all_goal = json.load(f)
        
        NL_goal = all_goal[task_type][desc]

        # pdb.set_trace()


        self.NL_goal = NL_goal

        return NL_goal




    def set_solver(self):
        self.planner = ALFWorldRegressionAgentNL(self.domain_NL, self.NL_goal, 
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

        
        # self.generate_plan_query()

    def check_object_property_gt(self, obj_name):
        """
        Check if the object is of a specific type.
        """
        object_properties = []
        if 'fridge' in obj_name:
            object_properties = ['chilling_device']
        if 'microwave' in obj_name:
            object_properties = ['heating_device']
        if 'sink' in obj_name:
            object_properties = ['washing_device']
        if 'lamp' in obj_name:
            object_properties = ['lighting_device']
        
        return object_properties
    
    def check_object_property_llm(self, obj_name):
        # Use GPT-4o-mini to infer whether the object is any of the device types
        # among: chilling_device, heating_device, washing_device, lighting_device.
        # Return a list of zero or more of these strings.
        # import json, re
        # name = (obj_name or "").strip()
        # allowed = {"chilling_device", "heating_device", "washing_device", "lighting_device"}

        # # Short-circuit on obviously empty names
        # if not name:
        #     return []

        # try:
        #     client = OpenAI()
        #     sys_prompt = (
        #         "You are an expert on household objects. Given an object name, "
        #         "decide which device categories apply from this ALLOWED SET ONLY: "
        #         "['chilling_device','heating_device','washing_device','lighting_device']. "
        #         "If none of these apply, say none by returning an EMPTY LIST. "
        #         "Return STRICT JSON with key 'properties' as a list of zero or more of those exact strings. "
        #         "Do NOT include the string 'none' in the list; use an empty list when none apply. "
        #         "Do not include any explanation or extra keys."
        #     )
        #     user_prompt = (
        #         f"Object name: '{name}'.\n"
        #         "Respond ONLY as JSON, e.g., {\"properties\": [\"washing_device\"]} or {\"properties\": []}."
        #     )

        #     resp = client.chat.completions.create(
        #         model="gpt-4o-mini",
        #         messages=[
        #             {"role": "system", "content": sys_prompt},
        #             {"role": "user", "content": user_prompt},
        #         ],
        #         temperature=0.0,
        #         max_tokens=50,
        #     )

        #     text = resp.choices[0].message.content.strip() if resp and resp.choices else "{}"
        #     # Extract JSON payload if wrapped in prose or code fences
        #     if not text.startswith("{"):
        #         start = text.find("{")
        #         end = text.rfind("}")
        #         if start != -1 and end != -1 and end > start:
        #             text = text[start:end+1]
        #     data = json.loads(text)
        #     props = data.get("properties", []) if isinstance(data, dict) else []
        #     # Validate and normalize
        #     out = []
        #     for p in props:
        #         if isinstance(p, str):
        #             val = p.strip().lower()
        #             if val in allowed and val not in out:
        #                 out.append(val)
        #     return out
        # except Exception:
        #     # On any API or parsing failure, return empty and let upstream behave gracefully
        #     return []

        return ['chilling_device','heating_device','washing_device','lighting_device']
        
    

    def update_observation(self, observation: str, gen_type: str = "gt"):
        """
        Parse a text observation into (name, type, state, location) tuples,
        then update or create the corresponding KB objects and facts.
        """


        
        if gen_type in ["gt", "vlm"]:

            objs = self.text_obs_parser.parse_text_observation(observation, self.current_location)


            for name, typ, state, loc in objs:
                obj = self._update_object(name, 'object', 'outside', self.current_location, gen_type)


        elif gen_type == "vlm":
            image = self.get_frame()
            voi_events = self.get_voi_objects()

            # Record image usage when we perform a visual parse
            self._record_image_usage(image)

            vlm_objects = self.visual_obs_parser.parse_vlm_observation(observation, image, voi_events)

            for name, typ, state, loc in vlm_objects:
                obj = self._update_object(name, 'object', 'outside', self.current_location, gen_type)


        else:
            raise ValueError(f"Unknown gen_type '{gen_type}")


        for obj in objs:
            name = obj[0]
            typ = obj[1]
            state = obj[2]
            loc = obj[3]
            if state in ("open", "close"):
                self.openable_receptacles[name] = obj
                if state == "close":
                    self.all_explore_actions.add(EnvExploreAction('open', self.objects[name]))
    
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

        img = self.env.get_frames()[0].astype(np.uint8)
        # img_arr = img_arr[..., ::-1]


        return img
    
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

            # Old Facts
            # observed_fact = ThorKBFact('is)'+name.split('_')[0], [obj], sense_type='observation')
            # isobject_fact = ThorKBFact('isobject', [obj], sense_type='observation')
            # contain_fact = ThorKBFact('cancontained', [self.all_objects[obj.location], obj], sense_type='observation')


            observed_fact = ThorKBFact('is_a_'+name.split('_')[0], [obj], sense_type='observation')
            isobject_fact = ThorKBFact('is_a_object', [obj], sense_type='observation')
            contain_fact = ThorKBFact('can_contain', [self.all_objects[obj.location], obj], sense_type='observation')

            

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

                # if object_properties != []:
                #     print(obj.name, object_properties)

                for i in object_properties:
                    # object_fact = ThorKBFact('is'+i+'object', [obj], sense_type='property')
                    object_fact = ThorKBFact('is_a_'+i, [obj], sense_type='property')
                    self.kb.add_fact(object_fact)
                    self.all_facts.add(object_fact)
            elif gen_type == 'llm':
                object_properties = self.check_object_property_llm(obj.name)
                for p in object_properties:
                    obj.obj_property.append(p)

                # if object_properties != []:
                #     print(obj.name, object_properties)

                for i in object_properties:
                    # object_fact = ThorKBFact('is'+i+'object', [obj], sense_type='property')
                    object_fact = ThorKBFact('is_a_'+i, [obj], sense_type='property')
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

            # print('Plan: ', plan.subgoal_pos)
            # print('Events: ', plan.events)
            # print('----------------')   
            
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
                
                
            # import pdb; pdb.set_trace()

            plan.type_event_sat = type_event_sat
            plan.relation_event_sat = relation_event_sat
            
            plan_query_str = plan.get_query_str()
            plan_sat = 'None' not in plan.get_grounding(plan_query_str)[0]


            # for i in self.kb.obs_facts:
            #     if 'is_a_spatula' in i:
            #         print(i)
                # elif 'is_a_drawer' in i:
                #     print(i)
                # elif 'can_wash' in i:
                #     print(i)


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
                if any(getattr(e, "type", None) == "relationship" for e in plan.events):
                    type_query_plans.append(plan)


        
        if len(complete_query_plans) > 0:
            return "plan_action", complete_query_plans
        elif len(type_query_plans) > 0:
            # for plan in type_query_plans:
            #     print(plan)
            #     print(plan.events)
            #     print(plan.get_query_str())
            #     print(plan.get_grounding(plan.get_query_str()))
            #     print("xxxxxxxxxxxxx")


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
        if 'pick up' in action_str:

            obj_target = None
            recep_target = None

            obj_target = list(action.parameters.keys())[0]

            obj_target = self.all_objects[var_mapping[obj_target]]

            if obj_target.location == 'holding':
                env_action = EnvInteractAction(None, None, None)
            else:
                recep_target = self.all_objects[obj_target.location]

            

                env_action = EnvInteractAction('take', obj_target, recep_target)

        # elif 'pick up from' in action_str:
        #     # answer = self.choose_query_answer_random(query_answers)
        #     # var_mapping = {v:g for v,g in zip(query_vars, answer)}
        #     obj_traget = None
        #     recep_target = None
        #     for pname in action.pos_preds:
        #         if 'inreceptacle' in pname.lower(): 
        #             pvar_list = self.predstr_to_list(pname)
        #             obj_target = self.all_objects[var_mapping[pvar_list[1]]]
        #             recep_target = self.all_objects[var_mapping[pvar_list[0]]]
        #             break

        #     env_action = EnvInteractAction('take', obj_target, recep_target)
            # action_str_list = env_action.gen_action_str_list()
        elif 'heat' in action_str:
            # answer = self.choose_query_answer_random(query_answers)
            # var_mapping = {v:g for v,g in zip(query_vars, answer)}
            obj_traget = None
            recep_target = None
            for pname in action.pos_preds:
                if 'can_heat' in pname.lower(): 
                    pvar_list = self.predstr_to_list(pname)
                    obj_target = self.all_objects[var_mapping[pvar_list[1]]]
                    recep_target = self.all_objects[var_mapping[pvar_list[0]]]
                    break
            env_action = EnvInteractAction('heat', obj_target, recep_target)
            # action_str_list = env_action.gen_action_str_list()
        elif 'wash' in action_str:
            # answer = self.choose_query_answer_random(query_answers)
            # var_mapping = {v:g for v,g in zip(query_vars, answer)}
            obj_traget = None
            recep_target = None
            for pname in action.pos_preds:
                if 'can_wash' in pname.lower():
                    pvar_list = self.predstr_to_list(pname)
                    obj_target = self.all_objects[var_mapping[pvar_list[1]]]
                    recep_target = self.all_objects[var_mapping[pvar_list[0]]]
                    break
            env_action = EnvInteractAction('clean', obj_target, recep_target)
            # action_str_list = env_action.gen_action_str_list()
        elif 'chill' in action_str:
            # answer = self.choose_query_answer_random(query_answers)
            # var_mapping = {v:g for v,g in zip(query_vars, answer)}
            obj_traget = None
            recep_target = None
            for pname in action.pos_preds:
                if 'can_chill' in pname.lower(): 
                    pvar_list = self.predstr_to_list(pname)
                    obj_target = self.all_objects[var_mapping[pvar_list[1]]]
                    recep_target = self.all_objects[var_mapping[pvar_list[0]]]
                    break
            env_action = EnvInteractAction('cool', obj_target, recep_target)
            # action_str_list = env_action.gen_action_str_list()
        elif 'light' in action_str:
            # answer = self.choose_query_answer_random(query_answers)
            # var_mapping = {v:g for v,g in zip(query_vars, answer)}
            obj_traget = None
            recep_target = None
            # pdb.set_trace()
            for pname in action.pos_preds:
                if 'can_light' in pname.lower(): 
                    pvar_list = self.predstr_to_list(pname)
                    
                    obj_target = self.all_objects[var_mapping[pvar_list[1]]]
                    recep_target = self.all_objects[var_mapping[pvar_list[0]]]
                    break
            env_action = EnvInteractAction('light', obj_target, recep_target)
            # action_str_list = env_action.gen_action_str_list()
        elif 'put' in action_str and 'into' in action_str:
            

            # answer = self.choose_query_answer_random(query_answers)
            # var_mapping = {v:g for v,g in zip(query_vars, answer)}
            obj_traget = None
            recep_target = None
            for pname in action.add_effects:
                if 'is_in' in pname.lower(): 
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
        if gen_type in ["gt", "llm"]:
            # plan = self._choose_plan(plan_list, criteria="length")
            added_any = False
            for plan in plan_list:
                before = len(self.all_facts)
                relationship_events = [e for e in plan.events if e.type == 'relationship']

                self._extract_grounded_facts(plan, relationship_events, gen_type=gen_type)
                after = len(self.all_facts)
                if after > before:
                    added_any = True
                # pdb.set_trace()
            if not added_any:
                # pdb.set_trace()
                raise ValueError('No relationship events found in plan_list')
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
                        # print(event_var_dict)
                        # print("ground list: ", ground_list)

            argument_list = []
            for v in event.query_vars:
                argument_list.append(event_var_dict[str(v)])

            # if no type declared, add all objects
            for i in range(len(argument_list)):
                if len(argument_list[i]) == 0:
                    for k,v in self.kb.obs_facts.items():
                        if 'is_a_object' in k:
                            argument_list[i].append(k.split('(')[1].split(')')[0])

            
            # pdb.set_trace()

            # generate affordance facts
            if gen_type == "gt":
                new_affordance_obj = []
                for i in argument_list[0]:
                    if "wash" in event.predicates[0].pname:
                        if "sinkbasin" in i or "sink" in i:
                            new_affordance_obj.append(i)
                    elif "heat" in event.predicates[0].pname:
                        if "microwave" in i:
                            new_affordance_obj.append(i)
                    elif "chill" in event.predicates[0].pname:
                        if "fridge" in i:
                            new_affordance_obj.append(i)
                    elif "light" in event.predicates[0].pname:
                        if "desklamp" in i or "tablelamp" in i:
                            new_affordance_obj.append(i)
                argument_list[0] = new_affordance_obj
            elif gen_type == "llm":
                # Let LLM pick a best affordance object/receptacle candidate.
                new_affordance_obj = []
                target_obj = argument_list[1][0] if argument_list[1] else None
                aff_objs = argument_list[0]
                pname = event.predicates[0].pname.split('(')[0].replace('_', ' ')
                prompt = (
                    "I am an emobided agent, I have observed a list of objects: \n"
                    + str(aff_objs)
                    + " I want to the best object X or receptalce that X " +  pname + " " + str(target_obj) + " is True\n"
                    + ". Please give me the best object or receptacle that would satisfy my objective. \n"
                    + ". Please give the answer in format: obj 1\n"
                )
                if len(self.failed_affordance_facts) > 0:
                    # pdb.set_trace()
                    prompt = (
                        prompt
                        + "Note: Here is a list of objects and receptcales that has failed previously: \n"
                        + " , ".join([str(f) for f in self.failed_affordance_facts])
                    )
                try:
                    messages = [{"role": "user", "content": prompt}]
                    client = OpenAI()
                    response = client.chat.completions.create(
                        model=self.llm_model,
                        messages=messages,
                        max_tokens=128,
                        stop=None,
                        temperature=0,
                    )
                    # Record usage from this call if available
                    self._record_usage(response)
                    answer = response.choices[0].message.content if response and response.choices else ""
                except Exception:
                    answer = ""
                # Normalize the answer: prefer underscore name, strip quotes and punctuation
                if isinstance(answer, list):
                    answer = answer[0] if answer else ""
                ans = str(answer).strip().strip('"').strip("'")
                ans = ans.replace(' ', '_').replace('__', '_')
                # Keep only allowed chars for keys
                import re as _re
                ans = _re.sub(r"[^a-zA-Z0-9_]", "", ans)
                # Fallback: if not in objects, try to pick first viable candidate from aff_objs
                if ans and ans in self.all_objects:
                    new_affordance_obj = [ans]
                else:
                    # try to map by prefix (e.g., sink -> sink_1)
                    candidates = [o for o in aff_objs if ans and o.startswith(ans)]
                    if candidates:
                        new_affordance_obj = [candidates[0]]
                    elif aff_objs:
                        new_affordance_obj = [aff_objs[0]]
                argument_list[0] = new_affordance_obj
                print("argument list: ", argument_list)
            else:
                raise NotImplementedError(gen_type, "affordance not implemented")
            
            all_argument_pairs = [(self.all_objects[x], self.all_objects[y]) for x in argument_list[0] for y in argument_list[1]]

            

            if len(all_argument_pairs) == 0:
                return
            
            pred_name = event.predicates[0].pname

            new_facts = []

            # print("all argument pairs: ", all_argument_pairs)

            for arg in all_argument_pairs:
                # Build a candidate fact and skip if it's already known (obs/property/hyp)
                fact = ThorKBFact(pred_name, arg, sense_type='hyp_fact')
                fname = fact.name
                if fname in self.kb.obs_facts or fname in self.kb.hyp_facts or fname in self.kb.property_facts:
                    continue
                if fact not in self.failed_relationships:
                    new_facts.append(fact)

            # print("new facts: ", new_facts)
            # print(event.predicates)
            # pdb.set_trace()

            if gen_type in ["gt", "llm"]:
                if not new_facts:
                    # Nothing new to add; mark this event as satisfied to avoid looping on it again
                    event.is_sat = True
                    continue
                chosen_fact = random.choice(new_facts)
                self.kb.add_fact(chosen_fact)
                self.all_facts.add(chosen_fact)
                # Mark the event satisfied and stop after adding one fact to prevent spamming
                event.is_sat = True
                return
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

            # print("affordance candidates: ", affordance_candidates)
            # print("all affordance facts: ", self.all_affordance_facts)
            # print("failed affordance facts: ", self.failed_affordance_facts)

            

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
        # Prioritize: go to -> open -> any
        go_to_feasible = {a for a in feasible_explore_actions if a.action_type == 'go to'}
        if go_to_feasible:
            return random.choice(list(go_to_feasible))

        open_feasible = {a for a in feasible_explore_actions if a.action_type == 'open'}
        if open_feasible:
            return random.choice(list(open_feasible))

        if feasible_explore_actions:
            return random.choice(list(feasible_explore_actions))

        # If no feasible actions left, fall back to any 'go to' or 'open' from all actions
        go_to_all = {i for i in self.all_explore_actions if i.action_type == 'go to'}
        return random.choice(list(go_to_all))


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
           
        
    # def update_failed_action(self, failed_action):
    #     action_str = str(failed_action)
    #     env_action = failed_action
    #     action_type = env_action.action_type
    #     self.failed_env_actions.add(env_action)

    #     if action_type in ["heat", "clean", "cool", "light"]:
    #         aff_name = self.action2affordance_dict[action_type]
    #         terms = [env_action.recep, env_action.obj]
    #         failed_affordance = ThorKBFact(aff_name, terms, sense_type='hyp_fact')
    #         # pdb.set_trace()
    #         self.kb.remove_fact(failed_affordance)
    #         self.failed_affordance_facts.add(failed_affordance)

    def try_another_action_object(self, failed_action):
        # print(failed_action)
        if failed_action == None:
            return False, None, None, None, False
        if "put " in failed_action['action_str'] and self.task_type != "pick_two_obj_and_place":
            obj= failed_action['env_action'].obj
            recep_name = failed_action['env_action'].recep.thor_name

            # robust parse: type is all but last token; number is last token (if numeric)
            parts = recep_name.split(' ')
            recep_num = parts[1]
            recep_type = parts[0]

            
            
            # find other receptacles (objects) of the same type but with a different instance number
            same_type_other_receps = []
            for r in self.all_objects:
                r_parts = r.split('_')
                r_num = r_parts[1]
                r_type = r_parts[0]
                if r_type == recep_type and (r_num != recep_num):
                    same_type_other_receps.append(r)

            other_receps = [self.all_objects[i] for i in same_type_other_receps]

            success, feedback, failed_action, env_action, done = False, None, None, None, False

            for recep in other_receps:
                put_action = EnvInteractAction('put', obj, recep)
                success, feedback, failed_action, env_action, done = self.step(put_action)
                if success:
                    return success, feedback, failed_action, env_action, done
                    
            return success, feedback, failed_action, env_action, done

        elif "clean" in failed_action['action_str']:
            # Try cleaning with another sink/sinkbasin of the same type but different instance number
            obj = failed_action['env_action'].obj
            recep_name = failed_action['env_action'].recep.thor_name

            parts = recep_name.split(' ')
            if len(parts) >= 2:
                recep_type = parts[0]
                recep_num = parts[1]
            else:
                recep_type = parts[0]
                recep_num = None

            # Candidate washing receptacles by name prefix
            washing_prefixes = {"sink", "sinkbasin"}

            same_type_other_receps = []
            for name in self.all_objects:
                segs = name.split('_')
                if len(segs) >= 2:
                    n_type, n_num = segs[0], segs[1]
                    # if our failed type is a sink-like type, prefer same type else allow any sink variant
                    if recep_type.lower() in washing_prefixes:
                        if n_type.lower() == recep_type.lower() and (recep_num is None or n_num != recep_num):
                            same_type_other_receps.append(name)
                    else:
                        # fallback: any sink-like different instance
                        if n_type.lower() in washing_prefixes and (recep_num is None or n_num != recep_num):
                            same_type_other_receps.append(name)

            other_receps = [self.all_objects[i] for i in same_type_other_receps]

            # default unsuccessful tuple
            success, feedback, failed_action2, env_action2, done = False, None, None, None, False

            for recep in other_receps:
                clean_action = EnvInteractAction('clean', obj, recep)
                # print(clean_action.gen_action_str_list(self.current_location))
                
                success, feedback, failed_action2, env_action2, done = self.step(clean_action)
                if success:
                    return success, feedback, failed_action2, env_action2, done

            return success, feedback, failed_action2, env_action2, done


        else:
            return False, None, None, None, False
            
    def check_task_finished(self, success, feedback, failed_action, env_action, done):
        # Be robust if an action attempt returned env_action=None
        if env_action is None:
            return done
        if self.task_type == "pick_two_obj_and_place":
            return done
        else:
            if env_action.action_type == "put":
                if success:
                    done = True
        return done
        
    
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
        # Suppress noisy internal tracebacks emitted by AI2-THOR/ALFWorld expert on stderr
        with open(os.devnull, 'w') as _devnull:
            with redirect_stderr(_devnull):
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
                return False, fb, failed, env_action, done

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
                return False, fb, failed, env_action, done

        # success: update KB and internal state
        env_action.obj.location = env_action.recep.location
        self.holding = None
        self.current_location = env_action.recep.location

        hold_f = ThorKBFact("holds", [env_action.obj])
        in_f   = ThorKBFact("inreceptacle", [env_action.recep, env_action.obj])
        # try:
        #     self.kb.remove_fact(hold_f)
        # except:
        #     print("Failed to remove hold fact")
        #     for fact, _ in self.kb.obs_facts.items():
        #         if "can_contain" not in fact:
        #             if "is_a_" not in fact:
        #                 print(fact)
        #     pdb.set_trace()
        self.kb.add_fact(in_f)
        self.all_facts.add(in_f)


        # clear stale facts
        self.kb.remove_fact(in_f)
        obs_f = ThorKBFact(f"is_a_{env_action.obj.name.split('_')[0]}", [env_action.obj])

        if self.task_type == "pick_two_obj_and_place":
            self.kb.remove_fact(obs_f)

        # self.kb.remove_fact(obs_f)



        # mark failed exploration for this receptacle (re-open next time)
        from_env = EnvExploreAction('go to', env_action.recep)
        open_env = EnvExploreAction('open', env_action.recep)
        self.failed_explore_actions.update({from_env, open_env})

        return True, fb, failed or {}, env_action, done


    def _handle_interact(self, env_action, cmds):
        failed = None
        for cmd in cmds:
            fb, done = self._thor_step(cmd)
            # print(cmd, fb, done)
            # pdb.set_trace()
            if "nothing happens" in fb and "open" not in cmd and "go to" not in cmd:
                failed = self._build_failed(cmd, env_action, fb)
                print(f"{env_action} {cmds} {cmd} failed.")
                if not cmd.startswith("go to"):
                    self.current_location = env_action.recep.location
                self.failed_env_actions.add(env_action)
                # remove failed affordance
                aff_name = self.action2affordance_dict[env_action.action_type]
                terms = [env_action.recep, env_action.obj]
                failed_affordance = ThorKBFact(aff_name, terms, sense_type='hyp_fact')
                # pdb.set_trace()
                self.kb.remove_fact(failed_affordance)
                self.failed_affordance_facts.add(failed_affordance)
                return False, fb, failed, env_action, done

        # success: add state fact (heat/cool/clean/light)
        state = self.action2state_dict[env_action.action_type]
        self.current_location = env_action.recep.location
        # fact = ThorKBFact(state, [env_action.obj])
        # self.kb.add_fact(fact)
        # self.all_facts.add(fact)

        # if failed:
        #     self.update_failed_action(env_action)

        return True, fb, failed or {}, env_action, done

   

    def get_objects_list(self):
        # init_objects = self.goal_objects
        # plan_objects = []
        # best_plan_prob = 0
        # best_plan = None

        object_set = set()
        for plan in self.kb_policy.plans:
            for e in plan.events:
                if e.type == 'type':
                    name_str = e.predicates[0].pname
                    if len(e.predicates) == 1 or "lamp" in name_str:
                        # if name_str.startswith("is_"):
                        if name_str.startswith("is_") and "device" not in name_str:
                            name = "_".join(e.predicates[0].pname.split("_")[2:])
                            object_set.add(name)

        # new_obj_set = set()
        # init_obj_list = [o.split("_")[0] for o in self.all_objects]
        # for o in object_set:
        #     if o not in init_obj_list:
        #         new_obj_set.add(o)

        # return list(new_obj_set)


        return list(object_set)


        
        #     if plan.llmprob > best_plan_prob:
        #         best_plan = plan
        #         best_plan_prob = plan.llmprob
        # plan_objects = [e for e in best_plan.events if e.type == 'type']
        # pos_voi_objects = []
        # for p in self.kb_policy.plans:
        #     for e in p.events:
        #         if e.type == 'type':
        #             voi = self.calculate_voi(e)
        #             if voi > 0 and e not in pos_voi_objects:
        #                 pos_voi_objects.append(e)
        #                 # print(e)
        
        # all_objects = list(set(plan_objects + pos_voi_objects))

        

    def update_object_list(self, obj_list, gen_type="gt"):
        """
        Parse a text observation into (name, type, state, location) tuples,
        then update or create the corresponding KB objects and facts.
        """


        objs = self.text_obs_parser.extract_objects_from_list(obj_list, self.current_location)


        for name, typ, state, loc in objs:
            obj = self._update_object(name, 'object', 'outside', self.current_location, gen_type)


    def update_initial_observation_vlm(self, obs, gen_type="gt"):

        cleaned_text = obs.replace("-= Welcome to TextWorld, ALFRED! =-", "").strip()
        parts = cleaned_text.split("Your task is to:", 1)

        room_description = parts[0].strip()
        task_description = parts[1].strip()

        self.task_text = task_description


        initial_observation = room_description

        initial_objs = self.text_obs_parser.parse_text_observation(initial_observation, self.current_location)

        for name, typ, state, loc in initial_objs:
            obj = self._update_object(name, typ, state, loc, gen_type=gen_type)
            self.all_explore_actions.add(EnvExploreAction('go to', obj))
            self.all_go_to_actions.add(EnvExploreAction('go to', obj))
            if obj.name.split("_")[0] in self.openable_init_objects:
                # print(obj)
                # pdb.set_trace()
                self.all_explore_actions.add(EnvExploreAction('open', obj))
                self.all_open_actions.add(EnvExploreAction('open', obj))


                
