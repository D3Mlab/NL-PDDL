import pddl
from pddl_planner.planner.planner import RegressionPlanner

from nl_pddl_planner.pddl_planner.planner.nl_planner import NLFOLRegressionPlanner


from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from event import Subgoal

from pprint import pprint

from event import Subgoal, EventGroup

from llm import LLMTruthProbEstimator

import numpy as np

import pdb

@dataclass
class Plan:

    def __init__(
        self,
        root_subgoal: Subgoal,
        nodes: List[Any] = None,
        actions: List[Any] = None,
        subgoals: List[Subgoal] = None
    ):
        self.root_subgoal = root_subgoal
        self.nodes = nodes
        self.actions = actions
        self.subgoals = subgoals
        self.events = self.get_event_groups()
    

    
    def get_event_groups(self, ignore_constants=True) -> List[EventGroup]:
        """
        Return all EventGroup instances in this plan:
          - from the root subgoal
          - from each intermediate subgoal
        """

        groups: List[EventGroup] = []
        # root
        for grp in self.root_subgoal.event_groups:
            if ignore_constants and grp.event_type == "constant":
                continue
            groups.append(grp)
        
        # groups.extend(self.root_subgoal.event_groups)
        # other subgoals
        # for sg in self.subgoals:
        #     groups.extend(sg.event_groups)

        return groups

    def __str__(self):
        return f"Plan: {self.root_subgoal.formula}"


class Policy:
    """
    Generates all regression plans up to a given length for a PDDL domain/problem.
    """
    def __init__(
        self,
        domain: str,
        goal: str,
        max_plan_length: int = 5
    ):
        # parse once
        # self.domain = pddl.parse_domain(domain_path)
        # self.problem = pddl.parse_problem(problem_path)
        self.domain = domain
        self.goal = goal
        self.max_plan_length = max_plan_length

        self.estimator = LLMTruthProbEstimator()

        # self.plans = self.generate_plans()
        # self.events = self.extract_events()

        self.plans = None
        self.events = None

        self.known_true_events = []
        self.known_false_events = []


    def check_validity(self, plan: Plan) -> bool:
        """
        Check if the plan is valid.
        This is a placeholder for actual validation logic.
        """
        for p in plan.root_subgoal.formula.collect_preds():
            if p.name == "holds":
            # if p.name == "holds" or p.name in ["isHot", "isCold", "isClean"]:
                return False
        return True
    
    def extract_events(self, ingore_constants = True) -> List[EventGroup]:
        all_groups: List[EventGroup] = []
        for plan in self.plans:
            # groups = plan.get_event_groups(ingore_constants)
            groups = plan.events
            all_groups.extend(groups)

        # print(all_groups)
        # print("----------------")
        

        return all_groups

    def generate_plans(self, max_plan_length=None) -> List[Plan]:



        if max_plan_length is None:
            max_plan_length = self.max_plan_length

        all_plans: List[Plan] = []

        # pdb.set_trace()

        planner = NLFOLRegressionPlanner(self.domain, self.goal, max_plan_length)
        # pdb.set_trace() 
        regressed_plans = planner.regress_plan_alfworld()

        # for plan in regressed_plans:
        #     print("Subgoal: ")
        #     print(plan[0])
        #     reversed_plan = plan[1]
        #     reversed_plan.reverse()
        #     print("Action: ", reversed_plan)
        #     print("Substitution: ", plan[2])
        #     print("--------------------")

        # pdb.set_trace()
        
        for plan in regressed_plans:
            
            root_formula = plan[0]
            root_sg = Subgoal(root_formula)
            substitution = plan[2]
            actions = [p.substitute(substitution) for p in plan[1]]

            #skip empty action plans
            if actions == []:
                continue

            actions.reverse()

            alfworld_plan = Plan(root_sg, actions=actions)
            all_plans.append(alfworld_plan)

        self.plans = all_plans

        self.events = self.extract_events()
        # return all_plans



            


        # for length in range(1, max_plan_length + 1):

        #     # # planner = RegressionPlanner(self.domain, self.problem, length)

        #     # planner = NLFOLRegressionPlanner(self.domain, self.goal, length)

        #     # regressed_plans = planner.regress_plan()

        #     # for plan in regressed_plans:
        #     #     print("Subgoal: ")
        #     #     print(plan[0])
        #     #     reversed_plan = plan[1]
        #     #     reversed_plan.reverse()
        #     #     print("Action: ", reversed_plan)
        #     #     print("Substitution: ", plan[2])
        #     #     print("--------------------")
                

        #     # pdb.set_trace()


        #     raw_plans = planner.plan_tree()

        #     for raw in raw_plans:
                
        #         root_formula = raw["plan_subgoal"]
        #         root_sg = Subgoal(root_formula)

        #         seq = raw["plan_sequence"]

        #         nodes = list(seq["node_list"])
        #         actions = list(seq["action_list"])
        #         formulas = seq["subgoal_list"]

        #         # reverse the order of actions, nodes and subgoals
        #         actions.reverse()
        #         nodes.reverse()
        #         formulas.reverse()

        #         subgoals = [Subgoal(f) for f in formulas]

        #         plan = Plan(root_sg, nodes, actions, subgoals)

        #         if self.check_validity(plan):
        #             # print(root_formula)
        #             # print(actions)
        #             # for i in formulas:
        #             #     print(i)
        #             # print("---------------------")  
        #             all_plans.append(plan)

        # self.plans = all_plans


        # self.events = self.extract_events()
        # # return all_plans

    def score_events(self, n_responses: int = 1):
        scored_events: List[EventGroup] = []
        for grp in self.events:
            # see if we've already scored an equivalent group
            match = next((e for e in scored_events if e.is_equivalent(grp)), None)
            if match:
                # reuse the existing probability
                grp.llmprob = match.llmprob
            else:
                # no equivalent yet: call LLM n_responses times, average
                probs = [
                    self.estimator.get_probability(grp.description)
                    for _ in range(n_responses)
                ]
                grp.llmprob = sum(probs) / len(probs) if probs else 0.0
                scored_events.append(grp)
        return self.events
    
    def calculate_plan_probability(self, plan: Plan, known_true_events=[], known_false_events=[]) -> float:
        """
        Calculate the probability of a plan based on its events.
        """
        known_true_events = self.known_true_events + known_true_events
        known_false_events = self.known_false_events + known_false_events
        prob = 1.0
        # print('--------------------------')
        # print(f"Plan: {plan.root_subgoal.formula}")
        # print(plan.events)
        for grp in plan.events:
            match_true = next((e for e in known_true_events if e.is_equivalent(grp)), None)
            match_false = next((e for e in known_false_events if e.is_equivalent(grp)), None)
            if match_true:
                pass
            elif match_false:
                return 0
            else:
                prob *= grp.llmprob
            # print(grp.description, grp.llmprob, prob)
        return prob
    
    def voi_score(self, event):
        prob_event_true = event.llmprob
        prob_event_false = 1 - event.llmprob

        best_plan_prob_event_true = max([self.calculate_plan_probability(p, known_true_events=[event]) for p in self.plans])
        best_plan_prob_event_false = max([self.calculate_plan_probability(p, known_false_events=[event]) for p in self.plans])

        EVPI = prob_event_true * best_plan_prob_event_true + prob_event_false * best_plan_prob_event_false

        EMV = max([
            prob_event_true * self.calculate_plan_probability(p, known_true_events=[event]) +
            prob_event_false * self.calculate_plan_probability(p, known_false_events=[event])
            for p in self.plans
        ])

        VOI = EVPI - EMV
        return VOI


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    domain_path = "/home/user/research/active_explore/alfworld_domain.pddl"
    problem_path = "/home/user/research/active_explore/alfworld_problem.pddl"

    policy = Policy(domain_path, problem_path, max_plan_length=5)
    policy.generate_plans()
    plans = policy.plans
    policy.score_events(n_responses=1)

    print(f"Found {len(plans)} plans")
    for idx, plan in enumerate(plans, start=1):
        print(f"\nPlan #{idx}:")
        print(f"Root subgoal: {plan.root_subgoal.formula}")
        for grp in plan.root_subgoal.event_groups:
            print(f"{grp.event_type.upper():12} | {grp.objects} | {grp.predicates} |{grp.description}")



    for event in policy.events:
        if event.event_type == "type" or event.event_type == "relationship":
            print(f"\nEvent: {event.event_type}")
            # print(f"Objects: {event.objects}")
            print(f"Predicates: {event.predicates}")
            print(f"Description: {event.description}")
            print(f"LLM Probability: {event.llmprob}")
            print(f"VOI Score: {policy.voi_score(event)}")
    
    for plan in policy.plans:
        prob = policy.calculate_plan_probability(plan)
        print(f"Plan Subgoal: {plan.root_subgoal.formula}")
        print(f"Plan Probability: {prob}")
