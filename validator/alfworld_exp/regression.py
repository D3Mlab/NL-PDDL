from kb import ThorKB, ThorKBFact, ThorKBPredicate, ThorKBVariable, ThorKBObject

from typing import List, Optional
from plan import Policy
from llm import LLMTruthProbEstimator
from event import EventGroup, Subgoal

from plan import Policy


class ALFWorldRegressionAgentNL:
    def __init__(
        self,
        domain_NL: str,
        goal_NL: str,
        max_plan_length: int = 5,
        llm_model: str = "gpt-4o-mini",
    ):

        self.estimator = LLMTruthProbEstimator(model=llm_model)
        
        # storage
        self.policy = Policy(
            domain=domain_NL,
            goal=goal_NL,
            max_plan_length=max_plan_length
        )

        self.plans = self.policy.plans
        self.scored_events: List[EventGroup] = []

    
    def generate_plans(self, max_plan_length=5) -> List:
        """Run regression search to get all valid plans."""

        self.policy.generate_plans(max_plan_length=max_plan_length)
        self.plans = self.policy.plans


        # return self.plans

    def extract_and_score_events(
        self,
        ignore_constants: bool = True,
        n_responses: int = 1
    ) -> List[EventGroup]:
        """
        1) Extract every EventGroup from every plan
        2) For each unique equivalence‐class of EventGroup, call the LLM
           `n_responses` times and store the average in `event.llmprob`.
        """
        # pull out & score in one pass
        scored: List[EventGroup] = []
        for plan in self.plans:
            for grp in plan.get_event_groups(ignore_constants):
                # check if we’ve already scored an equivalent group
                match = next((e for e in scored if e.is_equivalent(grp)), None)
                if match:
                    grp.llmprob = match.llmprob
                else:
                    # call LLM multiple times
                    probs = [
                        self.estimator.get_probability(grp.description)
                        for _ in range(n_responses)
                    ]
                    grp.llmprob = sum(probs) / len(probs) if probs else 0.0
                    scored.append(grp)
        self.scored_events = scored
        return self.scored_events

    def pick_best_event(self) -> Optional[EventGroup]:
        """Return the highest‐scoring event (or None if no events)."""
        if not self.events:
            return None
        return max(self.events, key=lambda e: e.llmprob)

    def plan_step(self) -> Optional[str]:
        """
        1) Generate & score if not already done
        2) Pick best next event
        3) Return its natural‐language `description`
        """
        if not self.plans:
            self.generate_plans()
        if not self.events:
            self.extract_and_score_events()

        best = self.pick_best_event()
        if best:
            return best.description
        return None