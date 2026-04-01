from openai import OpenAI
from math import exp
import numpy as np

import re
import ast

import pdb

class LLMTruthProbEstimator:

    

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_tokens: int = 100,
        temperature: float = 0.0,
    ):
        self.client = OpenAI()
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        # reuse your prompt template
        self.CLASSIFICATION_PROMPT_TRUE = """You are a home assistant robot.
        You are in the beginning of a task.
        You are trying to find objects in houshold based on a description.
        Is it true that: 
        {headline}
        Only answer in [TRUE, FALSE]
        """


        self.CLASSIFICATION_PROMPT_False = """You are a home assistant robot.
        You are in the beginning of a task.
        You are trying to find objects in houshold based on a description.
        Is it true that: 
        {headline}
        Only answer in [TRUE, FALSE]
        """

    def _call_model(self, headline: str, type="True"):
        if type == "True":
            prompt = self.CLASSIFICATION_PROMPT_TRUE
        else:
            prompt = self.CLASSIFICATION_PROMPT_False
            headline = headline.replace("exists", "does not exist")
        messages = [
            {"role": "user", "content": prompt.format(headline=headline)}
        ]
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            logprobs=True,
            top_logprobs=10,
        )

    def get_probability(self, headline: str) -> float:

        if "any object" in headline:
            return 0.99 

        resp_true = self._call_model(headline, type="True")
        resp_false = self._call_model(headline, type="False")
        # extract the top‐logprobs for the first (and only) output token
        top_logprobs_true = resp_true.choices[0].logprobs.content[0].top_logprobs
        top_logprobs_false = resp_false.choices[0].logprobs.content[0].top_logprobs

        


        # top_logprobs is a list of objects with .token and .logprob
        # convert to {token: linear_prob}
        lin_probs_true = {lp.token: exp(lp.logprob) for lp in top_logprobs_true}
        lin_probs_false = {lp.token: exp(lp.logprob) for lp in top_logprobs_false}



        total_true = sum(lin_probs_true.values())
        total_false = sum(lin_probs_false.values())
        
        avg_prop_true = lin_probs_true.get("TRUE", 0.0) / total_true
        avg_prop_false = lin_probs_false.get("FALSE", 0.0) / total_false

        # combine_prop = (avg_prop_true + (1 - avg_prop_false)) / 2

        # return combine_prop

        return avg_prop_true

        # if total == 0:
        #     return 0.0
        # return normalized probability of the exact token "True"
        # return lin_probs.get("YES", 0.0) / total


class LLMExplorer:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_tokens: int = 100,
        temperature: float = 0.0,
    ):
        self.client = OpenAI()
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        self.EXPLORE_PROMPT = """You are a home assistant robot.
        Your goal is to find objects in the environment.
        Assuming all the objects are in visible places.
        You want to find: {objects}
        The list of potential places to explore are: {places}
        PLease choose the best place to explore.
        Answer in this format: **place 1**
        """

        # Optional usage recorder callback; if set, will receive raw API responses
        self._usage_recorder = None

    def _call_model(self, objects, places: str):
        messages = [
            {"role": "user", "content": self.EXPLORE_PROMPT.format(objects=objects, places=places) }
        ]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        # Record token usage if a recorder is registered
        if getattr(self, "_usage_recorder", None):
            try:
                self._usage_recorder(resp)
            except Exception:
                pass
        return resp

    def set_usage_recorder(self, fn):
        """Register a callback that accepts the raw API response for usage accounting."""
        self._usage_recorder = fn

    def choose_exploration(self, subgoal_targets, explore_candidates) -> str:
        resp = self._call_model(subgoal_targets, explore_candidates)

        exp_str = resp.choices[0].message.content

        exp_str = re.findall(r'\*\*(.*?)\*\*', exp_str)[0]

        return exp_str
