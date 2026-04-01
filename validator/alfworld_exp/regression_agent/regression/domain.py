from collections import OrderedDict

from typing import Sequence

from pddl.logic.terms import Term

# from pddl.logic import Predicate, constants, variables
# from pddl.core import Domain, Problem
# from pddl.action import Action
# from pddl.formatter import domain_to_string, problem_to_string
# from pddl.requirements import Requirements

# from pddl.parser.domain import DomainParser
# from pddl.core import Domain

from pddl import parse_domain

from .action import RegressionAction

class RegressionDomain:

    def __init__(self, fpath):
        self.pddl_domain =  parse_domain(fpath)
        self.types = self.pddl_domain.types
        self.actions = self.gen_actions(self.pddl_domain)
        self.predicates = self.gen_predicates(self.pddl_domain)

    def parse_pddl_domain(self, fpath):
        self.pddl_domain = parse_domain(fpath)
        return self.pddl_domain
    
    def gen_actions(self, pddl_domain):
        actions = OrderedDict()
        for a in pddl_domain.actions:
            actions[a.name] = RegressionAction(a)
        return actions

    def gen_predicates(self, pddl_domain):
        predicates = OrderedDict()
        for p in pddl_domain.predicates:
            predicates[p.name] = p
        return predicates
    

    # @property
    # def predicates(self) -> Sequence[Term]:
    #     """Get the terms."""
    #     return self._predicates

    # @property
    # def actions(self) -> Sequence:
    #     """Get the terms."""
    #     return self._actions
    
    # @property
    # def types(self) -> Sequence:
    #     """Get the terms."""
    #     return self._types

    
    def run_test(self):
        print(self.predicates)
        # for i in self.predicates:
        #     print(i.name)

        

