import logging
from pddl_planner.logic.nl_parser import NLParser
from pddl_planner.pddl_core.nl_domain import NLDomain
from pddl_planner.logic.formula import Predicate
from typing import List

logger = logging.getLogger("pddl_planner.domain")

class NLInstance():
    def __init__(self, nl_problem: List[tuple], nl_init: List[tuple]|None, domain: NLDomain):
        self._domain = domain
        self._parser = NLParser()
        self._predicates = None
        self._goal = None
        self._objects = None
        self._init = None
        self._parse_problem(nl_problem)
        self._parse_predicates(nl_problem)
        if nl_init is not None:
            self._parse_init(nl_init)
        else:
            self._init = None
        
    def _parse_problem(self, nl_problem: List[tuple]):
        # self._init = self._parser.parse_init(list(nl_problem.init))
        self._goal = self._parser.parse_goal(nl_problem)
        #extract the terms and type from the goal
        type_tags = {}
        for block in nl_problem:
            for term in block[1].keys():
                type_tags[term] = block[1][term]
        self._objects = [self._parser.parse_term(obj, type_tags=type_tags) for obj in type_tags.keys()]
    
    def _parse_predicates(self, nl_problem: List[tuple]):
        """
        Parse and construct the list of predicates in the goal and store them in self._predicates.

        Args:
            nl_problem (List[tuple]): The PDDL problem.
        """
        predicates: List[Predicate] = []
        for nl_predicate in nl_problem:
            predicate = self._parser.parse_predicate(nl_predicate)
            predicates.append(predicate)
        self._predicates = predicates

    def _parse_init(self, nl_problem: List[tuple]):
        """
        Parse and construct the list of predicates in the init and store them in self._init.
        """
        init = self._parser.parse_goal(nl_problem)
        logger.debug("Parsed initial state: %s", init)
        self._init = init

    @property
    def domain(self):
        return self._domain
    
    @property
    def init(self):
        return self._init
    
    @property
    def goal(self):
        return self._goal
    
    @property
    def objects(self):
        return self._objects
    
    def __str__(self):
        return f"Instance: {self._domain.name}, Init: {self._init}, Goal: {self._goal}, Objects: {self._objects}"
    
    def __repr__(self):
        return f"Instance: {self._domain.name}, Init: {self._init}, Goal: {self._goal}, Objects: {self._objects}"
        