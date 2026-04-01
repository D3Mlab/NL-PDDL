from pddl_planner.logic.parser import Parser

class Instance():
    def __init__(self, pddl_problem, domain):
        self._domain = domain
        self._parser = Parser()
        self._init = None
        self._goal = None
        self._objects = None
        self._parse_problem(pddl_problem)
        
    def _parse_problem(self, pddl_problem):
        self._objects = [self._parser.parse_term(obj) for obj in pddl_problem.objects]
        self._init = self._parser.parse_init(list(pddl_problem.init))
        self._goal = self._parser.parse_goal(pddl_problem.goal)
    
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
        