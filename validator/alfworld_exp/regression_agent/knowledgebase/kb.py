from pyDatalog import pyDatalog

import pdb



class ThorKBObject: 
    
    def __init__(self, name, location, obj_type, obj_state=None, mask=None, sense_type='observation'):
        # super(ThorKBObject, self).__init__()
        pyDatalog.create_terms(name) 
        self.name = name
        self.thor_name = " ".join(name.split("_"))
        self.location = location           
        self.thor_location = " ".join(location.split("_"))
        self.mask = mask
        self.sense_type = sense_type
        self.obj_type = obj_type 
        self.obj_state = obj_state
                
    
    def __repr__(self):
        return self.name
    
    def __eq__(self, other):
        return self.name == other.name and self.location == other.location and self.obj_type == other.obj_type
    
    def __hash__(self):
        return hash(self.name)
    

class ThorKBVariable:
    def __init__(self, name):
        # super(ThorKBVariable, self).__init__()
        pyDatalog.create_terms(name) 
        self.name = name
    
    def __repr__(self):
        return self.name
    

class ThorKBPredicate:
    def __init__(self, pname, vars):
        pyDatalog.create_terms(pname)
        self.pname = pname
        self.vars = vars
        self.name = f"{self.pname}({', '.join([v.name for v in vars])})"

        self.fallback = self.generate_fallback(self.pname, vars)
        pyDatalog.load(self.fallback)

    
    def generate_fallback(self,  predicate_name, vars):

        if len(vars) == 0:
            return ''

        terms = ', '.join([f'{str(v)}' for v in vars])

        default_condition = ' & '.join([f"({term} == 'None')" for term in terms.split(', ')])
        fallback_rule = f"""
                        {predicate_name}({terms}) <= {default_condition} & ~{predicate_name}({terms})
                        """

        return fallback_rule

    def __repr__(self):
        return self.name
    
    
class ThorKBFact:
    def __init__(self, pname, objs, sense_type='observation'):
        pyDatalog.create_terms(pname)
        self.pname = pname
        self.vars = objs
        self.name = f"{self.pname}({', '.join([o.name for o in objs])})"
        self.sense_type = sense_type

    def __repr__(self):
        return self.name
    


# Define a KnowledgeBase class to simulate isolated environments
class ThorKB:
    def __init__(self):
        pyDatalog.clear()
        self.obs_facts = {}
        self.hyp_facts = {}
        self.terms = set()
        self.predicates = {}
        self.objects = {}
        self.vars = {}
    


    def add_predicate(self, pred_obj):
        self.predicates[pred_obj.name] = pred_obj
        self.terms.add(pred_obj.pname)
    
    def add_var(self, var):
        self.vars[var.name] = var
        self.terms.add(var.name)

    def add_object(self, obj):
        self.objects[obj.name] = obj
        self.terms.add(obj.name)
    
    def add_fact(self, fact):
        """Add a fact to the knowledge base"""

        pyDatalog.load("+ " + fact.name)

        if fact.sense_type == "observation":
            self.obs_facts[fact.name] = fact
        else:
            self.hyp_facts[fact.name] = fact
    
    def remove_fact(self, fact):
        pyDatalog.load("- " + fact.name)
        if fact.sense_type == "observation":
            self.obs_facts.pop(fact.name)
        else:
            self.hyp_facts.pop(fact.name)

    
    def query_pred(self, pred):
        """Run a query on the current knowledge base"""

        if type(pred) == str:
            answer = pyDatalog.ask(pred)
        else:
            answer = pyDatalog.ask(pred.name)

        return answer
    
    def clearn_kb(self):
        pyDatalog.clear()
        self.obs_facts = {}
        self.hyp_facts = {}
        self.terms = {}
        self.predicates = {}
        self.objects = {}



def pvar2kbvar(plan_var):
    if plan_var[0] == '?':
        kb_var = plan_var[1:].upper()
    else:
        kb_var = plan_var.upper()
    return kb_var

class ThorKBAction:

    def __init__(self, regress_action, regress_subgoal, kb):
        self.kb = kb

        self.ingore_list = []

        self.regress_action = regress_action
        self.regress_subgoal = regress_subgoal
        self.name = regress_action.name

        # vars = self.regression_plan.root_subgoal.formula.collect_terms()
        # preds = self.regression_plan.root_subgoal.formula.collect_preds()

        self.parameters = self.convert_vars(self.regress_action.parameters)
        self.vars = {**self.parameters, **self.convert_vars(self.regress_subgoal.collect_terms())}

        self.pos_preds = self.convert_preds(self.regress_action.preconditions.collect_preds())
        # self.neg_preds = self.convert_preds(self.regress_action.neg_preds)

        effects = self.regress_action.effects.collect_preds()
        pos_effects = [e for e in effects if not e.is_neg]
        neg_effects = [e for e in effects if e.is_neg]


        self.add_effects = self.convert_preds(pos_effects)
        self.del_effects = self.convert_preds(neg_effects)


        self.pos_subgoal = self.convert_preds(regress_subgoal.collect_preds())
        # self.neg_subgoal = self.convert_preds(self.regress_action.neg_subgoal)
        

    def convert_vars(self, regress_vars):
        kb_params = {}
        for v in regress_vars:
            str_name = pvar2kbvar(str(v))
            kb_params[str_name] = ThorKBVariable(str_name)
            self.kb.add_var(kb_params[str_name])
        return kb_params
    
    def convert_preds(self, regress_preds):
        kb_preds = {}
        for p in regress_preds:
            if p.name.lower() not in self.ingore_list:
                pname = p.name
                vars = tuple([self.vars[pvar2kbvar(str(i))] for i in p.terms])
                predicate = ThorKBPredicate(pname.lower(), vars)
                self.kb.add_predicate(predicate)
                kb_preds[predicate.name] = predicate
        return kb_preds
        

class ThorKBPlan:

    def __init__(self, regression_plan, kb, task_type=None):
        self.kb = kb
        # self.regression_state = regression_state
        # self.regression_plan = regression_state.plan.copy()
        self.regression_plan = regression_plan

        self.task_type = task_type

        self.subgoal_pos = {}
        # self.subgoal_neg = {}
        self.subgoal_vars = {}
        self.subgoal_query = []
        self.actions = []

        self.convert_reg_subgoal()
        self.convert_reg_plan()
        # self.gen_subgoal_query()

        self.current_action = self.actions[0]
        self.current_subgoal_pos = self.current_action.pos_subgoal
        # self.current_subgoal_neg = self.current_action.neg_subgoal
        self.current_plan_len = len(self.actions)
        
        self.current_query_model = []

    def convert_reg_subgoal(self):
        vars = self.regression_plan.root_subgoal.formula.collect_terms()
        preds = self.regression_plan.root_subgoal.formula.collect_preds()

        self.subgoal_vars = self.convert_vars(vars)
        self.subgoal_pos = self.convert_preds(preds, self.subgoal_vars)

    
    def reset_plan(self):
        self.subgoal_pos = {}
        # self.subgoal_neg = {}
        self.subgoal_vars = {}
        self.subgoal_query = []
        self.actions = []

        self.convert_reg_subgoal()
        self.convert_reg_plan()

        self.convert_reg_subgoal()
        self.convert_reg_plan()

        self.current_action = self.actions[0]
        self.current_subgoal_pos = self.current_action.pos_subgoal
        # self.current_subgoal_neg = self.current_action.neg_subgoal
        self.current_plan_len = len(self.actions)
        
        self.current_query_model = []


    
    def convert_reg_plan(self):

        regression_actions = [action for action in self.regression_plan.actions]
        regression_subgoals  = [subgoal.formula for subgoal in self.regression_plan.subgoals]

        self.actions = [ThorKBAction(i[0], i[1], self.kb) for i in zip(regression_actions, regression_subgoals)]


    def convert_vars(self, regress_vars):
        kb_params = {}
        for v in regress_vars:
            str_name = pvar2kbvar(str(v))
            kb_params[str_name] = ThorKBVariable(str_name)
            self.kb.add_var(kb_params[str_name])
        return kb_params
    
    def convert_preds(self, regress_preds, all_vars):
        kb_preds = {}
        for p in regress_preds:
            # if p.name.lower() not in self.ingore_list:
            pname = p.name.lower()
            vars = tuple([all_vars[pvar2kbvar(str(i))] for i in p.terms])            
            predicate = ThorKBPredicate(pname, vars)
            self.kb.add_predicate(predicate)
            kb_preds[predicate.name] = predicate
            # kb_preds[]
        return kb_preds
        
                
    def take_action(self):
        if len(self.actions) == 0:
            return None
        else:
            action = self.actions.pop(0)
            if len(self.actions) == 0:
                self.current_action = None
                self.current_subgoal_pos = None
                # self.current_subgoal_neg = None
                self.current_plan_len = len(self.actions)
                return None
            else:           
                self.current_action = self.actions[0]
                self.current_subgoal_pos = self.current_action.pos_subgoal
                # self.current_subgoal_neg = self.current_action.neg_subgoal
                self.current_plan_len = len(self.actions)

    def __str__(self):
        plan_list = [i.name for i in self.actions]
        return str(plan_list)
        


class ThorKBPolicy:

    def __init__(self, plans, kb, task_type=None):
        self.kb = kb
        self.task_type = task_type
        self.plans = [ThorKBPlan(s, self.kb, self.task_type) for s in plans]

    def __str__(self):
        plan_list = [str(i) for i in self.plans]
        return str(plan_list)

