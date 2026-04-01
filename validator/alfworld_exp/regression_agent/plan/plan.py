from regression_agent.utility.helpers import pvar2kbvar
from regression_agent.knowledgebase.kb import ThorKBVariable, ThorKBPredicate

import pdb



class ThorKBAction:

    def __init__(self, regress_action, kb):
        self.kb = kb

        self.ingore_list = []

        self.regress_action = regress_action
        self.name = regress_action.name

        self.parameters = self.convert_vars(self.regress_action.parameters)
        self.vars = self.convert_vars(self.regress_action.subgoal_vars)

        self.pos_preds = self.convert_preds(self.regress_action.pos_preds)
        self.neg_preds = self.convert_preds(self.regress_action.neg_preds)
        
        self.add_effects = self.convert_preds(self.regress_action.add_effects)
        self.del_effects = self.convert_preds(self.regress_action.del_effects)


        self.pos_subgoal = self.convert_preds(self.regress_action.pos_subgoal)
        self.neg_subgoal = self.convert_preds(self.regress_action.neg_subgoal)
        

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

    def __init__(self, regression_state, kb, task_type=None):
        self.kb = kb
        self.regression_state = regression_state
        self.regression_plan = regression_state.plan.copy()

        self.task_type = task_type

        self.subgoal_pos = {}
        self.subgoal_neg = {}
        self.subgoal_vars = {}
        self.subgoal_query = []
        self.actions = []

        self.convert_reg_subgoal()
        self.convert_reg_plan()
        # self.gen_subgoal_query()

        self.current_action = self.actions[0]
        self.current_subgoal_pos = self.current_action.pos_subgoal
        self.current_subgoal_neg = self.current_action.neg_subgoal
        self.current_plan_len = len(self.actions)
        
        self.current_query_model = []

    def convert_reg_subgoal(self):
        self.subgoal_vars = self.convert_vars(self.regression_state.vars)
        self.subgoal_pos = self.convert_preds(self.regression_state.pos_set, self.subgoal_vars)
        self.subgoal_neg = self.convert_preds(self.regression_state.neg_set, self.subgoal_vars)

        # pdb.set_trace()
    
    def reset_plan(self):
        self.subgoal_pos = {}
        self.subgoal_neg = {}
        self.subgoal_vars = {}
        self.subgoal_query = []
        self.actions = []

        self.convert_reg_subgoal()
        self.convert_reg_plan()

        self.convert_reg_subgoal()
        self.convert_reg_plan()

        self.current_action = self.actions[0]
        self.current_subgoal_pos = self.current_action.pos_subgoal
        self.current_subgoal_neg = self.current_action.neg_subgoal
        self.current_plan_len = len(self.actions)
        
        self.current_query_model = []


    
    def convert_reg_plan(self):

        
        self.actions = [ThorKBAction(action, self.kb) for action in self.regression_plan]

        if self.task_type == 'pick_two_obj_and_place':
            self.actions = self.actions * 2

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
                self.current_subgoal_neg = None
                self.current_plan_len = len(self.actions)
                return None
            else:           
                self.current_action = self.actions[0]
                self.current_subgoal_pos = self.current_action.pos_subgoal
                self.current_subgoal_neg = self.current_action.neg_subgoal
                self.current_plan_len = len(self.actions)

    def __str__(self):
        plan_list = [i.name for i in self.actions]
        return str(plan_list)
        


class ThorKBPolicy:

    def __init__(self, state_list, kb, task_type=None):
        self.kb = kb

        self.task_type = task_type

        self.plans = [ThorKBPlan(s, self.kb, self.task_type) for s in state_list]
        self.plans.sort(key=lambda plan: plan.current_plan_len, reverse=True)

        # need to change this later
        self.exp_plan = self.choose_exp_plan(self.plans)
        self.exp_plan_subgoal_pos = self.exp_plan.current_subgoal_pos
        self.exp_plan_subgoal_neg = self.exp_plan.current_subgoal_neg


    
    def choose_exp_plan(self, plans):

        exp_plan = min(plans, key=lambda plan: len(plan.current_action.pos_preds))

        return exp_plan

    def __str__(self):
        plan_list = [str(i) for i in self.plans]
        return str(plan_list)


