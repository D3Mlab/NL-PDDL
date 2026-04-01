from typing import Sequence
from pddl.logic import Predicate, constants, variables

from collections import defaultdict
from itertools import product

from .action import RegressionAction

import pdb

class RegressionState:

    def __init__(self, regression_domain):
        self.regression_domain = regression_domain

        self.vars = set()
        self.constants = set()
        self.pos_set = set()
        self.neg_set = set()
        self.plan = []
        self.task_type = None

        self.largest_standarization_number = 0
    
    def parse_goal(self, goal, task_type):
        """
        This method will take a language instruction and parse it into a goal or action.
        """
        if type(goal) == dict:
            obj = goal['object_target']
            recep = goal['parent_target']
            toggle = goal['toggle_target']
            mrecep = goal['mrecep_target']
            require_slice = goal['object_sliced']
        elif type(goal) == str:
            raise NotImplementedError
        else:
            raise NotImplementedError
        
        if task_type == 'pick_and_place_simple':
            goal_obj = variables("goal_obj", types=["obj"])[0]
            goal_recep = variables("goal_recep", types=["obj"])[0]

            self.vars = set([goal_obj,goal_recep])

            inrecep_pred = self.regression_domain.predicates['inReceptacle']
            iscontain_pred = self.regression_domain.predicates['isContained']
            holds_any_pred = self.regression_domain.predicates['holdsAny']

            cancontain_pred = self.regression_domain.predicates['canContain']

            obj_type_pred = Predicate("is{}".format(obj), goal_obj)
            recep_type_pred = Predicate("is{}".format(recep), goal_recep)

            self.pos_set = set([
                                obj_type_pred(goal_obj),
                                recep_type_pred(goal_recep),
                                inrecep_pred(goal_recep, goal_obj),
                                # iscontain_pred(goal_obj),
                                # cancontain_pred(goal_recep, goal_obj)
                                ])
            self.neg_set = set([holds_any_pred()])

        elif task_type == 'pick_two_obj_and_place':
            goal_obj = variables("goal_obj", types=["obj"])[0]
            goal_obj_2 = variables("goal_obj_2", types=["obj"])[0]
            goal_recep = variables("goal_recep", types=["obj"])[0]
            goal_recep_2 = variables("goal_recep_2", types=["obj"])[0]

            self.vars = set([goal_obj, goal_obj_2, goal_recep, goal_recep_2])

            inrecep_pred = self.regression_domain.predicates['inReceptacle']
            iscontain_pred = self.regression_domain.predicates['isContained']
            holds_any_pred = self.regression_domain.predicates['holdsAny']
            cancontain_pred = self.regression_domain.predicates['canContain']
            holds_pred = self.regression_domain.predicates['holds']

            obj_type_pred = Predicate("is{}".format(obj), goal_obj)
            obj_type_pred_2 = Predicate("is{}".format(obj), goal_obj_2)
            recep_type_pred = Predicate("is{}".format(recep), goal_recep)
            recep_type_pred_2 = Predicate("is{}".format(recep), goal_recep_2)

            self.pos_set = set([
                                obj_type_pred(goal_obj),
                                # obj_type_pred(goal_obj_2),
                                recep_type_pred(goal_recep),
                                # recep_type_pred(goal_recep_2),
                                inrecep_pred(goal_recep, goal_obj),
                                # inrecep_pred(goal_recep, goal_obj_2),
                                iscontain_pred(goal_obj),
                                # iscontain_pred(goal_obj_2),
                            
                                # cancontain_pred(goal_recep, goal_obj)
                                # iscontain_pred(goal_obj),
                                # cancontain_pred(goal_recep, goal_obj)
                                ])
            self.neg_set = set([holds_any_pred(), ])

        elif task_type == 'pick_heat_then_place_in_recep':
            goal_obj = variables("goal_obj", types=["obj"])[0]
            goal_recep = variables("goal_recep", types=["obj"])[0]

            self.vars = set([goal_obj,goal_recep])


            hot_pred = self.regression_domain.predicates['isHot']
            inrecep_pred = self.regression_domain.predicates['inReceptacle']
            iscontain_pred = self.regression_domain.predicates['isContained']
            holds_any_pred = self.regression_domain.predicates['holdsAny']

            cancontain_pred = self.regression_domain.predicates['canContain']

            obj_type_pred = Predicate("is{}".format(obj), goal_obj)
            recep_type_pred = Predicate("is{}".format(recep), goal_recep)

            self.pos_set = set([hot_pred(goal_obj),
                                obj_type_pred(goal_obj),
                                recep_type_pred(goal_recep),
                                inrecep_pred(goal_recep, goal_obj),
                                # iscontain_pred(goal_obj),
                                # cancontain_pred(goal_recep, goal_obj)
                                ])
            self.neg_set = set([holds_any_pred()])

        elif task_type == 'pick_clean_then_place_in_recep':
            goal_obj = variables("goal_obj", types=["obj"])[0]
            goal_recep = variables("goal_recep", types=["obj"])[0]

            self.vars = set([goal_obj,goal_recep])


            clean_pred = self.regression_domain.predicates['isClean']
            inrecep_pred = self.regression_domain.predicates['inReceptacle']
            iscontain_pred = self.regression_domain.predicates['isContained']
            holds_any_pred = self.regression_domain.predicates['holdsAny']

            cancontain_pred = self.regression_domain.predicates['canContain']

            obj_type_pred = Predicate("is{}".format(obj), goal_obj)
            recep_type_pred = Predicate("is{}".format(recep), goal_recep)

            self.pos_set = set([clean_pred(goal_obj),
                                obj_type_pred(goal_obj),
                                recep_type_pred(goal_recep),
                                inrecep_pred(goal_recep, goal_obj),
                                # iscontain_pred(goal_obj),
                                # cancontain_pred(goal_recep, goal_obj)
                                ])
            self.neg_set = set([holds_any_pred()])

        elif task_type == 'pick_cool_then_place_in_recep':
            goal_obj = variables("goal_obj", types=["obj"])[0]
            goal_recep = variables("goal_recep", types=["obj"])[0]

            self.vars = set([goal_obj,goal_recep])


            cool_pred = self.regression_domain.predicates['isCool']
            inrecep_pred = self.regression_domain.predicates['inReceptacle']
            iscontain_pred = self.regression_domain.predicates['isContained']
            holds_any_pred = self.regression_domain.predicates['holdsAny']

            cancontain_pred = self.regression_domain.predicates['canContain']

            obj_type_pred = Predicate("is{}".format(obj), goal_obj)
            recep_type_pred = Predicate("is{}".format(recep), goal_recep)

            self.pos_set = set([cool_pred(goal_obj),
                                obj_type_pred(goal_obj),
                                recep_type_pred(goal_recep),
                                inrecep_pred(goal_recep, goal_obj),
                                # iscontain_pred(goal_obj),
                                # cancontain_pred(goal_recep, goal_obj)
                                ])
            self.neg_set = set([holds_any_pred()])
        
        elif task_type == "look_at_obj_in_light":
            goal_obj = variables("goal_obj", types=["obj"])[0]
            goal_recep = variables("goal_recep", types=["obj"])[0]

            self.vars = set([goal_obj, goal_recep])

            light_pred = self.regression_domain.predicates['isLight']
            holds_any_pred = self.regression_domain.predicates['holdsAny']
            holds_pred = self.regression_domain.predicates['holds']

            obj_type_pred = Predicate("is{}".format(obj), goal_obj)
            recep_type_pred = Predicate("is{}".format(toggle), goal_recep)

            iscontain_pred = self.regression_domain.predicates['isContained']

            self.pos_set = set([light_pred(goal_recep, goal_obj),
                                obj_type_pred(goal_obj),
                                recep_type_pred(goal_recep),
                                holds_pred(goal_obj),
                                holds_any_pred(),
                                # iscontain_pred(goal_obj),
                                # cancontain_pred(goal_recep, goal_obj)
                                ])
            self.neg_set = set([iscontain_pred(goal_obj)])
        
        elif task_type == "pick_clean_heat_place_in_recep":
            goal_obj = variables("goal_obj", types=["obj"])[0]
            goal_recep = variables("goal_recep", types=["obj"])[0]

            self.vars = set([goal_obj,goal_recep])


            hot_pred = self.regression_domain.predicates['isHot']
            clean_pred = self.regression_domain.predicates['isClean']
            inrecep_pred = self.regression_domain.predicates['inReceptacle']
            iscontain_pred = self.regression_domain.predicates['isContained']
            holds_any_pred = self.regression_domain.predicates['holdsAny']

            cancontain_pred = self.regression_domain.predicates['canContain']

            obj_type_pred = Predicate("is{}".format(obj), goal_obj)
            recep_type_pred = Predicate("is{}".format(recep), goal_recep)

            self.pos_set = set([hot_pred(goal_obj),
                                clean_pred(goal_obj),
                                obj_type_pred(goal_obj),
                                recep_type_pred(goal_recep),
                                inrecep_pred(goal_recep, goal_obj),
                                # iscontain_pred(goal_obj),
                                # cancontain_pred(goal_recep, goal_obj)
                                ])
            self.neg_set = set([holds_any_pred()])

        elif task_type == "pick_clean_cool_place_in_recep":
            goal_obj = variables("goal_obj", types=["obj"])[0]
            goal_recep = variables("goal_recep", types=["obj"])[0]

            self.vars = set([goal_obj,goal_recep])


            cool_pred = self.regression_domain.predicates['isCool']
            clean_pred = self.regression_domain.predicates['isClean']
            inrecep_pred = self.regression_domain.predicates['inReceptacle']
            iscontain_pred = self.regression_domain.predicates['isContained']
            holds_any_pred = self.regression_domain.predicates['holdsAny']

            cancontain_pred = self.regression_domain.predicates['canContain']

            obj_type_pred = Predicate("is{}".format(obj), goal_obj)
            recep_type_pred = Predicate("is{}".format(recep), goal_recep)

            self.pos_set = set([cool_pred(goal_obj),
                                clean_pred(goal_obj),
                                obj_type_pred(goal_obj),
                                recep_type_pred(goal_recep),
                                inrecep_pred(goal_recep, goal_obj),
                                # iscontain_pred(goal_obj),
                                # cancontain_pred(goal_recep, goal_obj)
                                ])
            self.neg_set = set([holds_any_pred()])



        

        else:
            
            raise NotImplementedError(task_type)

        self.task_type = task_type

        # print("goal set for :" + task_type)
        # print("goal is :" + str(self.pos_set))

    
    def update_var_set(self):
        vars = set()
        for p in self.pos_set:
            vars.update(p.terms)
        for p in self.neg_set:
            vars.update(p.terms)
        self.vars = vars
    
    def subst_var_set(self, var_set, sub_var_dict):
        subed_var_set = set()
        for v in var_set:
            if v in sub_var_dict:
                subed_var_set.add(sub_var_dict[v])
            else:
                subed_var_set.add(v)
        return subed_var_set
    
    def subst_var_list(self, var_list, sub_var_dict):
        subed_var_list = []
        for v in var_list:
            if v in sub_var_dict:
                subed_var_list.append(sub_var_dict[v])
            else:
                subed_var_list.append(v)
        return subed_var_list

    
    def subst_predicate_set(seelf, pred_set, sub_var_dict):
        new_pred_set = set()
        for p in pred_set:
            new_terms = []
            for t in p.terms:
                if t in sub_var_dict:
                    new_terms.append(sub_var_dict[t])
                else:
                    new_terms.append(t)
            new_pred = p(*new_terms)
            new_pred_set.add(new_pred)
        return new_pred_set


    def subst_action(self, action, sub_var_dict):
        
        new_action = RegressionAction(action_name=action.name)

        params = action.parameters
        # ordered_params = action.order_parameters
        pos_preds = action.pos_preds
        neg_preds = action.neg_preds
        add_effects = action.add_effects
        del_effects = action.del_effects

        new_action.parameters = self.subst_var_set(params, sub_var_dict)
        # new_action.orderd_parameters = self.subst_var_list(ordered_params, sub_var_dict)
        new_action.pos_preds = self.subst_predicate_set(pos_preds, sub_var_dict)
        new_action.neg_preds = self.subst_predicate_set(neg_preds, sub_var_dict)
        new_action.add_effects = self.subst_predicate_set(add_effects, sub_var_dict)
        new_action.del_effects = self.subst_predicate_set(del_effects, sub_var_dict)


        return new_action
    
    # def standardize_old(self, action=None, vars=None, digits=5):
        
    #     if vars == None:
    #         vars = self.vars

    #     params = action.parameters
    #     common_vars = params & vars

    #     subst_dict = {}
    #     for v in common_vars:
    #         var_name = v.name.split('_')[0]
    #         if len(v.name.split('_')) > 1:
    #             var_number = int(v.name.split('_')[1]) + 1
    #         else:
    #             var_number = 0
    #         new_var_name = '_'.join([var_name, str(var_number).zfill(digits)])
    #         new_var = variables(f"{new_var_name}", types=v.type_tags)[0]
    #         subst_dict[v] = new_var
    #     standardized_action = self.subst_action(action, subst_dict)


    #     return standardized_action
    
    def standardize(self, action=None, vars=None, digits=5):


        
        if vars == None:
            vars = self.vars

       

        params = action.parameters
        common_vars = params & vars
        subst_dict = {}

        # print(params, vars)

        for v in common_vars:
            var_name = v.name.split('_')[0]

            if len(v.name.split('_')) > 1:
                # var_number = int(v.name.split('_')[1]) + 1
                var_number = self.largest_standarization_number + 1
                self.largest_standarization_number += 1
            else:
                # var_number = 0
                var_number = self.largest_standarization_number + 1
                self.largest_standarization_number += 1

            # var_number =  int(v.name.split('_')[1]) + 1

            new_var_name = '_'.join([var_name, str(var_number).zfill(digits)])
            new_var = variables(f"{new_var_name}", types=v.type_tags)[0]
            subst_dict[v] = new_var
        standardized_action = self.subst_action(action, subst_dict)

        # print('Action', action)
        # print('Standard Action', standardized_action)

        return standardized_action
    

    def sort_by_pred_name(self,item):
        return item.name
    
    def find_unification_candidates(self, source_set, target_set):
        source_list = sorted(list(source_set), key=self.sort_by_pred_name)

        target_list = sorted(list(target_set), key=self.sort_by_pred_name)
        # target_list.sort()

        mapping_dict = defaultdict(list)

        source_common_list = []
        target_common_lists = []

        for s in source_list:
            for t in target_list:
                if s.name == t.name:
                    mapping_dict[s].append(t)
                    source_common_list.append(s)
            if len(mapping_dict[s]) > 0:
                target_common_lists.append(mapping_dict[s])

        unify_candidates = list(product(*target_common_lists))

        return tuple(source_common_list), unify_candidates
       
    def unify(self, source_preds, target_preds):
        subst_dict = {}
        for s, t in zip(source_preds, target_preds):
            for i, j in zip(s.terms, t.terms):
                if i not in subst_dict:
                    if i != j:
                        subst_dict[j] = i
        return subst_dict
    
    def is_relevant(self, action, pos_set=None, neg_set=None):
        if pos_set == None:
            pos_set = self.pos_set
        if neg_set == None:
            neg_set = self.neg_set
        return action.add_effects & pos_set and not (action.del_effects & pos_set) and not (action.add_effects & neg_set)
        

    def unify_action(self, action, pred_set=None):
        if pred_set == None:
            state_pos_set = self.pos_set
        action_pos_set = action.add_effects
        
        source_preds, unify_candidates = self.find_unification_candidates(state_pos_set, action_pos_set)

        if len(unify_candidates[0]) == 0:
            return []

        subst_list = []
        subst_actions = []

        for t in unify_candidates:
            subst_list.append(self.unify(source_preds, t))

        for s in subst_list:
            subbed_action = self.subst_action(action, s)
            
            if self.is_relevant(subbed_action):
                subst_actions.append(self.subst_action(action, s))

        return subst_actions
    
    def __str__(self):
        str_prep =  'variables: '+ str(self.vars) + '\n' \
                    + 'constants: '+ str(self.constants) + '\n' \
                    + 'pos: '+ str(self.pos_set) + '\n' \
                    + 'neg: '+ str(self.neg_set) + '\n'
        return str_prep
    
    def __eq__(self, other):
        """Override equal operator."""
        pos_names = set([p.name for p in self.pos_set])
        neg_names = set([p.name for p in self.neg_set])
        other_pos_names = set([p.name for p in other.pos_set])
        other_neg_names = set([p.name for p in other.neg_set])
        var_len = len(self.vars)
        other_var_len = len(other.vars)

        # print(self.vars, other.vars)
        # print(var_len == other_var_len)

        return pos_names == other_pos_names and neg_names == other_neg_names and var_len == other_var_len
        # return pos_names == other_pos_names and neg_names == other_neg_names and self.vars == other.vars
        
        # pos_unify = self.unify(self.pos_set, other.pos_set)


    # def __hash__(self):
    #     """Get the has of a Predicate."""
    #     return hash((self.vars, self.constants, self.pos_set, self.neg_set))
        



    



