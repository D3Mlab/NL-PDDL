from typing import Sequence

from pddl.logic.base import And, Atomic, Not
from pddl.logic.effects import AndEffect
from pddl.logic.terms import Term

# from pddl.action import Action



class RegressionAction:

    def __init__(self, pddl_action=None, action_name='GenericAction'):

        if pddl_action != None:
            self.pddl_action = pddl_action

            self._name = pddl_action.name
            self.parameters = set(pddl_action.parameters)
            # self.order_parameters = list(pddl_action.parameters)
            

            self.pos_preds, self.neg_preds = self.gen_atomic_sets(self.pddl_action.precondition)
            self.add_effects, self.del_effects = self.gen_atomic_sets(self.pddl_action.effect)
        else:
            self.pddl_action = None
            self._name = action_name
            self.parameters = None
            # self.order_parameters = None
            self.pos_preds = None 
            self.neg_preds = None
            self.add_effects = None 
            self.del_effects = None

        self.pos_subgoal = None
        self.neg_subgoal = None
        self.subgoal_vars = None

            
    # function to get all atoms of a logic formula
    def gen_atomic_sets(self, formula):

        pos_set = set()
        neg_set = set()

        if isinstance(formula, Atomic):
            pos_set.add(formula)
        elif isinstance(formula, Not):
            neg_set.add(formula.argument)
        elif isinstance(formula, And) or isinstance(formula, AndEffect):
            operands = formula.operands
            for op in operands:
                pos_set_children, neg_set_children = self.gen_atomic_sets(op)
                pos_set.update(pos_set_children)
                neg_set.update(neg_set_children)
        else:
            print(formula)
            raise ValueError('{} not supported'.format(type(formula)))

        return pos_set, neg_set
    
    def standardize_self(self, var_dict):
        pass

    
    @property
    def name(self) -> str:
        """Get the name."""
        return self._name

    # @property
    # def parameters(self) -> Sequence[Term]:
    #     """Get the terms."""
    #     return self._parameters
    
    def __str__(self):
        str_prep = self._name + '\n' \
                    + 'parameters: '+ str(self.parameters) + '\n' \
                    + 'pos_pred: '+ str(self.pos_preds) + '\n' \
                    + 'neg_pred: '+ str(self.neg_preds) + '\n' \
                    + 'add_effects: '+ str(self.add_effects) + '\n' \
                    + 'del_effects: '+ str(self.del_effects)
        return str_prep
    
    
    
