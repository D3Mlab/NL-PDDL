from .action import RegressionAction
from .state import RegressionState

import pdb

class RegressionSolver:

    def __init__(self, domain=None, goal_state=None, initial_state=None):

        self.domain = domain
        self.goal_state = goal_state
        self.initial_state = initial_state
        # self.horizion = horizion

        self.visited_states = []
        self.fully_regressed_states = []
        self.fixed_horizon_regressed_states = []


    
    def regress_action(self, state, action):
        new_state = RegressionState(self.domain)
        new_state.pos_set = state.pos_set - action.add_effects | action.pos_preds
        new_state.neg_set = state.neg_set - action.del_effects | action.neg_preds

        # action.pos_subgoal = new_state.pos_set
        # action.neg_subgoal = new_state.neg_set

        new_state.update_var_set()

        action.pos_subgoal = new_state.pos_set
        action.neg_subgoal = new_state.neg_set
        action.subgoal_vars = new_state.vars

        new_state.plan = state.plan.copy()
        new_state.plan = [action] + new_state.plan

        return new_state
    
    def check_state_contridiction(self, state):
        if len(state.pos_set & state.neg_set) > 0:
            return True
    

    def solve(self, horizon=10, max_visited_states=1000):


        stack = [self.goal_state] 


        while len(stack) > 0:

            current_state = stack.pop()

            if current_state in self.visited_states:
                continue
        
            self.visited_states.append(current_state)

            if len(current_state.plan) >= horizon:
                if current_state not in self.fixed_horizon_regressed_states:
                    self.fixed_horizon_regressed_states.append(current_state)
                continue
        
            # for i in range(horizon):
            all_regressed_actions = []

            for action_name, action in self.domain.actions.items():
                standard_action = current_state.standardize(action)
                unified_actions = current_state.unify_action(standard_action)

                # ## not if unified action is empty but all action are empty!!
                # if len(unified_actions) == 0 and current_state not in self.fully_regressed_states:
                #         self.fully_regressed_states.append(current_state)

                for action in unified_actions:
                    new_state = self.regress_action(current_state, action)

                    if not self.check_state_contridiction(new_state) and new_state not in self.visited_states:
                        stack.append(new_state)
                        all_regressed_actions.append(new_state)

            if len(all_regressed_actions) == 0:
                # print(all_regressed_actions)
                self.fully_regressed_states.append(current_state)

            
        # for i in self.fully_regressed_states:
        #     print('------------------')
        #     print(i.pos_set)
        #     print([a.name for a in i.plan])
            

        return self.fully_regressed_states, self.fixed_horizon_regressed_states


        for i in self.fully_regressed_states:
            print('----------------')
            plan_names = [a.name for a in i.plan]
            plan_names.reverse()
            pos_set = [str(i) for i in i.pos_set]
            
            print('Plan: ', plan_names)
            print('State: ', pos_set)
            # print(i.neg_set)
        


                        
            
                        
        # for i in self.visited_states:
        #     print(i)



            
  
    







