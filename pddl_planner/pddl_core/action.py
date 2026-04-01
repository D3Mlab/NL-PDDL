from typing import List, Set
from pddl_planner.logic.formula import ConjunctiveFormula, Substitution
from pddl_planner.logic.operation import Operations
from pddl_planner.logic.formula import Formula, Term

class Action:
    """Action class to represent PDDL actions
    params: name: str, preconditions: Formula, effects: Formula"""
    def __init__(self, name: str, parameters: List[Term], preconditions: Formula, effects: Formula) -> None:
        """Initialize an Action object
        
        Args:
            name (str): The name of the action.
            parameters (List[Term]): The parameters of the action.
            preconditions (Formula): The preconditions of the action.
            effects (Formula): The effects of the action.
        """
        self._name = name
        self._preconditions = preconditions
        self._effects = effects
        self._parameters = parameters

    def substitute(self, substitution: Substitution) -> "Action":
        """Substitute the variables in the action.
        The variables in the preconditions and effects are substituted with the values in the substitution.
        
        Args:
            substitution (Substitution): The substitution to apply.
        
        Returns:
            Action: The action with the substitution applied.
        """
        substituted_parameters = [substitution.get(parameter, parameter) for parameter in self._parameters]
        return Action(self._name, substituted_parameters, self._preconditions.substitute(substitution), self._effects.substitute(substitution))
    
    def standardize(self, operations: Operations) -> "Action":
        """Standardize the variables in the action and return a new action.
        The variables in the preconditions and effe cts are standardized (new variables are created).
        
        Args:
            operations (Operations): The operations to use for standardizing.
        
        Returns:
            Action: The standardized action.
        """
        sub = Substitution()
        standardized_parameters = []
        for parameter in self._parameters:
            sub[parameter] = operations.get_new_var()
            standardized_parameters.append(sub[parameter])

        standardized_preconditions = self._preconditions.substitute(sub)
        standardized_effects = self._effects.substitute(sub)
            
        return Action(self._name, standardized_parameters, standardized_preconditions, standardized_effects)

    @property
    def name(self) -> str:
        """
        Get the name of the action.

        Returns:
            str: The name of the action.
        """
        return self._name
    
    @property
    def parameters(self) -> List[Term]:
        """
        Get the parameters of the action.
        
        Returns:
            List[Term]: The parameters of the action.
        """
        return self._parameters

    @property
    def preconditions(self) -> Formula:
        """
        Get the preconditions of the action.
        
        Returns:
            Formula: The preconditions of the action.
        """
        return self._preconditions
    
    @property
    def effects(self) -> Formula:
        """
        Get the effects of the action.

        Returns:
            Formula: The effects of the action.
        """
        return self._effects
    
    def __str__(self) -> str:
        """
        Get the string representation of the action.
        ActionName(Parameter1, Parameter2, ..., ParameterN)
        
        Returns:
            str: The string representation of the action.
        """
        return f"{type(self).__name__}({repr(self.name)}({', '.join(repr(param) for param in self.parameters)}))"

    def __repr__(self) -> str:
        """
        Get the string representation of the action.
        ActionName(Parameter1, Parameter2, ..., ParameterN)
        
        Returns:
            str: The string representation of the action.
        """
        return f"{type(self).__name__}({repr(self.name)}({repr(self.parameters)}))"
