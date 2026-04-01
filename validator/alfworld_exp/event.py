from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple, Optional
import re

from pddl_planner.logic.formula import ConjunctiveFormula, Predicate, Term

from itertools import permutations

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def split_name(name: str) -> str:
    """Strip 'is'/'can' prefix and split CamelCase or snake_case into words."""
    for prefix in ("is", "can"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    parts = re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+", name)
    words = [p.lower().replace("obj ", "object ") for p in parts if p]
    return " ".join(words)


# ------------------------------------------------------------------
# Data class for grouped events
# ------------------------------------------------------------------
@dataclass
class EventGroup:
    """
    Represents predicates grouped by type:
      - 'type':     unary predicates per object
      - 'relationship': binary+ predicates
      - 'constant': zero-arity predicates
    """
    event_type: str
    objects: List[Term]
    predicates: List[Predicate]
    description: str = ""
    llmprob: float = None
    generic_vars = ["x", "y", "z"]

    def to_nl_simple(self) -> str:
        """
        Convert this group's predicates into a simple NL sentence.
        Does not consider other groups.
        """
        if self.event_type == "type":


            phrases = []
            for obj in sorted(self.objects, key=str):
                props = [
                    f"{str(obj)} is a " + split_name(p.name)
                    for p in self.predicates
                    if p.terms and p.terms[0] == obj and p.name.lower() != "isobject"
                ]
                if props:
                    phrases.append(f"{', and '.join(props)}")
            if len(phrases) == 0:
                return ""
            else:
                return "; ".join(phrases)

        # relationship or constant
        if self.event_type == "relationship":
            parts = []
            for p in self.predicates:
                obj, subj = p.terms[:2]
                action = split_name(p.name)
                parts.append(f"object {obj} can {action} object {subj}")
            return "; ".join(parts)

        if self.event_type == "constant":
            # zero-arity predicates
            return "; ".join(f"{p.name} holds" for p in self.predicates)

    def is_equivalent(self, other: Any) -> bool:
        """
        Two EventGroups are equivalent if:
          - same event_type
          - same number of objects & predicates
          - if types are terms, check if they have the same set of predicates.
        """
        if not isinstance(other, EventGroup):
            return False

        if self.event_type != other.event_type:
            return False

        if len(self.objects) != len(other.objects):
            return False

        if len(self.predicates) != len(other.predicates):
            return False

        if self.event_type in ("type", "constant"):
            self_pred_name = {p.name for p in self.predicates}
            other_pred_name = {p.name for p in other.predicates}
            return self_pred_name == other_pred_name

        if self.event_type == "relationship":
            # Build a normalization helper
            self_norm = self.anonymize_vars(
                self.description,
                [t for t in self.predicates[0].terms]
            )
            other_norm = self.anonymize_vars(
                other.description,
                [t for t in other.predicates[0].terms]
            )
            return self_norm == other_norm

        raise NotImplementedError(
            f"EventGroup type '{self.event_type}' not implemented for equivalence check."
        )

    def anonymize_vars(self, desc: str, objs: List[Term]) -> str:
        norm_desc = desc
        # Map each object name to a generic variable
        for idx, term in enumerate(objs[:2]):
            orig = str(term)
            var = self.generic_vars[idx]
            # Replace whole-word occurrences
            norm_desc = norm_desc.replace(orig, var)
        return norm_desc


# ------------------------------------------------------------------
# Subgoal class with enriched NL
# ------------------------------------------------------------------
@dataclass
class Subgoal:
    formula: ConjunctiveFormula
    event_groups: List[EventGroup] = field(init=False)
    generic_vars = ["X", "Y", "Z"]
    pos_prefix = "There exists"
    negative_prefix = ""  

    def __post_init__(self):
        self.event_groups = self._build_event_groups(self.formula)
        self._generate_nl_descriptions()

    def _build_event_groups(
        self,
        phi: ConjunctiveFormula,
        *,
        descriptor_order: Optional[Sequence[str]] = None,
    ) -> List[EventGroup]:
        zero, unary, multi = [], [], []
        for p in phi.collect_preds():
            if p.arity == 0:
                zero.append(p)
            elif p.arity == 1:
                unary.append(p)
            else:
                multi.append(p)


        type_map: Dict[Term, List[Predicate]] = {}
        for p in unary:
            type_map.setdefault(p.terms[0], []).append(p)
        for t in phi.collect_terms():
            type_map.setdefault(t, [])

        groups: List[EventGroup] = []
        for term in sorted(type_map, key=str):
            preds = sorted(type_map[term], key=str)
            # check if preds is empty
            if preds!=[]:
                groups.append(EventGroup("type", [term], preds))

        if multi:
            for pred in multi:
                objs = {o for o in pred.terms}
                groups.append(
                    EventGroup("relationship", list(objs), sorted([pred], key=str))
                )

        if zero:
            groups.append(EventGroup("constant", [], sorted(zero, key=str)))

        return groups

    def _generate_nl_descriptions(self, anonymize_vars: bool = True) -> List[str]:
        """
        Populate `description` on each EventGroup and return
        a list of all descriptions.
        """
        # First handle 'type' groups
        type_desc: Dict[Term, str] = {}
        # type_desc_anonymize: Dict[Term, str] = {}
        for grp in self.event_groups:
            if grp.event_type == "type":
                prefix = f"{self.pos_prefix} object {str(grp.objects[0])} such that: "
                description = grp.to_nl_simple()
                if description != "":
                    if anonymize_vars and grp.predicates:
                        anony_predix = self.anonymize_vars(
                            prefix, [t for t in grp.predicates[0].terms]
                        )
                        anony_description = self.anonymize_vars(
                            description, [t for t in grp.predicates[0].terms]
                        )
                        grp.description = anony_predix + anony_description
                    else:
                        grp.description = prefix + description + '.'
                    if grp.objects:
                        type_desc[grp.objects[0]] = description + '.'
                else:
                    grp.description = "The envrioment has any object in it."
                    type_desc[grp.objects[0]] = ""

                    # type_desc_anonymize[grp.objects[0]] = anony_description

        # Next handle relationships, using type descriptions
        for grp in self.event_groups:
            if grp.event_type == "relationship":
                parts = []
                for p in grp.predicates:
                    obj, subj = p.terms[:2]
                    action = split_name(p.name)
                    subj_desc = type_desc.get(subj, '')
                    obj_desc = type_desc.get(obj, '')
                    prefix = f"{self.pos_prefix} objects {obj} and {subj} such that: "

                    description = (
                        prefix +
                        f"{obj} can {action} {subj}. Given {obj_desc} {subj_desc}"
                    )
                    if anonymize_vars:
                        anony_description = self.anonymize_vars(description, [t for t in p.terms])
                        parts.append(anony_description)
                    else:
                        parts.append(description)
                grp.description = " | ".join(parts)

            elif grp.event_type == "constant":
                # zero-arity predicates
                grp.description = grp.to_nl_simple()

        # Return all descriptions
        return [grp.description for grp in self.event_groups]

    def anonymize_vars(self, desc: str, objs: List[Term]) -> str:
        norm_desc = desc
        # Map each object name to a generic variable
        for idx, term in enumerate(objs[:2]):
            orig = str(term)
            var = self.generic_vars[idx]
            norm_desc = norm_desc.replace(orig, var)
        return norm_desc


# ------------------------------------------------------------------
# Demo usage
# ------------------------------------------------------------------
if __name__ == "__main__":
    from pddl_planner.logic.formula import Constant, Variable

    V1, V2 = Variable("V1"), Variable("V2")
    c1, c2 = Constant("p1"), Constant("r1")

    formula = ConjunctiveFormula(
        Predicate("handEmpty", False),
        Predicate("canHeat", False, V1, c1),
        Predicate("isHot", False, c1),
        Predicate("isHeatingObj", False, V1),
        Predicate("isPickupableObj", False, V2),
        Predicate("isPotato", False, c1),
        Predicate("isTable", False, c2),
    )

    sg = Subgoal(formula)
    for grp in sg.event_groups:
        print(f"{grp.event_type.upper():12} | {grp.objects} | {grp.predicates}| {grp.description}")
