#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert regressed plans from NLFOLRegressionPlanner into a JSON-friendly list
of dicts. Each item has:
  - subgoal: a dict whose keys are disjunct identifiers (disjoint_1, ...), and
             values are lists of clause strings in that disjunct (conjunction)
  - action:  the corresponding action sequence (list of strings)

Usage (example):
  python plan_to_subgoals.py \
    --domain blockworld_domain.json \
    --problem instance-10.json \
    --max-depth 3 \
    --out out.json

This script expects to be run from nl_pddl_planner/tests_user/ so that relative
paths for domain/problem JSON resolve.
"""
from __future__ import annotations
import argparse
import json
import os
from typing import Any, Dict, List

from pddl_planner.planner.nl_planner import NLFOLRegressionPlanner
from pddl_planner.logic.formula import DisjunctiveFormula, ConjunctiveFormula
from pddl_planner.logic.formula import Equality
import re


class PlanSubgoalConverter:
    """
    Reusable converter to transform regressed plans into
    [{ 'subgoal': {disjoint_i: [...]}, 'action': ["name(params)", ...] }, ...]
    """

    @staticmethod
    def _action_to_str(action: Any) -> str:
        try:
            params = []
            for p in getattr(action, 'parameters', []):
                try:
                    # Use str(term) so Variables appear like '?V32' and Constants like 'a'
                    params.append(str(p))
                except Exception:
                    params.append(str(p))
            return f"{getattr(action,'name','action')}({', '.join(params)})"
        except Exception:
            return str(action)

    @staticmethod
    def _formula_to_disjuncts_dict(formula: Any) -> Dict[str, List[str]]:
        """Turn a Formula into {disjoint_i: [clause_str, ...]}.
        Accepts DisjunctiveFormula, ConjunctiveFormula, or single clause.
        """
        disjuncts: Dict[str, List[str]] = {}
        if isinstance(formula, DisjunctiveFormula):
            for idx, conj in enumerate(formula.clauses, start=1):
                if isinstance(conj, ConjunctiveFormula):
                    disjuncts[f"disjoint_{idx}"] = [str(c) for c in conj.clauses]
                else:
                    disjuncts[f"disjoint_{idx}"] = [str(conj)]
        elif isinstance(formula, ConjunctiveFormula):
            disjuncts["disjoint_1"] = [str(c) for c in formula.clauses]
        else:
            disjuncts["disjoint_1"] = [str(formula)]
        return disjuncts

    @staticmethod
    def _extract_var_const_map_from_conj(conj: Any) -> Dict[str, str]:
        """From a ConjunctiveFormula, collect equalities of the form ?Vxx == const.
        Returns mapping {'?V32': 'a', ...}.
        """
        varmap: Dict[str, str] = {}
        if not isinstance(conj, ConjunctiveFormula):
            return varmap
        for clause in getattr(conj, 'clauses', []):
            if isinstance(clause, Equality) and not getattr(clause, 'is_neq', False):
                # Access proper API: Equality.term1 / term2
                t1 = getattr(clause, 'term1', None)
                t2 = getattr(clause, 'term2', None)
                try:
                    from pddl_planner.logic.formula import Variable, Constant
                except Exception:
                    Variable = type('Variable', (), {})  # fallback
                    Constant = type('Constant', (), {})

                # If ?V == const
                if t1 is not None and t2 is not None:
                    if isinstance(t1, Variable) and isinstance(t2, Constant):
                        varmap[str(t1)] = str(t2)
                    elif isinstance(t2, Variable) and isinstance(t1, Constant):
                        varmap[str(t2)] = str(t1)
        return varmap

    @staticmethod
    def _build_planner_sub_map(substitution: Any) -> Dict[str, str]:
        """Turn planner substitution (a dict-like mapping Terms) into {var_name: term_name}."""
        mapping: Dict[str, str] = {}
        try:
            for k, v in substitution.items():
                # Use str() so variables are '?V..' and constants are names
                kn = str(k)
                vn = str(v)
                mapping[kn] = vn
        except Exception:
            pass
        return mapping

    @staticmethod
    def _apply_name_mapping_to_str(s: str, mapping: Dict[str, str]) -> str:
        if not mapping:
            return s
        # Sort keys by length, longest first, to avoid partial replacements (e.g. ?V1 in ?V10)
        sorted_keys = sorted(mapping.keys(), key=len, reverse=True)
        for var in sorted_keys:
            # Basic string replacement
            s = s.replace(var, mapping[var])
        return s

    def convert_regressed_plans(self, regressed_plans: List[List[Any]]) -> List[Dict[str, Any]]:
        """
        Convert tuples (subgoal_formula, plan_actions, substitution) into a
        list of dicts with disjoint subgoals and action strings.
        """
        converted: List[Dict[str, Any]] = []
        for item in regressed_plans:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            subgoal = item[0]
            actions = item[1]
            substitution = item[2] if len(item) > 2 else {}

            # 1) Build raw action strings (no substitutions), and the planner-substituted ones
            sub_map = self._build_planner_sub_map(substitution)
            action_list_raw = [self._action_to_str(a) for a in actions]
            action_list = [self._apply_name_mapping_to_str(s, sub_map) for s in action_list_raw]

            # 2) For each disjunct, substitute ?V* variables to constants per-conjunct equalities
            conjuncts: List[Any] = []
            if isinstance(subgoal, DisjunctiveFormula):
                conjuncts = subgoal.clauses
            elif isinstance(subgoal, ConjunctiveFormula):
                conjuncts = [subgoal]
            else:  # is a single predicate/clause
                conjuncts = [ConjunctiveFormula(subgoal)]

            # Split output: one item per disjunct, each paired with the same action list
            for conj in conjuncts:
                # Build raw subgoal only for this specific disjunct
                if isinstance(conj, ConjunctiveFormula):
                    raw_clauses = [str(c) for c in conj.clauses]
                else:
                    raw_clauses = [str(conj)]
                varmap = self._extract_var_const_map_from_conj(conj)
                # Normalize to a ConjunctiveFormula
                if not isinstance(conj, ConjunctiveFormula):
                    conj = ConjunctiveFormula(conj)
                clauses: List[str] = []
                for c in conj.clauses:
                    # Skip positive equality clauses in output; they are applied via varmap
                    if isinstance(c, Equality) and not getattr(c, 'is_neq', False):
                        continue
                    s = str(c)
                    # Apply planner sub first, then per-disjunct varmap (ground vars like ?V32 to constants a,b,...)
                    s = self._apply_name_mapping_to_str(s, sub_map)
                    s = self._apply_name_mapping_to_str(s, varmap)
                    clauses.append(s)
                # Ground the actions per disjunct as well: apply per-disjunct varmap on top of planner substitution
                # Reverse the action list as requested
                action_list_rev = list(reversed(action_list))
                action_list_disj = [self._apply_name_mapping_to_str(s, varmap) for s in action_list_rev]
                # Also provide raw, unsubstituted actions (reversed to match ordering)
                action_list_raw_rev = list(reversed(action_list_raw))
                converted.append({
                    'subgoal_raw': {"disjoint": raw_clauses},  # only the matching disjunct
                    'subgoal': {"disjoint": clauses},
                    'action': action_list_disj,
                    'action_raw': action_list_raw_rev,
                    'substitution': sub_map,
                })
        return converted

    def convert_from_domain_and_goal(self, domain: List[Dict[str, Any]], goal: List[Any], max_depth: int = 3) -> List[Dict[str, Any]]:
        """
        Run the planner for the given domain and goal, then convert results.
        """
        planner = NLFOLRegressionPlanner(domain.copy(), goal.copy(), max_depth=max_depth)
        regressed_plans = planner.regress_plan()
        return self.convert_regressed_plans(regressed_plans)

    # ---------- PDDL predicate conversion utilities ----------
    @staticmethod
    def goal_entries_to_pddl(goal_items: List[Any]) -> List[str]:
        """Convert overall NL goal entries of the form ["sentence", {...}] to
        PDDL predicate strings like (on a b), (ontable a), etc.

        Supported patterns:
          - "a is on top of d"      -> (on a d)
          - "a is on the table"     -> (ontable a)
          - "the hand is empty"     -> (handempty)
          - "i am holding a"        -> (holding a)
          - "a is clear"            -> (clear a)
        """
        preds: List[str] = []
        for entry in goal_items:
            if not isinstance(entry, (list, tuple)) or not entry:
                continue
            sent = str(entry[0]).strip()
            s = re.sub(r"\s+", " ", sent)
            # a is on top of d -> (on a d)
            m = re.match(r"^([A-Za-z0-9_]+) is on top of ([A-Za-z0-9_]+)$", s)
            if m:
                preds.append(f"(on {m.group(1)} {m.group(2)})")
                continue
            # a is on the table -> (ontable a)
            m = re.match(r"^([A-Za-z0-9_]+) is on the table$", s)
            if m:
                preds.append(f"(ontable {m.group(1)})")
                continue
            # the hand is empty -> (handempty)
            if s.lower() == "the hand is empty":
                preds.append("(handempty)")
                continue
            # i am holding a -> (holding a)
            m = re.match(r"^i am holding ([A-Za-z0-9_]+)$", s, flags=re.IGNORECASE)
            if m:
                preds.append(f"(holding {m.group(1)})")
                continue
            # a is clear -> (clear a)
            m = re.match(r"^([A-Za-z0-9_]+) is clear$", s)
            if m:
                preds.append(f"(clear {m.group(1)})")
                continue
        return preds
    @staticmethod
    def _clauses_to_pddl_predicates(clauses: List[str]) -> List[str]:
        """Convert NL-style clause strings into canonical PDDL predicate strings.
        Supported mappings:
          - is clear(x)            -> (clear x)
          - is on the table(x)     -> (ontable x)
          - the hand is empty()    -> (handempty)
          - i am holding(x)        -> (holding x)
          - is on top of(x, y)     -> (on x y)
        Unrecognized clauses (including inequalities) are ignored.
        """
        preds: List[str] = []
        for s in clauses:
            t = s.strip()
            # Normalize whitespace inside
            t = re.sub(r"\s+", " ", t)
            # is clear(x)
            m = re.match(r"^is clear\(([^)]+)\)$", t)
            if m:
                arg = m.group(1).strip()
                preds.append(f"(clear {arg})")
                continue
            # is on the table(x)
            m = re.match(r"^is on the table\(([^)]+)\)$", t)
            if m:
                arg = m.group(1).strip()
                preds.append(f"(ontable {arg})")
                continue
            # the hand is empty()
            m = re.match(r"^the hand is empty\(\)$", t)
            if m:
                preds.append("(handempty)")
                continue
            # i am holding(x)
            m = re.match(r"^i am holding\(([^)]+)\)$", t)
            if m:
                arg = m.group(1).strip()
                preds.append(f"(holding {arg})")
                continue
            # is on top of(x, y)
            m = re.match(r"^is on top of\(([^,]+),\s*([^)]+)\)$", t)
            if m:
                a = m.group(1).strip()
                b = m.group(2).strip()
                preds.append(f"(on {a} {b})")
                continue
            # ignore others (e.g., inequalities)
        return preds

    def converted_items_to_pddl(self, converted: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Given the output from convert_regressed_plans, return a list where
        each item includes a PDDL-style predicate list for the subgoal.
        Output item shape:
          {
            'subgoal': { 'disjoint_1': [...] },
            'action': [...],
            'subgoal_predicates': ['(clear a)', '(on a b)', ...],
            'actions_pddl': ['(stack a d)', '(pick-up a)', ...]
          }
        """
        out: List[Dict[str, Any]] = []
        for item in converted:
            sub = item.get('subgoal', {})
            # We generate predicates from the single 'disjoint' list we emit per item
            clauses = []
            if isinstance(sub, dict):
                # prefer new key, but support legacy 'disjoint_1' for backward compat
                clauses = sub.get('disjoint') or sub.get('disjoint_1', []) or []
            pddl_preds = self._clauses_to_pddl_predicates(clauses)
            actions = item.get('action', []) or []
            pddl_actions = self._actions_to_pddl(actions)
            # Also convert raw actions if present
            raw_actions = item.get('action_raw', []) or []
            pddl_actions_raw = self._actions_to_pddl(raw_actions)
            new_item = dict(item)
            new_item['subgoal_predicates'] = pddl_preds
            new_item['actions_pddl'] = pddl_actions
            new_item['actions_pddl_raw'] = pddl_actions_raw
            out.append(new_item)
        return out

    @staticmethod
    def _actions_to_pddl(actions: List[str]) -> List[str]:
        """Convert grounded action strings into PDDL operator calls using the
        provided domain operator names and parameter order.

        Mapping rules (case-insensitive on action name):
          - pick up ... -> pick-up(x)
          - put down ... -> put-down(x)
          - stack ?b1 on top of ?b2(x, y) -> stack(x, y)
          - unstack ?b1 from ?b2(x, y) -> unstack(x, y)
        """
        result: List[str] = []
        for s in actions:
            m = re.match(r"^(.*)\((.*)\)$", s.strip())
            if not m:
                result.append(s)
                continue
            name = m.group(1).strip().lower()
            args_str = m.group(2).strip()
            args = [a.strip() for a in args_str.split(',') if a.strip()]

            if 'pick up' in name or 'pick-up' in name:
                if args:
                    result.append(f"pick-up({args[0]})")
                else:
                    result.append("pick-up()")
            elif 'put down' in name or 'put-down' in name:
                if args:
                    result.append(f"put-down({args[0]})")
                else:
                    result.append("put-down()")
            elif name.startswith('stack'):
                if len(args) >= 2:
                    result.append(f"stack({args[0]}, {args[1]})")
                elif len(args) == 1:
                    result.append(f"stack({args[0]})")
                else:
                    result.append("stack()")
            elif name.startswith('unstack'):
                if len(args) >= 2:
                    result.append(f"unstack({args[0]}, {args[1]})")
                elif len(args) == 1:
                    result.append(f"unstack({args[0]})")
                else:
                    result.append("unstack()")
            else:
                # Fallback to original if cannot map
                result.append(s)
        return result


def main():
    ap = argparse.ArgumentParser(description='Convert regressed plan to disjoint subgoals JSON')
    ap.add_argument('--domain', default='./blockworld_domain.json', help='Domain JSON path')
    ap.add_argument('--problem', default='./instance-10.json', help='Problem JSON path (goal/initial_state)')
    ap.add_argument('--max-depth', type=int, default=3)
    ap.add_argument('--out', default='./out.json/plan_subgoals.json', help='Output JSON path')
    args = ap.parse_args()

    # Load domain and problem (goal/initial state format like tests_user/instance-10.json)
    with open(args.domain, 'r') as f:
        domain = json.load(f)
    with open(args.problem, 'r') as f:
        prob = json.load(f)

    goal = prob['goal']
    converter = PlanSubgoalConverter()
    converted = converter.convert_from_domain_and_goal(domain, goal, max_depth=args.max_depth)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(converted, f, indent=2)
    print(f"Wrote converted subgoals -> {args.out}")


if __name__ == '__main__':
    main()
