#!/usr/bin/env python3
"""
Block world initial state checker using PyDatalog.

- Reads example_out_pddl_regression.json (default path adjustable via CLI)
- Adds initial_state predicates as PyDatalog facts
- For each plan's subgoal_predicates, checks which are satisfied in the initial state
- Prints a summary and optionally writes a JSON report

Usage:
  python tests_user/block_world_test/initial_state_checker.py \
      --input tests_user/example_out_pddl_regression.json \
      --output tests_user/block_world_test/initial_state_eval.json

PyDatalog docs: https://sites.google.com/site/pydatalog/ (if needed)
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from pyDatalog import pyDatalog
except ImportError as e:
    raise SystemExit(
        "PyDatalog is required. Install with `pip install pyDatalog` and re-run."
    )


PredicateArg = Union[str, "Variable"]


@dataclass
class Variable:
    name: str

    def __repr__(self) -> str:
        return self.name


@dataclass
class Atom:
    name: str
    args: List[PredicateArg] = field(default_factory=list)

    def is_zero_arity(self) -> bool:
        return len(self.args) == 0


class BlockWorldInitialStateChecker:
    def __init__(self, input_path: str) -> None:
        self.input_path = input_path
        self.data: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}
        # Track predicates for which we have asserted extensional facts
        self.defined_preds: set[str] = set()

        # Prepare common predicate symbols used in Block World
        # Extendable if other symbols appear in the JSON
        pyDatalog.clear()
        # Terms creation is optional when using assert_fact/ask with string queries,
        # but we can keep some common ones defined for completeness.
        pyDatalog.create_terms(
            'on, ontable, clear, holding, handempty, X, Y, Z, U, V'
        )

    # ----------------------------- Parsing ----------------------------- #
    _token_pattern = re.compile(r"\(|\)|,|\s+")

    def parse_atom(self, s: str) -> Atom:
        """Parse a predicate like "(on a b)" or "(holding a)" or "(handempty)".
        Variables appear as tokens that start with '?', e.g., ?V53.
        """
        s = s.strip()
        if not (s.startswith('(') and s.endswith(')')):
            raise ValueError(f"Invalid atom syntax: {s}")
        inner = s[1:-1].strip()
        # split by spaces, but keep tokens in order
        tokens = [t for t in re.split(r"\s+|,", inner) if t]
        if not tokens:
            raise ValueError(f"Empty atom: {s}")
        name = tokens[0]
        raw_args = tokens[1:]
        args: List[PredicateArg] = []
        for a in raw_args:
            if a.startswith('?'):
                # convert ?V53 -> Variable('V53')
                args.append(Variable(a[1:]))
            else:
                # constants keep as strings
                args.append(a)
        return Atom(name=name, args=args)

    # --------------------------- PyDatalog I/O -------------------------- #
    def _ensure_terms(self, atoms: List[Atom]) -> None:
        """Ensure terms exist; with assert_fact/ask this is not strictly necessary,
        but creating them keeps consistency if mixed APIs are used later."""
        preds = {a.name for a in atoms}
        vars_ = {arg.name for a in atoms for arg in a.args if isinstance(arg, Variable)}
        term_str = ', '.join(sorted(preds | vars_))
        if term_str:
            pyDatalog.create_terms(term_str)

    def _add_fact(self, atom: Atom) -> None:
        """Assert a fact into PyDatalog for the given atom using assert_fact."""
        self._ensure_terms([atom])
        if atom.is_zero_arity():
            # e.g., handempty
            pyDatalog.assert_fact(atom.name)
            self.defined_preds.add(atom.name)
        else:
            # Only constants should appear in facts; variables in facts are unusual.
            args = []
            for a in atom.args:
                if isinstance(a, Variable):
                    # Skip variable in fact or treat as symbol name
                    args.append(a.name)
                else:
                    args.append(a)
            pyDatalog.assert_fact(atom.name, *args)
            self.defined_preds.add(atom.name)

    def _query_atom(self, atom: Atom) -> Tuple[bool, Optional[List[Tuple[str, ...]]]]:
        """Query whether an atom is entailed by current facts using ask().
        Returns (is_true, substitutions) where substitutions is a list of tuples
        of variable bindings (as strings) if the atom contains variables.
        """
        self._ensure_terms([atom])
        # If we have no facts for this predicate and no rules are defined, avoid
        # pyDatalog.ask error by treating it as false directly.
        if atom.name not in self.defined_preds:
            return False, None
        # Build a Datalog query string, variables start with uppercase; our Variable
        # objects already hold names like V53. Constants are lowercase like 'a','b'.
        if atom.is_zero_arity():
            q = f"{atom.name}()"
            ans = pyDatalog.ask(q)
            return ans is not None, None
        else:
            def fmt(a: PredicateArg) -> str:
                if isinstance(a, Variable):
                    return a.name
                else:
                    # quote constants so parser treats them as atoms/strings
                    return repr(a)
            call = f"{atom.name}({', '.join(fmt(a) for a in atom.args)})"
            ans = pyDatalog.ask(call)
            if ans is None:
                return False, None
            # ans.answers is a list of tuples of constants (possibly empty tuples)
            rows = [tuple(map(str, row)) for row in ans.answers]
            # If there were no variables, answers will be [()] if true
            vars_in_atom = [a for a in atom.args if isinstance(a, Variable)]
            if not vars_in_atom:
                return True, None
            else:
                return len(rows) > 0, rows

    def _query_conjunction(self, atoms: List[Atom]) -> Tuple[bool, List[str], Optional[List[Tuple[str, ...]]]]:
        """Query a conjunction of atoms. Returns (is_true, var_names, bindings).
        var_names is the sorted list of variable symbols seen across the atoms.
        If there are no variables, bindings is None.
        """
        if not atoms:
            return True, [], None
        # Ensure terms exist and predicate facts are defined
        self._ensure_terms(atoms)
        # If any predicate has no facts defined, treat as false
        for a in atoms:
            if a.name not in self.defined_preds:
                return False, [], None
        # Collect variables
        vars_set = {arg.name for a in atoms for arg in a.args if isinstance(arg, Variable)}
        var_names = sorted(vars_set)
        # Build conjunction string like: on('a','d') & clear('d') & ...
        def fmt_arg(a: PredicateArg) -> str:
            if isinstance(a, Variable):
                return a.name
            return repr(a)
        conj_parts = []
        for a in atoms:
            if a.is_zero_arity():
                conj_parts.append(f"{a.name}()")
            else:
                conj_parts.append(f"{a.name}({', '.join(fmt_arg(x) for x in a.args)})")
        query = ' & '.join(conj_parts)
        ans = pyDatalog.ask(query)
        if ans is None:
            return False, var_names, None
        if not var_names:
            # No variables: any answer means conjunction holds
            return True, var_names, None
        # With variables, ans.answers is a list of tuples. The order of variables is
        # implementation-defined; try to use ans.variables if present; otherwise use var_names.
        try:
            # pyDatalog may expose variable order in ans.variables
            if hasattr(ans, 'variables') and ans.variables:
                # Reorder tuples to match our var_names if possible
                # Build index mapping from ans.variables to our var_names
                order = [ans.variables.index(v) for v in var_names if v in ans.variables]
                rows = [tuple(map(str, [row[i] for i in order])) for row in ans.answers]
                return len(rows) > 0, var_names, rows
        except Exception:
            pass
        rows = [tuple(map(str, row)) for row in ans.answers]
        return len(rows) > 0, var_names, rows

    # ------------------------------ Workflow --------------------------- #
    def load_json(self) -> None:
        with open(self.input_path, 'r') as f:
            self.data = json.load(f)

    def load_initial_facts(self) -> None:
        atoms = [self.parse_atom(s) for s in self.data.get('initial_state', [])]
        # Ensure known symbols exist first to avoid accidental typos
        self._ensure_terms(atoms)
        for atom in atoms:
            self._add_fact(atom)

    def evaluate_subgoals(self) -> Dict[str, Any]:
        plans: List[Dict[str, Any]] = self.data.get('plans', [])
        overall: List[Dict[str, Any]] = []
        for idx, plan in enumerate(plans):
            sub_preds = plan.get('subgoal_predicates', [])
            parsed = [self.parse_atom(s) for s in sub_preds]
            plan_results: List[Dict[str, Any]] = []
            all_true = True
            for atom in parsed:
                is_true, bindings = self._query_atom(atom)
                plan_results.append({
                    'predicate': f"({atom.name}{(' ' + ' '.join(str(a) for a in atom.args)) if atom.args else ''})",
                    'satisfied': is_true,
                    'bindings': bindings,  # present when variables are used and satisfied
                })
                if not is_true:
                    all_true = False
            # Conjunction check across all parsed subgoals
            conj_true, conj_vars, conj_bindings = self._query_conjunction(parsed)
            overall.append({
                'plan_index': idx,
                'all_subgoals_satisfied': all_true,
                'subgoals': plan_results,
                'action': plan.get('action', []),
                'conjunction': {
                    'satisfied': conj_true,
                    'variables': conj_vars,
                    'bindings': conj_bindings,
                }
            })
        self.results = {
            'input': os.path.abspath(self.input_path),
            'initial_state': self.data.get('initial_state', []),
            'plans_evaluation': overall,
        }
        return self.results

    def save_results(self, output_path: str) -> None:
        if not self.results:
            raise RuntimeError('No results to save. Run evaluate_subgoals() first.')
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Saved results to {output_path}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description='Check block world initial state with PyDatalog')
    parser.add_argument('--input', '-i', default='nl_pddl_planner/tests_user/example_out_pddl_regression.json', help='Path to input JSON file')
    parser.add_argument('--output', '-o', default='', help='Optional path to save JSON results')
    args = parser.parse_args(argv)

    checker = BlockWorldInitialStateChecker(input_path=args.input)
    checker.load_json()
    checker.load_initial_facts()
    results = checker.evaluate_subgoals()

    # Print concise summary
    print("Initial State:")
    for s in checker.data.get('initial_state', []):
        print(f"  - {s}")
    print("\nPlans evaluation:")
    for plan_res in results['plans_evaluation']:
        idx = plan_res['plan_index']
        ok = plan_res['all_subgoals_satisfied']
        print(f"Plan {idx}: all_subgoals_satisfied={ok}")
        actions = plan_res.get('action', [])
        if actions:
            print("   Actions:")
            for act in actions:
                print(f"     - {act}")
        conj = plan_res.get('conjunction', {}) or {}
        if conj:
            print(f"   Conjunction satisfied: {conj.get('satisfied', False)}")
            vars_ = conj.get('variables') or []
            binds = conj.get('bindings')
            if vars_ and binds:
                print(f"   Conjunction bindings (variables: {vars_}):")
                for row in binds:
                    print(f"     - {row}")
        for sg in plan_res['subgoals']:
            b = ''
            if sg['bindings'] is not None:
                b = f" bindings={sg['bindings']}"
            print(f"   * {sg['predicate']}: {sg['satisfied']}{b}")

    if args.output:
        checker.save_results(args.output)


if __name__ == '__main__':
    main()
