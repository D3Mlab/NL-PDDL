#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDDL -> JSON translator tailored for Blocksworld-4ops.

Outputs:
1) Domain JSON matching the schema in files/blockworld_domain.json
2) For each problem PDDL, a JSON file with keys:
   - goal: list of [phrase, {var_or_obj: "object", ...}] entries (same style as blockworld_goal.json)
   - initial_state: { predicate_name: ["(predicate args)", ...], ... }
"""
from __future__ import annotations
import argparse
import json
import os
import re
from typing import Any, Dict, List, Tuple

# -----------------------------
# Natural language templates for Blocksworld
# -----------------------------
PRED_NL = {
    "handempty": ("the hand is empty", []),  # no args
    "clear": ("?b is clear", ["?b"]),
    "ontable": ("?b is on the table", ["?b"]),
    "on": ("?b1 is on top of ?b2", ["?b1", "?b2"]),
    "holding": ("I am holding ?b", ["?b"]),
}

ACTION_NL = {
    "pick-up": {
        "action": "pick_up",
        "name": ("pick up ?b", ["?b"]),
    },
    "put-down": {
        "action": "put_down",
        "name": ("put down ?b", ["?b"]),
    },
    "stack": {
        "action": "stack",
        "name": ("stack ?b1 on top of ?b2", ["?b1", "?b2"]),
    },
    "unstack": {
        "action": "unstack",
        "name": ("unstack ?b1 from ?b2", ["?b1", "?b2"]),
    },
}

# -----------------------------
# Public class wrapper
# -----------------------------
class PDDLToJSON:
    """Convenient, importable wrapper to use this module programmatically."""

    # Expose core parsing/conversion methods by delegating to module-level helpers
    def parse_domain(self, domain_text: str) -> Dict[str, Any]:
        return parse_domain(domain_text)

    def domain_to_blockworld_json(self, parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
        return domain_to_blockworld_json(parsed)

    def parse_problem(self, problem_text: str) -> Dict[str, Any]:
        return parse_problem(problem_text)

    def goal_to_blockworld_format(self, goal_atoms: List[Tuple[str, List[str]]]) -> List[List[Any]]:
        return goal_to_blockworld_format(goal_atoms)

    def init_to_predicate_string_dict(self, init_atoms: List[Tuple[str, List[str]]]) -> Dict[str, List[str]]:
        return init_to_predicate_string_dict(init_atoms)

    # File-oriented helpers
    def convert_domain_file(self, domain_path: str, domain_json_out: str | None = None) -> str:
        with open(domain_path, 'r') as f:
            domain_txt = f.read()
        parsed_domain = self.parse_domain(domain_txt)
        domain_json = self.domain_to_blockworld_json(parsed_domain)
        if not domain_json_out:
            base, _ = os.path.splitext(domain_path)
            domain_json_out = base + '.json'
        os.makedirs(os.path.dirname(domain_json_out), exist_ok=True)
        with open(domain_json_out, 'w') as f:
            json.dump(domain_json, f, indent=2)
        return domain_json_out

    def convert_problem_file(self, problem_path: str, out_dir: str) -> str:
        with open(problem_path, 'r') as f:
            prob_txt = f.read()
        parsed_prob = self.parse_problem(prob_txt)
        goal_json = self.goal_to_blockworld_format(parsed_prob['goal'])
        init_json = self.init_to_predicate_string_dict(parsed_prob['init'])
        result = {
            'goal': goal_json,
            'initial_state': init_json,
            'objects': parsed_prob['objects'],
        }
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(problem_path))[0] + '.json')
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
        return out_path

    def convert_problems_in_dir(self, problems_dir: str, domain_basename: str | None = None, out_dir: str | None = None) -> List[str]:
        problems: List[str] = []
        for name in os.listdir(problems_dir):
            if name.lower().endswith('.pddl') and (domain_basename is None or name != domain_basename):
                problems.append(os.path.join(problems_dir, name))
        problems = sorted(set(problems))
        out_dir_final = out_dir or problems_dir
        os.makedirs(out_dir_final, exist_ok=True)
        outputs: List[str] = []
        for prob in problems:
            outputs.append(self.convert_problem_file(prob, out_dir_final))
        return outputs

    # Convenience runner for CLI-parsed args
    def run(self, args: argparse.Namespace) -> None:
        domain_out = self.convert_domain_file(args.domain, args.domain_json)
        print(f"Wrote domain JSON -> {domain_out}")

        # Build problem list
        problems: List[str] = []
        if getattr(args, 'problem', None):
            problems.append(args.problem)
        if getattr(args, 'problems_dir', None):
            for name in os.listdir(args.problems_dir):
                if name.lower().endswith('.pddl') and name != os.path.basename(args.domain):
                    problems.append(os.path.join(args.problems_dir, name))
        problems = sorted(set(problems))

        out_dir = args.problems_json_dir or (args.problems_dir if args.problems_dir else (os.path.dirname(args.problem) if args.problem else '.'))
        os.makedirs(out_dir, exist_ok=True)

        for prob in problems:
            out_path = self.convert_problem_file(prob, out_dir)
            print(f"Wrote problem JSON -> {out_path}")
# -----------------------------
# Basic S-expression tokenizer/parser
# -----------------------------
TOKEN_RE = re.compile(r"\(|\)|[^\s()]+")


def _remove_comments(text: str) -> str:
    # PDDL comments are started with ";" and go to end of line
    lines = []
    for line in text.splitlines():
        semi = line.find(";")
        if semi != -1:
            line = line[:semi]
        lines.append(line)
    return "\n".join(lines)


def parse_sexpr(text: str) -> List[Any]:
    tokens = TOKEN_RE.findall(_remove_comments(text))
    stack: List[List[Any]] = [[]]
    for tok in tokens:
        if tok == "(":
            stack.append([])
        elif tok == ")":
            if len(stack) == 1:
                raise ValueError("Unbalanced parentheses in PDDL")
            lst = stack.pop()
            stack[-1].append(lst)
        else:
            stack[-1].append(tok)
    if len(stack) != 1:
        raise ValueError("Unbalanced parentheses in PDDL at EOF")
    return stack[0]


# -----------------------------
# Helpers to walk the PDDL AST
# -----------------------------

def _find_block(lst: List[Any], key: str) -> List[Any] | None:
    for item in lst:
        if isinstance(item, list) and item:
            if isinstance(item[0], str) and item[0].lower() == key:
                return item
    return None


def _find_all_blocks(lst: List[Any], key: str) -> List[List[Any]]:
    out = []
    for item in lst:
        if isinstance(item, list) and item:
            if isinstance(item[0], str) and item[0].lower() == key:
                out.append(item)
    return out


def _lower_deep(node: Any) -> Any:
    if isinstance(node, str):
        return node.lower()
    if isinstance(node, list):
        return [_lower_deep(x) for x in node]
    return node


# -----------------------------
# Domain parsing
# -----------------------------

def parse_domain(domain_text: str) -> Dict[str, Any]:
    ast = parse_sexpr(domain_text)
    ast = _lower_deep(ast)

    # Accept either [define ...] or [[define ...]] from the tokenizer
    if ast and isinstance(ast[0], list) and ast[0] and ast[0][0] == 'define' and len(ast) == 1:
        ast = ast[0]
    # Expect (define (domain ...) ...)
    if not ast or ast[0] != 'define':
        raise ValueError("Not a PDDL define form for domain")

    # Extract :predicates block
    predicates_block = None
    actions_blocks: List[List[Any]] = []

    for item in ast[1:]:
        if isinstance(item, list) and item:
            if item[0] == 'domain':
                continue
            if item[0] == ':predicates':
                predicates_block = item
            if item[0] == ':action':
                actions_blocks.append(item)

    if predicates_block is None:
        raise ValueError(":predicates block not found")

    # Predicates
    predicate_defs: List[Tuple[str, List[str]]] = []
    for pred in predicates_block[1:]:
        # pred like: ['clear', ['?x']] or ['handempty'] or ['on', ['?x'], ['?y']]
        if not isinstance(pred, list) or not pred:
            continue
        name = pred[0]
        args = []
        for arg in pred[1:]:
            if isinstance(arg, list) and arg:
                # Could be typed: ['?x', '-', 'block']
                if '-' in arg:
                    args.append(arg[0])
                else:
                    args.extend([a for a in arg if isinstance(a, str) and a.startswith('?')])
            elif isinstance(arg, str) and arg.startswith('?'):
                args.append(arg)
        predicate_defs.append((name, args))

    # Actions
    actions = []
    for act in actions_blocks:
        # [:action, name, :parameters, [...], :precondition, <expr>, :effect, <expr>]
        try:
            raw_name = act[1]
        except Exception as e:
            raise ValueError(f"Malformed :action: {act}") from e

        params: List[str] = []
        precond: List[Any] | None = None
        effect: List[Any] | None = None

        i = 2
        while i < len(act):
            key = act[i]
            if key == ':parameters':
                params_expr = act[i+1]
                params = []
                # params_expr is typically a list like ['?ob'] or ['?b1','?b2'] or nested typed lists
                if isinstance(params_expr, list):
                    for p in params_expr:
                        if isinstance(p, list):
                            if '-' in p:
                                params.append(p[0])
                            else:
                                params.extend([a for a in p if isinstance(a, str) and a.startswith('?')])
                        elif isinstance(p, str) and p.startswith('?'):
                            params.append(p)
                i += 2
            elif key == ':precondition':
                precond = act[i+1]
                i += 2
            elif key == ':effect':
                effect = act[i+1]
                i += 2
            else:
                i += 1

        actions.append({
            'name': raw_name,
            'params': params,
            'precond': precond,
            'effect': effect,
        })

    return {
        'predicates': predicate_defs,
        'actions': actions,
    }


# -----------------------------
# Conversions to target JSON schema
# -----------------------------

def _nl_predicate(pred_name: str, args: List[str]) -> Tuple[str, Dict[str, str]]:
    if pred_name not in PRED_NL:
        # Fallback: raw S-expression text
        phrase = f"({pred_name} {' '.join(args)})".strip()
        mapping = {a: 'object' for a in args}
        return phrase, mapping
    template, std_vars = PRED_NL[pred_name]
    # Map by position
    repl: Dict[str, str] = {}
    for i, v in enumerate(std_vars):
        if i < len(args):
            repl[v] = args[i]
    phrase = template
    for k, v in repl.items():
        phrase = phrase.replace(k, v)
    # Variable mapping should use std vars as keys with type 'object'
    # Replace keys with actual object/variable names present in phrase
    var_dict = {}
    for i, v in enumerate(std_vars):
        if i < len(args):
            var_dict[args[i]] = 'object'
    return phrase, var_dict


def _extract_literals(expr: Any) -> Tuple[List[Tuple[str, List[str]]], List[Tuple[str, List[str]]]]:
    """Return (positives, negatives) lists of (pred, args)."""
    pos: List[Tuple[str, List[str]]] = []
    neg: List[Tuple[str, List[str]]] = []

    def walk(e: Any, negated: bool = False):
        if isinstance(e, str):
            return
        if not isinstance(e, list) or not e:
            return
        head = e[0]
        if head == 'and':
            for sub in e[1:]:
                walk(sub, negated)
            return
        if head == 'not' and len(e) == 2:
            walk(e[1], True)
            return
        # otherwise this should be an atom: [pred, arg1, arg2, ...]
        pred = e[0]
        args = [a for a in e[1:] if isinstance(a, str)]
        if negated:
            neg.append((pred, args))
        else:
            pos.append((pred, args))

    walk(expr, False)
    return pos, neg


def domain_to_blockworld_json(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Predicates
    pred_entries = []
    for name, args in parsed['predicates']:
        if name in PRED_NL:
            template, std_vars = PRED_NL[name]
            phrase = template
            # Use std vars directly for the var map
            var_map = {v: 'object' for v in std_vars}
        else:
            phrase, var_map = _nl_predicate(name, [f"?b{i+1}" if i > 0 else "?b" for i in range(len(args))])
        pred_entries.append([phrase, var_map])
    domain_json: List[Dict[str, Any]] = [{"Predicate": pred_entries}]

    # Actions
    for act in parsed['actions']:
        name: str = act['name']
        pre = act['precond']
        eff = act['effect']

        # Normalize action naming and NL
        if name in ACTION_NL:
            action_key = ACTION_NL[name]['action']
            action_name_phrase, action_vars = ACTION_NL[name]['name']
        else:
            action_key = name.replace('-', '_')
            action_name_phrase = name.replace('-', ' ') + ' ' + ' '.join(act['params'])
            action_vars = act['params']

        # Parameters dict using standardized variables by position
        params_dict: Dict[str, str] = {}
        for i, _ in enumerate(action_vars):
            std = f"?b{i+1}" if i > 0 else "?b"
            params_dict[std] = 'object'

        # Preconditions -> NL pairs
        pos_pre, neg_pre = _extract_literals(pre) if pre else ([], [])
        precond_entries: List[List[Any]] = []
        for pred, args in pos_pre:
            # Map args to standardized variables by position
            std_args = [f"?b{i+1}" if i > 0 else "?b" for i in range(len(args))]
            phrase, var_map = _nl_predicate(pred, std_args)
            precond_entries.append([phrase, var_map])

        # Effects -> Positive/Negative NL pairs
        pos_eff, neg_eff = _extract_literals(eff) if eff else ([], [])
        pos_entries: List[List[Any]] = []
        for pred, args in pos_eff:
            std_args = [f"?b{i+1}" if i > 0 else "?b" for i in range(len(args))]
            phrase, var_map = _nl_predicate(pred, std_args)
            pos_entries.append([phrase, var_map])

        neg_entries: List[List[Any]] = []
        for pred, args in neg_eff:
            std_args = [f"?b{i+1}" if i > 0 else "?b" for i in range(len(args))]
            phrase, var_map = _nl_predicate(pred, std_args)
            neg_entries.append([phrase, var_map])

        domain_json.append({
            "Action": action_key,
            "Action name": [action_name_phrase, {k: 'object' for k in params_dict.keys()}],
            "Parameters": {k: 'object' for k in params_dict.keys()},
            "Preconditions": precond_entries,
            "Effects": {
                "Positive": pos_entries,
                "Negative": neg_entries,
            }
        })

    return domain_json


# -----------------------------
# Problem parsing and conversion
# -----------------------------

def parse_problem(problem_text: str) -> Dict[str, Any]:
    ast = parse_sexpr(problem_text)
    ast = _lower_deep(ast)
    # Accept either [define ...] or [[define ...]] from the tokenizer
    if ast and isinstance(ast[0], list) and ast[0] and ast[0][0] == 'define' and len(ast) == 1:
        ast = ast[0]
    if not ast or ast[0] != 'define':
        raise ValueError("Not a PDDL define form for problem")

    objects: List[str] = []
    init_block: List[Any] | None = None
    goal_block: List[Any] | None = None

    for item in ast[1:]:
        if isinstance(item, list) and item:
            tag = item[0]
            if tag == ':objects':
                # flat objects list; ignore typing
                for tok in item[1:]:
                    if isinstance(tok, str) and not tok.startswith('-'):
                        objects.append(tok)
            elif tag == ':init':
                init_block = item
            elif tag == ':goal':
                goal_block = item

    if init_block is None or goal_block is None:
        raise ValueError(":init or :goal not found in problem")

    # Initial state: collect atoms inside (:init ...)
    init_atoms: List[Tuple[str, List[str]]] = []
    for entry in init_block[1:]:
        if isinstance(entry, list) and entry:
            pred = entry[0]
            args = [a for a in entry[1:] if isinstance(a, str)]
            init_atoms.append((pred, args))

    # Goal: expression is inside (:goal <expr>)
    goal_expr = goal_block[1] if len(goal_block) > 1 else []
    pos_goal, _ = _extract_literals(goal_expr)

    return {
        'objects': objects,
        'init': init_atoms,
        'goal': pos_goal,
    }


def goal_to_blockworld_format(goal_atoms: List[Tuple[str, List[str]]]) -> List[List[Any]]:
    """Return list of [phrase, {obj: 'object', ...}] pairs."""
    out: List[List[Any]] = []
    for pred, args in goal_atoms:
        if pred in PRED_NL:
            template, std_vars = PRED_NL[pred]
            phrase = template
            # Replace std vars with objects by position
            for i, sv in enumerate(std_vars):
                if i < len(args):
                    phrase = phrase.replace(sv, args[i])
        else:
            phrase = f"({pred} {' '.join(args)})"
        var_map = {obj: 'object' for obj in args}
        out.append([phrase, var_map])
    return out


def init_to_predicate_string_dict(init_atoms: List[Tuple[str, List[str]]]) -> Dict[str, List[str]]:
    d: Dict[str, List[str]] = {}
    for pred, args in init_atoms:
        s = f"({pred}{'' if not args else ' ' + ' '.join(args)})"
        d.setdefault(pred, []).append(s)
    return d


# -----------------------------
# Main CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Translate PDDL (Blocksworld) to JSON")
    ap.add_argument('--domain', required=True, help='Path to domain.pddl')
    ap.add_argument('--domain-json', help='Where to write domain JSON (defaults next to domain file with .json)')
    ap.add_argument('--problem', help='Single problem .pddl to convert')
    ap.add_argument('--problems-dir', help='Directory containing problem .pddl files (convert all)')
    ap.add_argument('--problems-json-dir', help='Directory to write problem JSON outputs (defaults to problems dir)')
    args = ap.parse_args()
    converter = PDDLToJSON()
    converter.run(args)


if __name__ == '__main__':
    main()
