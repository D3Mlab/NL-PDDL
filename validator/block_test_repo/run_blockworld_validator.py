#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone end-to-end pipeline for Block-World regression text dumps (no CLI calls).

Given a directory of text files (e.g., outputs like those in
block_world_test/planbench/blockworld_results_depth10), this script will:

1) Convert each text file to the JSON schema used by example_out_pddl_regression.json
   by parsing the text directly (replicating logic similar to block_text_converter.py).
2) Validate each JSON's subgoal plans with VAL by generating temporary problem files
   (replicating logic similar to validate_with_val.py) and produce a per-file summary JSON.
3) Check initial state satisfaction for each JSON via PyDatalog by importing and
   using BlockWorldInitialStateChecker (from initial_state_checker.py) directly.

This script does not modify other project files and does not invoke existing CLI scripts.

Usage example:
  python run_blockworld_validator.py \
      --input-dir block_world_test/planbench/blockworld_results_depth10 \
      --output-dir block_world_test/planbench/blockworld_results_depth10_validation \
      --goal-mode first_subgoal \
      --domain ../nl_pddl_planner/tests_usr/block_world_test/planbench/generated_domain.pddl \
      --problem ../nl_pddl_planner/tests_usr/block_world_test/planbench/instance-10.pddl \
      --val ./val/bin/Validate

All paths are interpreted relative to this script unless absolute.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Optional import: reuse the PyDatalog checker internals without running its CLI
try:
    from initial_state_checker import BlockWorldInitialStateChecker
except Exception:
    BlockWorldInitialStateChecker = None  # type: ignore

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = Path(__file__).resolve().parent.parent

# -------------------- In-file defaults (edit here if you prefer not to pass CLI flags) --------------------
DEFAULT_INPUT_DIR = os.path.join(RESULTS_DIR, 'pddl_planner/tests/results/blockworld_results_depth10')
DEFAULT_OUTPUT_DIR = os.path.join(RESULTS_DIR, 'pddl_planner/tests/results/blockworld_results_depth10_validation')

# Optional config file next to this script to override defaults/CLI
CONFIG_FILENAME = os.path.join(SCRIPT_DIR, 'run_blockworld_validator.config.json')

# -------------------- Text -> JSON conversion helpers (similar to block_text_converter) --------------------

def read_text(path: str | None) -> str:
    if path:
        with open(path, 'r') as f:
            return f.read()
    return sys.stdin.read()


def strip_outer_double_parens(s: str) -> str:
    t = s.strip()
    if t.startswith('((') and t.endswith('))'):
        return t[2:-2].strip()
    if t.startswith('(') and t.endswith(')'):
        return t[1:-1].strip()
    return t


def split_conjunction_clauses(s: str) -> List[str]:
    parts = [p.strip() for p in s.split('∧')]
    parts = [p for p in parts if p]
    cleaned = []
    for p in parts:
        q = p
        if q.startswith('(') and q.endswith(')'):
            q = q[1:-1].strip()
        cleaned.append(q)
    return cleaned


def parse_actions_line(line: str) -> List[str]:
    t = line.strip()
    if t == '[]' or t == '':
        return []
    pattern = r"Action\('([^']+)'\s*\(\[([^\]]*)\]\)\)"
    actions: List[str] = []
    try:
        for m in re.finditer(pattern, t):
            name = m.group(1).strip()
            args_str = m.group(2).strip()
            args = [a.strip() for a in args_str.split(',') if a.strip()] if args_str else []
            actions.append(f"{name}({', '.join(args)})")
    except re.error:
        inside = t.strip()
        if inside.startswith('[') and inside.endswith(']'):
            inside = inside[1:-1]
        chunks = re.split(r"\)\]\)\s*,\s*Action\(", inside)
        for ch in chunks:
            if not ch.startswith('Action('):
                ch = 'Action(' + ch
            m2 = re.search(r"Action\('([^']+)'\s*\(\[([^\]]*)\]\)\)", ch)
            if not m2:
                continue
            name = m2.group(1).strip()
            args_str = m2.group(2).strip()
            args = [a.strip() for a in args_str.split(',') if a.strip()] if args_str else []
            actions.append(f"{name}({', '.join(args)})")
    return actions


def parse_substitution_line(line: str) -> Dict[str, str]:
    t = line.strip()
    if t == '{}' or t == '':
        return {}
    if t.startswith('{') and t.endswith('}'):
        t = t[1:-1].strip()
    if not t:
        return {}
    subs: Dict[str, str] = {}
    pairs = [p.strip() for p in t.split(',') if p.strip()]
    for p in pairs:
        if ':' not in p:
            continue
        k, v = p.split(':', 1)
        subs[k.strip()] = v.strip()
    return subs


def extract_initial_block(text: str) -> Tuple[List[str], List[str], Dict[str, str], int]:
    m = re.search(r"(?im)^Initial\s+State\s*:\s*$", text)
    if m:
        pos = m.end()
        start_after = text[pos:].lstrip()
    else:
        lines_all = text.splitlines()
        fallback_idx = -1
        for i, ln in enumerate(lines_all):
            if ln.strip().startswith('(('):
                fallback_idx = i
                break
        if fallback_idx == -1:
            preview = "\n".join(lines_all[:10])
            raise RuntimeError("Could not find 'Initial State:' or any '((...)' block. Preview:\n" + preview)
        start_after = "\n".join(lines_all[fallback_idx:])
        pos = text.find(lines_all[fallback_idx])

    after = start_after
    lines = after.splitlines()

    def next_nonempty(start_idx: int) -> Tuple[str, int]:
        i = start_idx
        while i < len(lines) and lines[i].strip() == "":
            i += 1
        if i >= len(lines):
            return "", i
        return lines[i], i + 1

    l1, idx = next_nonempty(0)
    clauses_raw_block = strip_outer_double_parens(l1)
    initial_raw_clauses = split_conjunction_clauses(clauses_raw_block)

    l2, idx2 = next_nonempty(idx)
    if l2.strip().startswith('['):
        initial_actions_raw = parse_actions_line(l2)
        idx = idx2
        l3, idx3 = next_nonempty(idx)
        if l3.strip().startswith('{'):
            initial_subs = parse_substitution_line(l3)
            idx = idx3
        else:
            initial_subs = {}
    else:
        initial_actions_raw = []
        initial_subs = {}

    consumed = "\n".join(lines[:idx])
    end_pos = pos + after.index(lines[0]) + len(consumed)
    return initial_raw_clauses, initial_actions_raw, initial_subs, end_pos


essubgoal_header_re = re.compile(r"(?im)^Subgoal(?:\s+S\d+)?\s*:\s*$")

def extract_subgoals(text: str, start_pos: int) -> List[Tuple[str, List[str], Dict[str, str]]]:
    m = re.search(r"(?im)^Regressed\s+goals\s*:\s*$", text[start_pos:])
    if m:
        anchor = start_pos + m.end()
    else:
        m2 = re.search(r"(?im)^Regressed\s+goals\s*:\s*$", text)
        if not m2:
            return []
        anchor = m2.end()

    segment = text[anchor:]
    header_iter = list(essubgoal_header_re.finditer(segment))
    if not header_iter:
        return []

    def slice_region(i: int) -> str:
        start = header_iter[i].end()
        end = header_iter[i + 1].start() if i + 1 < len(header_iter) else len(segment)
        return segment[start:end]

    results: List[Tuple[str, List[str], Dict[str, str]]] = []
    for i in range(len(header_iter)):
        region = slice_region(i)
        lines = [ln for ln in region.splitlines() if ln.strip() != ""]
        if not lines:
            continue
        idx = 0
        clauses_line = lines[idx]
        if not clauses_line.strip().startswith('(('):
            found = next((k for k, ln in enumerate(lines) if ln.strip().startswith('((')), -1)
            if found == -1:
                continue
            idx = found
            clauses_line = lines[idx]
        clauses_block = strip_outer_double_parens(clauses_line)
        subgoal_clauses_block = clauses_block
        idx += 1
        actions_raw: List[str] = []
        if idx < len(lines) and lines[idx].strip().startswith('['):
            actions_raw = parse_actions_line(lines[idx])
            idx += 1
        subs: Dict[str, str] = {}
        if idx < len(lines) and lines[idx].strip().startswith('{'):
            subs = parse_substitution_line(lines[idx])
            idx += 1
        results.append((subgoal_clauses_block, actions_raw, subs))

    return results


# PDDL conversion mappings (subset replicated)

def clauses_to_predicates(clauses: List[str]) -> List[str]:
    preds: List[str] = []
    for s in clauses:
        t = re.sub(r"\s+", " ", s.strip())
        m = re.match(r"^is clear\(([^)]+)\)$", t)
        if m:
            preds.append(f"(clear {m.group(1).strip()})")
            continue
        m = re.match(r"^is on the table\(([^)]+)\)$", t)
        if m:
            preds.append(f"(ontable {m.group(1).strip()})")
            continue
        m = re.match(r"^the hand is empty\(\)$", t)
        if m:
            preds.append("(handempty)")
            continue
        m = re.match(r"^i am holding\(([^)]+)\)$", t, flags=re.IGNORECASE)
        if m:
            preds.append(f"(holding {m.group(1).strip()})")
            continue
        m = re.match(r"^is on top of\(([^,]+),\s*([^)]+)\)$", t)
        if m:
            a = m.group(1).strip()
            b = m.group(2).strip()
            preds.append(f"(on {a} {b})")
            continue
    return preds


def actions_to_pddl(actions: List[str]) -> List[str]:
    out: List[str] = []
    for s in actions:
        m = re.match(r"^(.*)\((.*)\)$", s.strip())
        if not m:
            out.append(s)
            continue
        name = m.group(1).strip().lower()
        args_str = m.group(2).strip()
        args = [a.strip() for a in args_str.split(',') if a.strip()]
        if 'pick up' in name or 'pick-up' in name:
            out.append(f"pick-up({args[0]})" if args else "pick-up()")
        elif 'put down' in name or 'put-down' in name:
            out.append(f"put-down({args[0]})" if args else "put-down()")
        elif name.startswith('stack'):
            if len(args) >= 2:
                out.append(f"stack({args[0]}, {args[1]})")
            elif len(args) == 1:
                out.append(f"stack({args[0]})")
            else:
                out.append("stack()")
        elif name.startswith('unstack'):
            if len(args) >= 2:
                out.append(f"unstack({args[0]}, {args[1]})")
            elif len(args) == 1:
                out.append(f"unstack({args[0]})")
            else:
                out.append("unstack()")
        else:
            out.append(s)
    return out


# -------------------- VAL validation helpers (subset replicated) --------------------

def normalize_predicates(preds: List[str]) -> List[str]:
    norm: List[str] = []
    for p in preds:
        s = p.strip()
        if not (s.startswith('(') and s.endswith(')')):
            parts = s.split()
            if parts:
                s = '(' + ' '.join(parts) + ')'
        def repl(m: re.Match[str]) -> str:
            tok = m.group(0)
            tok = tok.replace('?', '').lower()
            return tok
        inner = s[1:-1]
        inner = re.sub(r"[A-Za-z0-9_?\-]+", repl, inner)
        norm.append('(' + inner + ')')
    return norm


def build_init_section(predicates: List[str]) -> str:
    preds_text = "\n  ".join(predicates)
    return f"(:init\n  {preds_text}\n)"


def build_goal_section(predicates: List[str]) -> str:
    if not predicates:
        return "(:goal (and))"
    preds_text = "\n    ".join(predicates)
    return f"(:goal\n  (and\n    {preds_text}\n  )\n)"


def actions_to_val_plan(actions: List[str]) -> str:
    lines = []
    for a in actions:
        a = a.strip()
        m = re.match(r"^([a-zA-Z0-9_-]+)\((.*)\)$", a)
        if not m:
            if a.startswith('(') and a.endswith(')'):
                lines.append(a)
            else:
                lines.append(f"({a})")
            continue
        name = m.group(1)
        args_str = m.group(2).strip()
        args = [x.strip() for x in args_str.split(',') if x.strip()]
        args = [re.sub(r"^\?", "", t).lower() for t in args]
        if args:
            lines.append(f"({name} {' '.join(args)})")
        else:
            lines.append(f"({name})")
    return "\n".join(lines) + "\n"


def extract_problem_sections(problem_text: str) -> Tuple[str, str, str]:
    goal_match = re.search(r"\(:goal\s*(\([^)]*(?:\([^)]*\)[^)]*)*\))\s*\)", problem_text, re.DOTALL)
    goal_text = goal_match.group(0) if goal_match else ""
    init_pattern = re.compile(r"\(:init\s*(.*?)\)", re.DOTALL)
    init_match = init_pattern.search(problem_text)
    if not init_match:
        raise RuntimeError("Could not find (:init ...) section in problem file")
    init_start, init_end = init_match.span()
    prefix = problem_text[:init_start]
    suffix = problem_text[init_end:]
    return prefix, goal_text, suffix


def extract_objects_from_prefix(prefix: str) -> Tuple[str, List[str], str]:
    m = re.search(r"\(:objects\s*([^)]*)\)\s*", prefix, re.DOTALL)
    if not m:
        return prefix, [], ''
    start, end = m.span()
    before = prefix[:start]
    after = prefix[end:]
    objs_raw = m.group(1)
    objs = [tok for tok in re.split(r"\s+", objs_raw.strip()) if tok]
    return before, objs, after


def format_objects_section(objs: List[str]) -> str:
    return "(:objects " + " ".join(objs) + " )\n"


def collect_symbols_from_text(text: str) -> List[str]:
    candidates = re.findall(r"[A-Za-z0-9_\-]+", text)
    exclude = {"and", "init", "goal", "define", "domain", "objects",
               "clear", "ontable", "handempty", "holding", "on"}
    syms = []
    for t in candidates:
        if t.lower() in exclude:
            continue
        if re.fullmatch(r"\d+", t):
            continue
        syms.append(t)
    return syms


def parse_domain_name(domain_pddl: str) -> str:
    """Parse the domain name from a domain PDDL file contents.
    Fallback to 'domain' if not found."""
    m = re.search(r"\(define\s*\(domain\s+([^\s\)]+)\)", domain_pddl, flags=re.IGNORECASE)
    return m.group(1) if m else "domain"


def run_val_validate(val_bin: str, domain_path: str, problem_path: str, plan_path: str) -> Tuple[bool, str]:
    cmd = [val_bin, '-v', domain_path, problem_path, plan_path]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        out = proc.stdout
        success = ('Plan valid' in out) or ('Successful plans:' in out)
        return success, out
    except Exception as e:
        return False, f"Exception running VAL: {e}"


# -------------------- Pipeline --------------------

def build_json_from_text(text_path: Path, goal_mode: str) -> Dict[str, Any]:
    text = read_text(str(text_path))
    init_clauses_raw, _init_actions_raw, _init_subs, end_pos = extract_initial_block(text)
    init_predicates = clauses_to_predicates(init_clauses_raw)
    subgoals = extract_subgoals(text, end_pos)

    plans_out: List[Dict[str, Any]] = []
    def apply_subs_to_action(s: str, subs: Dict[str, str]) -> str:
        if not subs:
            return s
        for k in sorted(subs.keys(), key=len, reverse=True):
            try:
                pat = re.compile(r"%s(?![0-9A-Za-z_])" % re.escape(k))
                s = pat.sub(str(subs[k]), s)
            except Exception:
                s = s.replace(k, str(subs[k]))
        return s

    for (subgoal_raw_str, actions_raw_list, substitution) in subgoals:
        subgoal_clauses = split_conjunction_clauses(subgoal_raw_str)
        subgoal_preds = clauses_to_predicates(subgoal_clauses)
        actions_after_subs = [apply_subs_to_action(a, substitution) for a in actions_raw_list]
        pddl_actions = actions_to_pddl(actions_after_subs)
        plans_out.append({
            'subgoal_raw': {'disjoint': subgoal_clauses},
            'action_raw': actions_raw_list,
            'substitution': substitution,
            'subgoal_predicates': subgoal_preds,
            'action': pddl_actions,
        })

    if goal_mode == 'empty':
        goal_predicates: List[str] = []
    elif goal_mode == 'first_subgoal':
        goal_predicates = plans_out[0]['subgoal_predicates'] if plans_out else []
    elif goal_mode == 'last_subgoal':
        goal_predicates = plans_out[-1]['subgoal_predicates'] if plans_out else []
    else:  # union_subgoals
        seen = set()
        goal_predicates = []
        for p in plans_out:
            for pr in p.get('subgoal_predicates', []):
                if pr not in seen:
                    seen.add(pr)
                    goal_predicates.append(pr)

    return {
        'initial_state': init_predicates,
        'goal_predicates': goal_predicates,
        'plans': plans_out,
    }


def process_with_val(summary_out: Path, data: Dict[str, Any], domain_path: Path, problem_path: Path | None, val_bin: Path) -> Dict[str, Any]:
    plans = data.get('plans', []) or []
    domain_text = domain_path.read_text()
    domain_name = parse_domain_name(domain_text)

    base_problem_text = None
    prefix = ""
    problem_goal_text = ""
    if problem_path is not None and str(problem_path) and Path(problem_path).exists():
        base_problem_text = problem_path.read_text()
        prefix, problem_goal_text, _suffix = extract_problem_sections(base_problem_text)

    json_goal_preds_raw = data.get('goal_predicates', []) or []
    if json_goal_preds_raw:
        goal_norm = normalize_predicates(json_goal_preds_raw)
        goal_text = build_goal_section(goal_norm)
    else:
        goal_text = problem_goal_text

    results = []
    failures_log: List[str] = []
    workdir = Path(tempfile.mkdtemp(prefix='val_subgoal_'))

    try:
        for idx, item in enumerate(plans):
            sub_preds = item.get('subgoal_predicates', []) or []
            actions = item.get('action', []) or []
            substitution = item.get('substitution', {}) or {}
            subgoal_raw = item.get('subgoal_raw', {}) or {}
            actions_raw = item.get('action_raw', []) or []

            init_preds = normalize_predicates(sub_preds)
            init_text = build_init_section(init_preds)

            # If no problem prefix was provided, synthesize a minimal one using the domain name
            if prefix:
                before_obj, existing_objs, after_obj = extract_objects_from_prefix(prefix)
                new_prefix = before_obj + format_objects_section(sorted(set(existing_objs))) + after_obj
            else:
                header = f"(define (problem auto)\n  (:domain {domain_name})\n"
                before_obj = header
                existing_objs = []
                after_obj = ""  # nothing else until :init
                new_prefix = before_obj  # we'll append objects next
            needed_syms = set(collect_symbols_from_text(init_text) + collect_symbols_from_text(goal_text))
            merged_objs = sorted(set(existing_objs) | needed_syms)
            new_prefix = before_obj + format_objects_section(merged_objs) + after_obj
            temp_problem_text = new_prefix + init_text + "\n" + goal_text

            if "(define" not in temp_problem_text or "(:domain" not in temp_problem_text:
                if base_problem_text:
                    m_head = re.search(r"^\(define[\s\S]*?\n\(:domain[\s\S]*?\)\n", base_problem_text)
                    if m_head:
                        header = m_head.group(0)
                    else:
                        header = base_problem_text.split("(:init", 1)[0]
                else:
                    header = f"(define (problem auto)\n  (:domain {domain_name})\n"
                temp_problem_text = header + temp_problem_text

            opens = temp_problem_text.count('(')
            closes = temp_problem_text.count(')')
            if closes < opens:
                temp_problem_text = temp_problem_text.rstrip() + "\n" + (')' * (opens - closes)) + "\n"

            problem_tmp = workdir / f'problem_{idx}.pddl'
            plan_tmp = workdir / f'plan_{idx}.plan'
            problem_tmp.write_text(temp_problem_text)
            plan_text = actions_to_val_plan(actions)
            plan_tmp.write_text(plan_text)

            ok, log = run_val_validate(str(val_bin), str(domain_path), str(problem_tmp), str(plan_tmp))
            results.append({
                'index': idx,
                'init_predicates': init_preds,
                'actions': actions,
                'plan_val': plan_text.strip().splitlines(),
                'valid': ok,
                'val_output': log,
                'substitution': substitution,
                'subgoal_raw': subgoal_raw,
                'actions_raw': actions_raw,
            })
            if not ok:
                block_lines = [f"[INVALID] Subgoal #{idx}"]
                raw_clauses = []
                if isinstance(subgoal_raw, dict):
                    raw_clauses = subgoal_raw.get('disjoint') or subgoal_raw.get('disjoint_1', []) or []
                if raw_clauses:
                    block_lines.append("subgoal_raw (clauses):")
                    block_lines.extend(["  " + c for c in raw_clauses])
                if actions_raw:
                    block_lines.append("actions_raw:")
                    block_lines.extend(["  " + a for a in actions_raw])
                block_lines.append("init_predicates:")
                block_lines.extend(["  " + p for p in init_preds])
                block_lines.append("actions:")
                block_lines.extend(["  " + a for a in actions])
                if substitution:
                    try:
                        subs_lines = [f"{k}: {v}" for k, v in substitution.items()]
                    except Exception:
                        subs_lines = [str(substitution)]
                    block_lines.append("substitution:")
                    block_lines.extend(["  " + line for line in subs_lines])
                block_lines.append("plan_val:")
                block_lines.extend(["  " + ln for ln in plan_text.strip().splitlines()])
                block_lines.append("val_output:")
                block_lines.append(log)
                block_lines.append("")
                failures_log.append("\n".join(block_lines))

        # Write summary
        failed = sum(1 for r in results if not r.get('valid', False))
        total = len(results)
        summary = {
            'domain': str(domain_path.resolve()),
            'base_problem': (str(problem_path.resolve()) if problem_path else ''),
            'json_inline': data,  # include inline for traceability
            'val_bin': str(val_bin.resolve()) if not str(val_bin).startswith('./') else str(val_bin),
            'results': results,
            'failed': failed,
            'total': total,
        }
        summary_out.write_text(json.dumps(summary, indent=2))

        if failures_log:
            (summary_out.parent / (summary_out.stem.replace('_val_summary', '') + '_validation_failures.txt')).write_text("\n\n".join(failures_log))

        return summary
    finally:
        # Keep workdir for debugging? Comment out to retain.
        # shutil.rmtree(workdir, ignore_errors=True)
        pass


def check_initial_state(json_path: Path, out_txt_path: Path) -> Tuple[int, int]:
    """Run the PyDatalog initial-state check.
    Returns a tuple (satisfied_count, total_plans)."""
    if BlockWorldInitialStateChecker is None:
        out_txt_path.write_text("initial_state_checker not available for import; skipping.")
        return 0, 0
    checker = BlockWorldInitialStateChecker(input_path=str(json_path))
    checker.load_json()
    checker.load_initial_facts()
    results = checker.evaluate_subgoals()

    # Mimic the readable output style from test_initial_state.py for satisfied conjunctions
    lines: List[str] = []
    plans_eval = results.get('plans_evaluation', []) if isinstance(results, dict) else []
    total_plans = len(plans_eval)
    satisfied_count = 0
    for plan_res in plans_eval:
        conj = (plan_res.get('conjunction') or {}) if isinstance(plan_res, dict) else {}
        if not bool(conj.get('satisfied', False)):
            continue
        satisfied_count += 1
        idx = plan_res.get('plan_index', '?')
        lines.append("\n====================================")
        lines.append(f"Plan {idx}")
        lines.append("------------------------------------")
        try:
            conj_preds = [sg.get('predicate', '') for sg in plan_res.get('subgoals', [])]
            if conj_preds:
                lines.append("Subgoal Predicates:")
                lines.extend([f"  - {p}" for p in conj_preds])
                lines.append("")
        except Exception:
            pass
        try:
            raw_plan = (checker.data.get('plans', []) or [])[idx] if hasattr(checker, 'data') else None
            raw_disj = raw_plan.get('subgoal_raw', {}) if isinstance(raw_plan, dict) else {}
            clauses = raw_disj.get('disjoint') or raw_disj.get('disjoint_1', []) or [] if isinstance(raw_disj, dict) else []
            if clauses:
                lines.append("Disjoint (raw clauses):")
                lines.extend([f"  - {c}" for c in clauses])
                lines.append("")
        except Exception:
            pass
        actions = plan_res.get('action', [])
        if actions:
            lines.append("Actions:")
            lines.extend([f"  - {a}" for a in actions])
            lines.append("")
        try:
            if plan_res.get('subgoals'):
                lines.append("Conjunction:")
                lines.append("  " + " & ".join([sg.get('predicate', '') for sg in plan_res['subgoals']]))
                lines.append("")
        except Exception:
            pass
        lines.append(f"Conjunction satisfied: {conj.get('satisfied', False)}")
        vars_ = conj.get('variables') or []
        binds = conj.get('bindings')
        if vars_ and binds:
            lines.append(f"Conjunction bindings (variables: {vars_}):")
            for row in binds:
                lines.append(f"  - {row}")
    out_txt_path.write_text("\n".join(lines) + "\n")
    return satisfied_count, total_plans


def main() -> None:
    ap = argparse.ArgumentParser(description='Standalone end-to-end: convert -> VAL validate -> PyDatalog init check')
    ap.add_argument('--input-dir', default=str(DEFAULT_INPUT_DIR), help='Directory containing input .txt regression dumps')
    ap.add_argument('--output-dir', default=str(DEFAULT_OUTPUT_DIR), help='Directory to write per-file outputs')
    ap.add_argument('--goal-mode', choices=['empty','first_subgoal','last_subgoal','union_subgoals'], default='first_subgoal', help='How to populate goal_predicates in the converted JSON')
    ap.add_argument('--domain', default=str((SCRIPT_DIR / '../block_test_repo/generated_domain.pddl').resolve()), help='Path to domain.pddl for VAL')
    ap.add_argument('--problem', default=str((SCRIPT_DIR / '../block_test_repo/instance-10.pddl').resolve()), help='Path to a base problem.pddl (goal may be replaced)')
    ap.add_argument('--val', default=str((SCRIPT_DIR / 'val/bin/Validate').resolve()), help='Path to VAL Validate binary')
    ap.add_argument('--limit', type=int, default=0, help='Optional limit on number of files to process (0 = no limit)')
    args = ap.parse_args()

    # Optional JSON config override
    try:
        if CONFIG_FILENAME.exists():
            cfg = json.loads(CONFIG_FILENAME.read_text())
            # Only override if keys present
            args.input_dir = cfg.get('input_dir', cfg.get('input-dir', args.input_dir))
            args.output_dir = cfg.get('output_dir', cfg.get('output-dir', args.output_dir))
            args.goal_mode = cfg.get('goal_mode', cfg.get('goal-mode', args.goal_mode))
            args.domain = cfg.get('domain', args.domain)
            args.problem = cfg.get('problem', args.problem)
            args.val = cfg.get('val', args.val)
            args.limit = int(cfg.get('limit', args.limit))
    except Exception as e:
        print(f"Warning: failed to read config {CONFIG_FILENAME}: {e}")

    input_dir = Path(args.input_dir).expanduser()
    if not input_dir.is_absolute():
        input_dir = (SCRIPT_DIR / input_dir).resolve()
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (SCRIPT_DIR / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory does not exist or is not a directory: {input_dir}")
        sys.exit(2)

    txt_files = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == '.txt'])
    if args.limit and args.limit > 0:
        txt_files = txt_files[: args.limit]
    if not txt_files:
        print(f"No .txt files found under {input_dir}")
        sys.exit(0)

    print(f"Found {len(txt_files)} txt file(s) to process under {input_dir}")

    for idx, txt_path in enumerate(txt_files, start=1):
        stem = txt_path.stem
        print(f"\n=== [{idx}/{len(txt_files)}] Processing: {txt_path.name} ===")

        # 1) Convert text -> JSON
        data = build_json_from_text(txt_path, args.goal_mode)
        json_out = output_dir / f"{stem}.json"
        json_out.write_text(json.dumps(data, indent=2))
        print(f"Wrote JSON -> {json_out}")

        # 2) Validate with VAL
        summary_out = output_dir / f"{stem}_val_summary.json"
        val_summary = process_with_val(summary_out, data, Path(args.domain), Path(args.problem) if args.problem else None, Path(args.val))
        print(f"Wrote VAL summary -> {summary_out}")

        # 3) Initial state check via PyDatalog
        init_txt = output_dir / f"{stem}_initial_state.txt"
        sat_count, total_plans = check_initial_state(json_out, init_txt)
        print(f"Wrote Initial State grounding -> {init_txt}")

        # 4) Write per-file quick stats
        try:
            val_results = (val_summary or {}).get('results', []) if isinstance(val_summary, dict) else []
            val_total = len(val_results)
            val_invalid = sum(1 for r in val_results if not r.get('valid', False))
            val_valid = val_total - val_invalid
            stats = {
                'file': txt_path.name,
                'initial_state_satisfied': sat_count,
                'initial_state_total': total_plans,
                'val_valid': val_valid,
                'val_invalid': val_invalid,
                'val_total': val_total,
                'json_path': str(json_out),
                'val_summary_path': str(summary_out),
                'initial_state_txt_path': str(init_txt),
            }
            # Print concise stats summary to console
            print(
                f"Stats [{txt_path.name}]: INIT {sat_count}/{total_plans} | "
                f"VAL valid={val_valid}, invalid={val_invalid}, total={val_total}"
            )
            stats_path = output_dir / f"{stem}_stats.json"
            stats_path.write_text(json.dumps(stats, indent=2))
            print(f"Wrote stats -> {stats_path}")
        except Exception as e:
            print(f"Warning: could not write stats for {txt_path.name}: {e}")

    print(f"\nDone. Outputs written under: {output_dir}")


if __name__ == '__main__':
    main()
