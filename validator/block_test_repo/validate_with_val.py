#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import sys
from typing import List, Tuple

# -------------------- Defaults & Config --------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Default inputs (relative to repo structure); adjust as needed
DEFAULT_JSON = os.path.join(SCRIPT_DIR, 'example_out_pddl_regression2.json')
# DEFAULT_JSON = os.path.join(SCRIPT_DIR, 'txt_converted_regression.json')
DEFAULT_DOMAIN = os.path.normpath(os.path.join(SCRIPT_DIR, '../nl_pddl_planner/tests_user/block_world_test/planbench/generated_domain.pddl'))
DEFAULT_PROBLEM = os.path.normpath(os.path.join(SCRIPT_DIR, '../nl_pddl_planner/tests_user/block_world_test/planbench/instance-10.pddl'))
DEFAULT_VAL_BIN = os.path.normpath(os.path.join(SCRIPT_DIR, 'val/bin/Validate'))
DEFAULT_OUT = os.path.join(SCRIPT_DIR, 'subgoal_validation_summary.json')

CONFIG_FILENAME = os.path.join(SCRIPT_DIR, 'validate_with_val.config.json')


def read_file(path: str) -> str:
    with open(path, 'r') as f:
        return f.read()


def write_file(path: str, content: str) -> None:
    with open(path, 'w') as f:
        f.write(content)


def extract_problem_sections(problem_text: str) -> Tuple[str, str, str]:
    """
    Roughly split a PDDL problem file into three sections:
    - prefix (everything before :init)
    - goal (the full :goal form including parentheses)
    - suffix (everything after :goal)
    We'll replace only the :init section by simple regex substitutions.
    """
    # Extract goal first
    goal_match = re.search(r"\(:goal\s*(\([^)]*(?:\([^)]*\)[^)]*)*\))\s*\)", problem_text, re.DOTALL)
    goal_text = goal_match.group(0) if goal_match else ""
    # Split around :init
    init_pattern = re.compile(r"\(:init\s*(.*?)\)", re.DOTALL)
    init_match = init_pattern.search(problem_text)
    if not init_match:
        raise RuntimeError("Could not find (:init ...) section in problem file")

    init_start, init_end = init_match.span()
    prefix = problem_text[:init_start]
    suffix = problem_text[init_end:]
    return prefix, goal_text, suffix


def build_init_section(predicates: List[str]) -> str:
    # Ensure one predicate per line for readability
    preds_text = "\n  ".join(predicates)
    return f"(:init\n  {preds_text}\n)"


def build_goal_section(predicates: List[str]) -> str:
    # Always wrap in (and ... ) for simplicity
    if not predicates:
        return "(:goal (and))"
    preds_text = "\n    ".join(predicates)
    return f"(:goal\n  (and\n    {preds_text}\n  )\n)"


def actions_to_val_plan(actions: List[str]) -> str:
    """Convert actions like 'unstack(b, c)' into VAL plan lines '(unstack b c)'."""
    lines = []
    for a in actions:
        a = a.strip()
        m = re.match(r"^([a-zA-Z0-9_-]+)\((.*)\)$", a)
        if not m:
            # fallback: if already in VAL format
            if a.startswith('(') and a.endswith(')'):
                lines.append(a)
            else:
                # try to wrap
                lines.append(f"({a})")
            continue
        name = m.group(1)
        args_str = m.group(2).strip()
        args = [x.strip() for x in args_str.split(',') if x.strip()]
        # normalize variables -> constants by removing '?' and lowercasing
        args = [re.sub(r"^\?", "", t).lower() for t in args]
        if args:
            lines.append(f"({name} {' '.join(args)})")
        else:
            lines.append(f"({name})")
    return "\n".join(lines) + "\n"


def normalize_predicates(preds: List[str]) -> List[str]:
    norm = []
    for p in preds:
        # Expect '(<pred> <args...>)'
        s = p.strip()
        if not (s.startswith('(') and s.endswith(')')):
            # attempt to coerce: e.g., 'clear a' -> '(clear a)'
            parts = s.split()
            if parts:
                s = '(' + ' '.join(parts) + ')'
        # remove '?' and lowercase tokens inside
        def repl(m):
            tok = m.group(0)
            tok = tok.replace('?', '').lower()
            return tok
        inner = s[1:-1]
        inner = re.sub(r"[A-Za-z0-9_?\-]+", repl, inner)
        norm.append('(' + inner + ')')
    return norm


def validate_with_val(val_bin: str, domain_path: str, problem_path: str, plan_path: str) -> Tuple[bool, str]:
    cmd = [val_bin, '-v', domain_path, problem_path, plan_path]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        out = proc.stdout
        success = ('Plan valid' in out) or ('Successful plans:' in out)
        return success, out
    except Exception as e:
        return False, f"Exception running VAL: {e}"


def extract_objects_from_prefix(prefix: str) -> Tuple[str, List[str], str]:
    """Extract the :objects section from the problem prefix, returning
    (prefix_before_objects, objects_list, prefix_after_objects_up_to_init).
    If no objects section is found, returns (prefix, [], '')."""
    # Match (:objects ... ) greedily until ) on same line or next lines before (:init
    m = re.search(r"\(:objects\s*([^)]*)\)\s*", prefix, re.DOTALL)
    if not m:
        return prefix, [], ''
    start, end = m.span()
    before = prefix[:start]
    after = prefix[end:]
    objs_raw = m.group(1)
    # Split by whitespace
    objs = [tok for tok in re.split(r"\s+", objs_raw.strip()) if tok]
    return before, objs, after


def format_objects_section(objs: List[str]) -> str:
    return "(:objects " + " ".join(objs) + " )\n"


def collect_symbols_from_text(text: str) -> List[str]:
    # collect identifiers; exclude predicate/logical keywords
    candidates = re.findall(r"[A-Za-z0-9_\-]+", text)
    exclude = {"and", "init", "goal", "define", "domain", "objects",
               "clear", "ontable", "handempty", "holding", "on"}
    # Filter out purely numeric tokens and keywords
    syms = []
    for t in candidates:
        if t.lower() in exclude:
            continue
        if re.fullmatch(r"\d+", t):
            continue
        syms.append(t)
    return syms


def main():
    ap = argparse.ArgumentParser(description='Validate subgoal actions against overall goal using VAL')
    ap.add_argument('--json', default=DEFAULT_JSON, help='Path to example_out_pddl_regression.json')
    ap.add_argument('--domain', default=DEFAULT_DOMAIN, help='Path to domain.pddl')
    ap.add_argument('--problem', default=DEFAULT_PROBLEM, help='Path to base problem.pddl (goal taken from here)')
    ap.add_argument('--val', default=DEFAULT_VAL_BIN, help='Path to VAL Validate binary')
    ap.add_argument('--out', default=DEFAULT_OUT, help='Summary output JSON path')
    args = ap.parse_args()

    # Optional config override: if a config JSON exists next to this script, load it
    # and use any provided fields to override defaults/CLI values.
    if os.path.exists(CONFIG_FILENAME):
        try:
            cfg = json.load(open(CONFIG_FILENAME))
            # Only override if keys present
            args.json = cfg.get('json', args.json)
            args.domain = cfg.get('domain', args.domain)
            args.problem = cfg.get('problem', args.problem)
            args.val = cfg.get('val', args.val)
            args.out = cfg.get('out', args.out)
        except Exception as e:
            print(f"Warning: failed to read config {CONFIG_FILENAME}: {e}")

    data = json.load(open(args.json))
    plans = data.get('plans', [])

    # plans = plans[:10]

    # Load base problem and split
    base_problem_text = read_file(args.problem)

    # Extract prefix and problem goal (for fallback only)
    prefix, problem_goal_text, suffix = extract_problem_sections(base_problem_text)

    # Build goal from JSON goal_predicates if provided
    json_goal_preds_raw = data.get('goal_predicates', []) or []
    if json_goal_preds_raw:
        goal_norm = normalize_predicates(json_goal_preds_raw)
        goal_text = build_goal_section(goal_norm)
    else:
        goal_text = problem_goal_text


    # if not goal_text:
    #     # try to extract goal from problem if not matched by extract, fallback naive
    #     goal_match = re.search(r"\(:goal[\s\S]*\)\s*\)", base_problem_text)
    #     if goal_match:
    #         goal_text = goal_match.group(0)
    #     else:
    #         raise RuntimeError('Could not find a (:goal ...) section in base problem')

    results = []
    failures_log: List[str] = []
    workdir = tempfile.mkdtemp(prefix='val_subgoal_')

    try:
        total = len(plans)
        def show_progress(i: int, total: int, width: int = 30):
            filled = int((i / total) * width) if total else width
            bar = '#' * filled + '-' * (width - filled)
            sys.stdout.write(f"\rValidating subgoals: [{bar}] {i}/{total}")
            sys.stdout.flush()

        show_progress(0, total)
        for idx, item in enumerate(plans):
            sub_preds = item.get('subgoal_predicates', []) or []
            actions = item.get('action', []) or []
            substitution = item.get('substitution', {}) or {}
            # Raw fields from JSON (if provided)
            subgoal_raw = item.get('subgoal_raw', {}) or {}
            actions_raw = item.get('action_raw', []) or []

            # Normalize predicates: remove '?' and lowercase tokens
            init_preds = normalize_predicates(sub_preds)
            init_text = build_init_section(init_preds)

            # Build temp problem with augmented :objects so that all symbols in init/goal appear as objects
            before_obj, existing_objs, after_obj = extract_objects_from_prefix(prefix)
            needed_syms = set(collect_symbols_from_text(init_text) + collect_symbols_from_text(goal_text))
            merged_objs = sorted(set(existing_objs) | needed_syms)
            new_prefix = before_obj + format_objects_section(merged_objs) + after_obj
            temp_problem_text = new_prefix + init_text + "\n" + goal_text
            # Safety: ensure header '(define ...)' and '(:domain ...)' exist
            if "(define" not in temp_problem_text or "(:domain" not in temp_problem_text:
                # Extract header from base problem up to first (:objects or (:init
                m_head = re.search(r"^\(define[\s\S]*?\n\(:domain[\s\S]*?\)\n", base_problem_text)
                if m_head:
                    header = m_head.group(0)
                else:
                    # Fallback: take everything before first (:init)
                    header = base_problem_text.split("(:init", 1)[0]
                temp_problem_text = header + temp_problem_text

            # Ensure the overall problem ends with the final ')' to close (define ...)
            opens = temp_problem_text.count('(')
            closes = temp_problem_text.count(')')
            if closes < opens:
                temp_problem_text = temp_problem_text.rstrip() + "\n" + (')' * (opens - closes)) + "\n"

            # Write temp files
            problem_path = os.path.join(workdir, f'problem_{idx}.pddl')
            plan_path = os.path.join(workdir, f'plan_{idx}.plan')
            write_file(problem_path, temp_problem_text)

            plan_text = actions_to_val_plan(actions)
            write_file(plan_path, plan_text)

            ok, log = validate_with_val(args.val, args.domain, problem_path, plan_path)
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
                # Print failure details only
                sys.stdout.write("\n\n[INVALID] Subgoal #{}\n".format(idx))
                # Raw sections first (if any)
                try:
                    if isinstance(subgoal_raw, dict):
                        raw_clauses = subgoal_raw.get('disjoint') or subgoal_raw.get('disjoint_1', []) or []
                    else:
                        raw_clauses = []
                except Exception:
                    raw_clauses = []
                if raw_clauses:
                    sys.stdout.write("subgoal_raw (clauses):\n  " + "\n  ".join(raw_clauses) + "\n")
                if actions_raw:
                    sys.stdout.write("actions_raw:\n  " + "\n  ".join(actions_raw) + "\n")
                sys.stdout.write("init_predicates:\n  " + "\n  ".join(init_preds) + "\n")
                sys.stdout.write("actions:\n  " + "\n  ".join(actions) + "\n")
                # Print substitution mapping nicely
                if substitution:
                    # normalize mapping to strings
                    try:
                        subs_lines = [f"{k}: {v}" for k, v in substitution.items()]
                    except Exception:
                        subs_lines = [str(substitution)]
                    sys.stdout.write("substitution:\n  " + "\n  ".join(subs_lines) + "\n")
                sys.stdout.write("plan_val:\n  " + "\n  ".join(plan_text.strip().splitlines()) + "\n")
                sys.stdout.write("val_output:\n" + log + "\n")
                sys.stdout.flush()

                # Also record to failures log buffer
                block_lines = []
                block_lines.append(f"[INVALID] Subgoal #{idx}")
                # Raw sections
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
                    block_lines.append("substitution:")
                    block_lines.extend(["  " + line for line in subs_lines])
                block_lines.append("plan_val:")
                block_lines.extend(["  " + ln for ln in plan_text.strip().splitlines()])
                block_lines.append("val_output:")
                block_lines.append(log)
                block_lines.append("")
                failures_log.append("\n".join(block_lines))
            show_progress(idx + 1, total)

        sys.stdout.write("\n")

        # Save failures to a text file for later inspection
        try:
            failures_path = os.path.join(SCRIPT_DIR, 'validation_failures.txt')
            with open(failures_path, 'w') as f:
                f.write("\n\n".join(failures_log))
            # Optional minimal notice
            if failures_log:
                print(f"Saved failed validations -> {failures_path}")
        except Exception as e:
            print(f"Warning: could not write validation_failures.txt: {e}")

    finally:
        # Keep workdir for debugging? Uncomment to remove
        # shutil.rmtree(workdir, ignore_errors=True)
        pass

    failed = sum(1 for r in results if not r.get('valid', False))
    total = len(results)

    summary = {
        'domain': os.path.abspath(args.domain),
        'base_problem': os.path.abspath(args.problem),
        'json': os.path.abspath(args.json),
        'val_bin': os.path.abspath(args.val) if not args.val.startswith('./') else args.val,
        'results': results,
        'failed': failed,
        'total': total,
    }
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=2)
    # Print concise final summary only
    print(f"Failed {failed}/{total} subgoal plans")


if __name__ == '__main__':
    main()
