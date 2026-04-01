#!/usr/bin/env python3

import argparse
import json
import os
from typing import Optional, List

from initial_state_checker import (
    BlockWorldInitialStateChecker,
)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Check block-world initial state against subgoals using PyDatalog",
    )
    parser.add_argument(
        "--input",
        "-i",
        # default="txt_converted_regression2.json",
        default="example_out_pddl_regression2.json",
        help="Path to input JSON file (e.g., example_out_pddl_regression.json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="",
        help="Optional path to save JSON results",
    )
    parser.add_argument(
        "--print-mode",
        choices=["all", "subgoals_if_conj"],
        default="subgoals_if_conj",
        help=(
            "Control subgoal printing: 'all' prints all subgoals; "
            "'subgoals_if_conj' prints subgoals only when conjunction is satisfied"
        ),
    )
    args = parser.parse_args(argv)

    # Prepare output capture
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_TXT = os.path.join(SCRIPT_DIR, 'initial_state_subgoal_grounding.txt')
    out_lines = []
    def emit(s: str = ""):
        print(s)
        out_lines.append(s)

    checker = BlockWorldInitialStateChecker(input_path=args.input)
    checker.load_json()
    checker.load_initial_facts()
    results = checker.evaluate_subgoals()

    # Print only plans where the conjunction is satisfied
    for plan_res in results["plans_evaluation"]:
        # Determine conjunction satisfaction first; skip others
        conj = plan_res.get("conjunction", {}) or {}
        conj_sat = bool(conj.get("satisfied", False))
        if not conj_sat:
            continue

        idx = plan_res["plan_index"]

        # Plan separator/header for readability
        emit("\n====================================")
        emit(f"Plan {idx}")
        emit("------------------------------------")

        # Subgoal predicates (from evaluated subgoals)
        try:
            conj_preds = [sg.get("predicate", "") for sg in plan_res.get("subgoals", [])]
            if conj_preds:
                emit("Subgoal Predicates:")
                for p in conj_preds:
                    emit(f"  - {p}")
                emit("")
        except Exception:
            pass

        # Disjoint (raw clauses) from original plan input
        try:
            raw_plan = (checker.data.get("plans", []) or [])[idx]
            raw_disj = {}
            if isinstance(raw_plan, dict):
                raw_disj = raw_plan.get("subgoal_raw", {}) or {}
            clauses = []
            if isinstance(raw_disj, dict):
                clauses = raw_disj.get("disjoint") or raw_disj.get("disjoint_1", []) or []
            if clauses:
                emit("Disjoint (raw clauses):")
                for c in clauses:
                    emit(f"  - {c}")
                emit("")
        except Exception:
            pass

        # Actions
        actions = plan_res.get("action", [])
        if actions:
            emit("Actions:")
            for act in actions:
                emit(f"  - {act}")
            emit("")

        # Conjunction formula and details
        try:
            if plan_res.get("subgoals"):
                emit("Conjunction:")
                emit("  " + " & ".join([sg.get("predicate", "") for sg in plan_res["subgoals"]]))
                emit("")
        except Exception:
            pass

        # Conjunction details (satisfied will be True here)
        emit(f"Conjunction satisfied: {conj.get('satisfied', False)}")
        vars_ = conj.get("variables") or []
        binds = conj.get("bindings")
        if vars_ and binds:
            emit(f"Conjunction bindings (variables: {vars_}):")
            for row in binds:
                emit(f"  - {row}")
            # Also print the subgoals with variable bindings substituted
            try:
                base_preds = [sg.get("predicate", "") for sg in plan_res.get("subgoals", [])]
                if base_preds:
                    emit("")
                    emit("Subgoals with bindings applied:")
                    for b_idx, row in enumerate(binds, start=1):
                        mapping = {vars_[i]: row[i] for i in range(min(len(vars_), len(row)))}
                        # Header for this binding
                        assigns = ", ".join(f"{vars_[i]}={row[i]}" for i in range(min(len(vars_), len(row))))
                        emit(f"  Binding {b_idx}: {assigns}")
                        # Subgoals under this binding
                        for p in base_preds:
                            s = p.strip()
                            new_pred = p
                            if s.startswith("(") and s.endswith(")"):
                                inner_tokens = s[1:-1].strip().split()
                                if inner_tokens:
                                    name = inner_tokens[0]
                                    pred_args = inner_tokens[1:]
                                    new_args = [mapping.get(tok, tok) for tok in pred_args]
                                    new_pred = f"({name}{(' ' + ' '.join(new_args)) if new_args else ''})"
                            emit(f"    - {new_pred}")
                    emit("")
            except Exception:
                pass
            print("")

    # No overall summary when restricting output to requested sections

    # Persist captured output to a text file
    try:
        with open(OUTPUT_TXT, 'w') as f:
            f.write("\n".join(out_lines) + "\n")
        # Also support optional JSON save as before
        if args.output:
            checker.save_results(args.output)
        print(f"Saved initial state subgoal grounding -> {OUTPUT_TXT}")
    except Exception as e:
        print(f"Warning: could not write {OUTPUT_TXT}: {e}")


if __name__ == "__main__":
    main()
