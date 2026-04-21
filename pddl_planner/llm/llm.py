# Class to interact with the LLM
import openai
from pddl_planner.pddl_core.action import Action
from pddl_planner.logic.formula import Substitution
from pddl_planner.logic.nl_formula import NLPredicate
from pddl_planner.logic.operation import Operations
from typing import List, Dict, Tuple, Optional
import json
import copy
import time
import random
import itertools
import os
import logging

logger = logging.getLogger("pddl_planner.llm")

class LLM:
    """
    A class to intilize and interact with the LLM for various task in the regression planner.

    Two backends are supported:
        * ``"openai"`` (default) — the hosted OpenAI chat completions API.
        * ``"gemma"`` — a locally hosted Google Gemma instruction-tuned model via
          :class:`pddl_planner.llm.gemma_llm.GemmaLLM`, which transparently exposes
          an OpenAI-compatible ``chat.completions.create`` surface so the rest of
          this class does not need to branch on backend.

    Backend selection is either explicit (``backend="gemma"``) or auto-detected
    from ``model_name`` (anything starting with ``"gemma"`` or
    ``"google/gemma"`` routes to the local Gemma backend).
    """

    def __init__(self, model_name: str, api_key: str|None = None, cache_path: str|None ='cache.json', verbose: bool = True,
                  backend: str|None = None, backend_options: dict|None = None):
        """
        Initialize the LLM.

        Args:
            model_name (str): The name of the model to use. For the Gemma
                backend this is either a size token (``"1b"``, ``"4b"``,
                ``"12b"``, ``"27b"``), a generation-qualified name
                (``"gemma-3-4b"``), or a full Hugging Face repo id
                (``"google/gemma-3-4b-it"``).
            api_key (str|None): The API key for the OpenAI backend. Ignored
                when the Gemma backend is selected.
            cache_path (str|None): The path to the cache file.
            verbose (bool): Passthrough verbosity flag for call-site logging.
            backend (str|None): ``"openai"`` or ``"gemma"``. When ``None``,
                the backend is inferred from ``model_name``.
            backend_options (dict|None): Extra keyword arguments forwarded to
                the selected backend's constructor. For ``"gemma"`` this
                accepts e.g. ``device``, ``dtype``, ``max_new_tokens``,
                ``hf_token``, ``cache_dir`` — see
                :class:`pddl_planner.llm.gemma_llm.GemmaLLM`.
        """
        self.model_name = model_name
        self._api_key = api_key
        self._cache_path = cache_path
        self._cache = self._load_cache()
        self.backend = (backend or self._detect_backend(model_name)).lower()
        if self.backend not in {"openai", "gemma"}:
            raise ValueError(
                f"Unsupported LLM backend {backend!r}. Expected 'openai' or 'gemma'."
            )
        backend_options = dict(backend_options or {})
        if self.backend == "gemma":
            # Lazy import so users on the OpenAI backend don't pay the
            # torch/transformers import cost.
            from pddl_planner.llm.gemma_llm import GemmaLLM
            logger.info("Initializing local Gemma backend for model %r", model_name)
            self.client = GemmaLLM(model_size=model_name, **backend_options)
        else:
            self.client = openai.OpenAI(api_key=api_key, **backend_options)
        self._n_iter = 5 # number of iterations for the entailment check for self consistency check
        # Local Gemma backend returns deterministic-ish output already; running
        # five samples per check would waste compute for no real vote gain.
        if self.backend == "gemma":
            self._n_iter = 1
        self._operations = Operations()
        self._verbose = verbose
        # Track total entailment checks (including cache-served)
        self.call_count = 0
        # Track sample-level counts
        self.cache_call_count = 0  # number of cached responses consumed
        self.api_call_count = 0    # number of API sample queries (excludes per-attempt retries)
        # Track API latency (successful request durations only)
        self.api_latency_sum = 0.0
        self.api_latency_count = 0

    @staticmethod
    def _detect_backend(model_name: str) -> str:
        """Infer the backend from a model name, defaulting to OpenAI.

        Any identifier starting with ``"gemma"`` (e.g. ``"gemma-3-4b"``) or
        containing ``"/gemma"`` (e.g. ``"google/gemma-3-4b-it"``,
        ``"unsloth/gemma-3-4b-it"``) routes to the local Gemma backend.
        """
        if not isinstance(model_name, str):
            return "openai"
        n = model_name.strip().lower()
        if n.startswith("gemma") or "/gemma" in n:
            return "gemma"
        return "openai"

    def entailment(self, predicate: NLPredicate, predicates: List[NLPredicate], background_predicates: Tuple[Action, List[NLPredicate]] = (None, []),
                    domain_predicates: bool = False, flag = True) -> NLPredicate|None:
        """
        Determine whether a target NL predicate is entailed by any predicate schema in a list.

        Args:
            predicate (NLPredicate): The target predicate to check for entailment.
            predicates (List[NLPredicate]): Candidate predicate schemas to test against.
            background_predicates (Tuple[Action, List[NLPredicate]]): Optional (Action, background NL predicates)
                context used to enrich the LLM prompt.

        Returns:
            NLPredicate | None: The input predicate annotated with its entailed schema if found
                (or list of candidates if multiple), otherwise None.
        """
        # update current cache
        self._cache = self._load_cache()
        logger.info('Checking entailment for "%s"', predicate.nl_description)
        entailed_preds = []
        for pred in predicates:
            # Create deep copies to prevent modifications to original objects
            predicate_copy = copy.deepcopy(predicate)
            pred_copy = copy.deepcopy(pred)

            # Find proper substitution between the target predicate and the current predicate
            # Use unify_with_different_name for entailment tasks to allow different predicate names
            substitution = self._operations.unify_with_different_name(pred_copy, predicate_copy,  Substitution())
            if substitution is None:
                continue
            logger.debug('Substitution: %s between "%s" and "%s"', substitution, str(predicate_copy), str(pred_copy))

            # Build and try all permutations of value assignments if multiple variables are present
            keys = list(substitution.keys())
            values = list(substitution.values())
            permuted_values_list = [tuple(values)]
            if len(values) > 1:
                # Deduplicate permutations in case of repeated values
                permuted_values_list = list({tuple(p) for p in itertools.permutations(values, len(values))})
                logger.debug("Trying %d substitution permutations for keys %s", len(permuted_values_list), keys)

            entailed_for_this_pred = False
            winning_perm_sub = None
            for perm_vals in permuted_values_list:
                logger.debug('Permutation: %s for keys %s', perm_vals, keys)
                perm_sub = Substitution({k: v for k, v in zip(keys, perm_vals)})

                # Apply substitution on fresh copies to avoid cross-permutation side effects
                perm_target = copy.deepcopy(predicate_copy)
                perm_pred = copy.deepcopy(pred_copy).substitute(perm_sub)

                # Conduct entailment between the substituted string representations
                target_str = perm_target.nl_description
                pred_str = perm_pred.nl_description

                if predicate_copy._is_neg:
                    # reverse the entailment check for negative predicates
                    entailment_result, response_text = self._entailment_check(
                        pred_str,
                        target_str,
                        background_predicates,
                        target_predicate_name=pred_copy.name,
                        flag=flag,
                    )
                else:
                    # conduct entailment check
                    entailment_result, response_text = self._entailment_check(
                        target_str,
                        pred_str,
                        background_predicates,
                        target_predicate_name=predicate_copy.name,
                        flag=flag,
                    )

                if entailment_result:
                    entailed_for_this_pred = True
                    winning_perm_sub = perm_sub
                    break

            if entailed_for_this_pred:
                logger.info('Entailment found: "%s" -> "%s"', str(predicate), pred.name)
                entailed_preds.append(perm_pred)
                # Record the substitution used for entailment for later SSA alignment
                try:
                    predicate.add_entailed_substitution(pred.name, winning_perm_sub if winning_perm_sub is not None else Substitution())
                except Exception:
                    pass
        if len(entailed_preds) == 1:
            # if there is only one entailed predicate, overwrite the original predicate's entailment with the entailed predicate
            predicate.entailed = entailed_preds[0]
            return predicate
        elif len(entailed_preds) > 1:
            # if there are multiple entailed predicates, return the list of entailed predicates
            predicate.entailed = entailed_preds
            return predicate
        else:
            # if there are no entailed predicates, return None
            logger.warning('No entailment found for "%s" against %d candidates', predicate.nl_description, len(predicates))
            return None

    def _entailment_check(self, target_str: str, pred_str: str, background_predicates: Tuple[Action, List[NLPredicate]] = (None, []),
target_predicate_name: Optional[str] = None, flag: bool = True) -> Tuple[bool, str]:
        """
        Check if the target description is entailed by the candidate description, with caching and self-consistency.

        Args:
            target_str (str): The target predicate string representation.
            pred_str (str): The candidate predicate string representation.
            background_predicates (Tuple[Action, List[NLPredicate]]): The background predicates and action.
            target_predicate_name (Optional[str]): The name of the target predicate.

        Returns:
            Tuple[bool, str]: (decision, representative_raw_text)
        """

        # Count every entailment check (regardless of cache hit/miss)
        if flag: self.call_count += 1

        # Check cache first, then complete to n_iter with LLM calls and decide by self-consistency
        # 1) Parse cached responses (if any), then complete to n_iter using LLM, then decide
        cached_texts = self._get_cached_llm_responses(target_str, pred_str) or []
        normal_results: List[Tuple[Optional[bool], str]] = []
        # Parse existing cached responses (up to n_iter)
        used_cached = cached_texts[: self._n_iter]
        # count cached samples used
        if flag: self.cache_call_count += 1
        for t in used_cached:
            decision, _ = self._parse_yes_no_response(t)
            normal_results.append((decision, t))
        if used_cached:
            logger.debug("Cache hit: %d/%d samples for \"%s\" vs \"%s\"", len(used_cached), self._n_iter, target_str, pred_str)
        # If we have fewer than n_iter cached, complete by querying LLM and updating cache
        missing = max(0, self._n_iter - len(cached_texts))
        last_text = ""
        if flag and missing > 0: self.api_call_count += 1
        if missing > 0:
            logger.debug("Querying LLM for %d additional samples", missing)
        for _ in range(missing):
            # count one API sample query (regardless of internal retries)
            decision, text = self._get_llm_responses(target_str, pred_str, background_predicates, target_predicate_name=target_predicate_name)
            if text is not None:
                self._update_cache_llm_response(target_str, pred_str, text, predicate_name=target_predicate_name)
                last_text = text or last_text
            normal_results.append((decision, text or ""))
        votes = [result[0] for result in normal_results]
        yes_count = sum(1 for v in votes if v is True)
        no_count = sum(1 for v in votes if v is False)
        logger.info('Entailment vote: "%s" entailed by "%s"? %d/%d yes (%s)', target_str, pred_str, yes_count, len(votes), votes)
        majority_decision, majority_text = self._self_consistent_decision(normal_results)
        if majority_decision is not None:
            return bool(majority_decision), (majority_text or last_text)

        # Default to False if still ambiguous
        return False, last_text

    def _self_consistent_decision(self, results: List[Tuple[Optional[bool], str]]) -> Tuple[Optional[bool], str]:
        """
        Given a list of parsed (decision, text) tuples, return the majority decision and
        a representative text. Does not perform cache or LLM calls.

        Args:
            results (List[Tuple[Optional[bool], str]]): The list of parsed (decision, text) tuples.

        Returns:
            Tuple[Optional[bool], str]: The majority decision and a representative text.
        """
        yes_count = 0
        no_count = 0
        text_yes: Optional[str] = None
        text_no: Optional[str] = None
        last_text = ""
        for decision, text in results:
            if text:
                last_text = text
            if decision is None:
                continue
            if decision:
                yes_count += 1
                if text:
                    text_yes = text
            else:
                no_count += 1
                if text:
                    text_no = text
        if yes_count > no_count:
            return True, (text_yes or last_text)
        if no_count > yes_count:
            return False, (text_no or last_text)
        return None, last_text

    def _get_llm_responses(self, target_str: str, pred_str: str, background_predicates: Tuple[Action, List[NLPredicate]] = (None, []),
                            max_retries: int = 3, timeout: float = 30.0, target_predicate_name: Optional[str] = None) -> Tuple[Optional[bool], str]:
        """
        Build the entailment prompt and call the chat API with retries.
        Returns (decision, raw_text).

        Args:
            target_str (str): The target predicate string representation.
            pred_str (str): The candidate predicate string representation.
            background_predicates (Tuple[Action, List[NLPredicate]]): The background predicates and action.
            max_retries (int): The maximum number of retries.
            timeout (float): The timeout for the LLM call.

        Returns:
            Tuple[Optional[bool], str]: (decision, raw_text) where decision can be True/False/None on parse failure.
        """

        prompt = self._build_entailment_prompt(
            target_str,
            pred_str,
            background_predicates=background_predicates,
            include_action=False,
            include_background_predicates=True,
            include_examples=False,
        )
        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                client = self.client.with_options(timeout=timeout)
                t0 = time.time()
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                )
                latency = time.time() - t0
                try:
                    self.api_latency_sum += latency
                    self.api_latency_count += 1
                except Exception:
                    self.api_latency_sum = latency
                    self.api_latency_count = 1
                logger.debug("API call completed in %.3fs (model=%s)", latency, self.model_name)
                return self._parse_yes_no_response(response.choices[0].message.content.strip())
            except Exception as err:
                last_error = err
                wait_seconds = (2 ** attempt) + random.uniform(0, 0.5)
                logger.warning("API call failed (attempt %d/%d): %s — retrying in %.2fs", attempt + 1, max_retries, err, wait_seconds)
                time.sleep(wait_seconds)
        logger.error("API call failed after %d attempts: %s", max_retries, last_error)
        return None, ""

    def _build_entailment_prompt(
        self,
        target_str: str,
        pred_str: str,
        background_predicates: Tuple[Action, List[NLPredicate]] = (None, []),
        include_action: bool = True,
        include_background_predicates: bool = True,
        include_examples: bool = True,
    ) -> str:
        """
        Build an entailment prompt with optional sections.

        Args:
            target_str (str): NL text for Predicate 1 (target).
            pred_str (str): NL text for Predicate 2 (candidate).
            background_predicates (Tuple[Action, List[NLPredicate]]): (Action, [NLPredicate]) context.
            include_action (bool): Whether to include action preconditions/effects.
            include_background_predicates (bool): Whether to include background predicate list.
            include_examples (bool): Whether to include example entailments.

        Returns:
            str: The full prompt string.
        """
        from pddl_planner.llm.prompts import load_prompt

        action_description = ""
        if include_action and background_predicates and background_predicates[0] is not None:
            action = background_predicates[0]
            action_description = (
                f"{action.name}\n"
                f"with the following preconditions: {[clause.nl_description for clause in action.preconditions.clauses if isinstance(clause, NLPredicate)]}\n"
                f"and the following effects: {[clause.nl_description for clause in action.effects.clauses if isinstance(clause, NLPredicate)]}"
            )

        background_predicates_str = ""
        if include_background_predicates and background_predicates and len(background_predicates) > 1:
            background_predicates_str = "\n ".join([f"- {pred.nl_description}" for pred in background_predicates[1]
            if pred.nl_description != pred_str and pred.nl_description != target_str])

        examples_block = ""
        if include_examples:
            examples_block = load_prompt("entailment_examples")

        has_action = bool(action_description)
        has_bg = bool(background_predicates_str)

        prompt = load_prompt(
            "entailment",
            target_str=target_str,
            pred_str=pred_str,
            action_description=action_description,
            role_suffix=" that currently doing the following action:" if has_action else ":",
            question_suffix=" when doing the action" if has_action else "",
            background_instruction=(
                "2.You know following background to determine the sepcific information of the objects within Predicate 1 and Predicate 2."
                if has_bg else ""
            ),
            background_predicates_str=background_predicates_str,
            step_consider="3." if include_background_predicates else "2.",
            consider_suffix=" in the context of the action" if has_action else " in common contexts",
            step_creative="4." if include_background_predicates else "3.",
            examples_block=examples_block,
        )
        return prompt

    def _get_cached_llm_responses(self, target_str: str, candidate_pred_nl: str) -> Optional[List[str]]:
        """
        Retrieve cached raw LLM response texts (list) for the given NL pair if available.

        Args:
            target_str (str): The target predicate string representation.
            candidate_pred_nl (str): The candidate predicate string representation.

        Returns:
            Optional[List[str]]: The cached raw LLM response texts if present; otherwise None.
        """
        # load the current cache to up to date version
        self._cache = self._load_cache()
        if target_str in self._cache and isinstance(self._cache[target_str], dict):
            val = self._cache[target_str].get(candidate_pred_nl)
            if isinstance(val, list):
                return val
            if isinstance(val, str):
                # Backward-compat: single string stored before switch to list
                return [val]
        return None

    def _load_cache(self) -> Dict[str, str]:
        """
        Load the cache from the file.

        Returns:
            Dict[str, str]: The cache of previous entailments.
        """
        if self._cache_path is not None:
            # if the cache path is provided, load the cache from the file
            try:
                with open(self._cache_path, 'r') as f:
                    self._cache = json.load(f)
            except FileNotFoundError:
                self._cache = {}
                self._cache_path = 'cache.json'
                self._save_cache()
        else:
            # if the cache path is not provided, create a new cache
            self._cache = {}
            self._cache_path = 'cache.json'
            self._save_cache()
        return self._cache


    def _load_cache_entailment(self, predicate: NLPredicate, predicates: List[NLPredicate]) -> Tuple[bool, NLPredicate|List[NLPredicate]|None]:
        """
        (Legacy) Cache the entailment of the predicate by the predicates.
        """
        #check if cache is loaded
        if self._cache is not None:
            # check if the predicate string representation is in the cache of previous entailments
            predicate_str = predicate.nl_description
            if predicate_str in self._cache:
                logger.debug('Cache hit for predicate "%s"', predicate_str)
                cached_value = self._cache[predicate_str]
                # New schema: dict of pred_name -> raw_response_text
                if isinstance(cached_value, dict):
                    entailed_preds: List[NLPredicate] = []
                    for pred in predicates:
                        if pred.name in cached_value:
                            decision, _ = self._parse_yes_no_response(cached_value[pred.name])
                            if decision is True:
                                entailed_preds.append(copy.deepcopy(pred))
                    if len(entailed_preds) == 1:
                        return True, entailed_preds[0]
                    if len(entailed_preds) > 1:
                        return True, entailed_preds
                    # We have cached responses but none entailed
                    return True, None
                # Backward compatibility: old schema
                entailed_pred_name = cached_value
                if entailed_pred_name is None:
                    return True, None
                if isinstance(entailed_pred_name, list):
                    entailed_preds = []
                    for entailed_pred in entailed_pred_name:
                        for pred in predicates:
                            if pred.name == entailed_pred:
                                entailed_preds.append(copy.deepcopy(pred))
                    return True, entailed_preds
                else:
                    for pred in predicates:
                        if pred.name == entailed_pred_name:
                            return True, copy.deepcopy(pred)
        # if the cache is not loaded, return False and None
        return False, None

    def _update_cache_llm_response(self, target_str: str, candidate_pred_nl: str, response_text: str, predicate_name: Optional[str] = None) -> None:
        """
        Update cache with raw LLM response for a specific target and candidate predicate pair.
        Cache schema: cache[target_str]["predicate_name"] = str (optional);
        cache[target_str][candidate_pred_nl] = List[str]

        Args:
            target_str (str): The target predicate string representation.
            candidate_pred_nl (str): The candidate predicate string representation.
            response_text (str): The raw LLM response text.
            predicate_name (Optional[str]): The name of the target predicate for reference.
        """
        # Initialize mapping for target_str if absent or not a dict
        if target_str not in self._cache or not isinstance(self._cache[target_str], dict):
            self._cache[target_str] = {}
        # Store target predicate name for future reference
        if predicate_name is not None:
            self._cache[target_str]["predicate_name"] = predicate_name
        if candidate_pred_nl not in self._cache[target_str] or not isinstance(self._cache[target_str][candidate_pred_nl], list):
            self._cache[target_str][candidate_pred_nl] = []
        self._cache[target_str][candidate_pred_nl].append(response_text)
        self._save_cache()


    def _parse_yes_no_response(self, text: str) -> Tuple[Optional[bool], str]:
        """
        Parse a chain-of-thought style response and extract a YES/NO decision.

        Strategy:
        - Prefer the last explicit 'Response:' line if present.
        - Then check the last non-empty sentence/line.
        - Then check the first non-empty sentence/line.
        - Finally, fallback to whole-text heuristic if unambiguous.

        Returns (decision, original_text) where decision is True/False or None if undecidable.
        """
        if text is None:
            return None, ""
        original_text = text
        normalized_all = text.strip()
        if not normalized_all:
            return None, original_text

        def to_upper_clean(s: str) -> str:
            return s.strip().upper().strip(":,.!;()[]{}\n\t ")

        lines = [ln for ln in (ln.strip() for ln in normalized_all.splitlines()) if ln]

        # 1) Prefer explicit 'Response:' lines
        for ln in reversed(lines):
            if ln.upper().startswith("RESPONSE:"):
                answer_raw = ln.split(":", 1)[1] if ":" in ln else ln[9:]
                answer = to_upper_clean(answer_raw)
                if answer.startswith("YES"):
                    return True, original_text
                if answer.startswith("NO"):
                    return False, original_text

        # Helper to split into sentences conservatively
        def split_sentences(block: str) -> List[str]:
            parts: List[str] = []
            buf = ""
            for ch in block:
                buf += ch
                if ch in ".!?\n":
                    if buf.strip():
                        parts.append(buf.strip())
                    buf = ""
            if buf.strip():
                parts.append(buf.strip())
            return parts

        sentences = split_sentences(normalized_all)
        last_sentence = to_upper_clean(sentences[-1]) if sentences else ""
        first_sentence = to_upper_clean(sentences[0]) if sentences else ""

        # 2) Check last sentence
        if last_sentence.startswith("YES"):
            return True, original_text
        if last_sentence.startswith("NO"):
            return False, original_text

        # 3) Check first sentence
        if first_sentence.startswith("YES"):
            return True, original_text
        if first_sentence.startswith("NO"):
            return False, original_text

        # 4) Fallback heuristic on full text only if unambiguous
        upper_all = normalized_all.upper()
        has_yes = "YES" in upper_all
        has_no = "NO" in upper_all
        if has_yes and not has_no:
            return True, original_text
        if has_no and not has_yes:
            return False, original_text

        return None, original_text

    def _save_cache(self) -> None:
        """
        Save the cache to the file based on the provided cache path.
        """
        with open(self._cache_path, 'w') as f:
            json.dump(self._cache, f, indent=2)

    def replace_predicate_name(self, target_predicate: NLPredicate, entailed_predicate: NLPredicate) -> NLPredicate:
        """
        Replace the name of the target predicate with the name of the entailed predicate,
        while keeping all terms the same as the original target predicate.

        Args:
            target_predicate (NLPredicate): The target predicate whose name will be replaced.
            entailed_predicate (NLPredicate): The entailed predicate whose name will be used.

        Returns:
            NLPredicate: A new predicate with the entailed predicate's name but target predicate's terms.
        """
        # Create a deep copy of the target predicate to avoid modifying the original
        target_copy = copy.deepcopy(target_predicate)
        entailed_predicate_copy = copy.deepcopy(entailed_predicate)
        # Get the original string representation
        original_str_rep = str(target_copy)

        # Replace the target predicate name with the entailed predicate name in the string representation
        # This handles cases where the name might appear multiple times or in different contexts
        updated_str_rep = original_str_rep.replace(target_copy.name, entailed_predicate.name)

        #perform substitution on the entailed_predicate
        substitution = self._operations.unify_with_different_name(target_copy, entailed_predicate_copy, Substitution())
        if substitution is not None:
            entailed_predicate_copy = entailed_predicate_copy.substitute(substitution)

        # Create a new NLPredicate with the entailed predicate's name but target predicate's terms
        new_predicate = NLPredicate(
            entailed_predicate_copy.name,
            updated_str_rep,
            target_copy._is_neg,
            *target_copy.terms,
            term_type_dict=entailed_predicate_copy.term_type_dict,
            entailed_by=entailed_predicate_copy
        )

        return new_predicate
