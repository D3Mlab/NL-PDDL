"""Local Google Gemma LLM backend with an OpenAI-compatible chat interface.

This module provides :class:`GemmaLLM`, a drop-in alternative to
``openai.OpenAI`` for the rest of :mod:`pddl_planner.llm.llm`. It loads an
instruction-tuned Gemma checkpoint locally through **Unsloth's**
:class:`unsloth.FastLanguageModel`, which wraps the Hugging Face stack with
4-bit (bitsandbytes) quantization, fused kernels, and a faster inference
path — materially less VRAM and latency than the vanilla
``AutoModelForCausalLM`` loader we used previously.

The object exposes ``client.with_options(timeout=...).chat.completions.create(
model=..., messages=[...])`` and returns a response whose shape matches
``openai.types.chat.ChatCompletion`` enough for downstream code:
``response.choices[0].message.content``.

Selecting the model
-------------------
``GemmaLLM(model_size="e4b")`` is the default and resolves to
``"unsloth/gemma-4-e4b-it-unsloth-bnb-4bit"`` — the 4-bit Efficient-4B
Gemma-4 instruction checkpoint. Accepted selectors:

    * ``"e2b"``, ``"e4b"``           — Gemma 4 Efficient tier (Unsloth 4-bit).
    * ``"1b"``, ``"4b"``, ``"12b"``, ``"27b"`` — Gemma 3 sizes (Unsloth fp16).
    * Full repo id, e.g. ``"unsloth/gemma-3-4b-it"`` — used verbatim.

Gemma checkpoints on Hugging Face are gated. Provide an access token via the
``hf_token`` kwarg, or set ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` in your
environment before instantiation. Unsloth's mirrors are ungated and the
default path does not require a token.
"""

from __future__ import annotations

import logging
import os
import threading
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger("pddl_planner.llm.gemma")


# Canonical size tokens → Unsloth repo ids. Gemma 4 uses the Efficient-tier
# "E2B"/"E4B" naming and ships pre-quantized 4-bit variants; Gemma 3 keeps
# the parameter-count sizing and is loaded in fp16/bf16.
_UNSLOTH_GEMMA4_E: Dict[str, str] = {
    "e2b": "unsloth/gemma-4-e2b-it-unsloth-bnb-4bit",
    "e4b": "unsloth/gemma-4-e4b-it-unsloth-bnb-4bit",
}
_UNSLOTH_GEMMA3: Dict[str, str] = {
    "1b": "unsloth/gemma-3-1b-it",
    "4b": "unsloth/gemma-3-4b-it",
    "12b": "unsloth/gemma-3-12b-it",
    "27b": "unsloth/gemma-3-27b-it",
}


def available_sizes() -> List[str]:
    """Return the canonical Gemma sizes accepted by :class:`GemmaLLM`."""
    return sorted(list(_UNSLOTH_GEMMA4_E) + list(_UNSLOTH_GEMMA3))


def _resolve_repo_id(model_size: str) -> str:
    """Map a user-friendly size token to a Hugging Face repo id.

    Full repo ids (anything containing ``"/"``) are returned verbatim. Size
    tokens are resolved to Unsloth mirrors — Gemma 4 E-series maps to the
    4-bit bitsandbytes checkpoints; Gemma 3 sizes map to the fp16 ones.
    """
    if not isinstance(model_size, str) or not model_size.strip():
        raise ValueError("model_size must be a non-empty string.")
    if "/" in model_size:
        return model_size
    s = model_size.strip().lower().removesuffix("-it")
    if s in _UNSLOTH_GEMMA4_E:
        return _UNSLOTH_GEMMA4_E[s]
    if s in _UNSLOTH_GEMMA3:
        return _UNSLOTH_GEMMA3[s]
    # Accept "gemma-3-4b" / "gemma-4-e4b" style prefixes too.
    if s.startswith("gemma-"):
        parts = s.split("-")
        if len(parts) >= 3:
            _, _, size = parts[0], parts[1], "-".join(parts[2:])
            if size in _UNSLOTH_GEMMA4_E:
                return _UNSLOTH_GEMMA4_E[size]
            if size in _UNSLOTH_GEMMA3:
                return _UNSLOTH_GEMMA3[size]
    raise ValueError(
        f"Unsupported Gemma model selector {model_size!r}. "
        f"Pass one of {available_sizes()}, or a full Hugging Face repo id "
        f"like \"unsloth/gemma-4-e4b-it-unsloth-bnb-4bit\"."
    )


class GemmaLLM:
    """Local Gemma inference client with an OpenAI-compatible chat surface.

    Loads weights through :class:`unsloth.FastLanguageModel`, which enables
    4-bit quantization (``load_in_4bit=True``) and fused inference kernels.
    The object mirrors the tiny slice of the OpenAI client used by
    :class:`pddl_planner.llm.llm.LLM`: ``with_options`` (no-op here) and
    ``chat.completions.create``. Generation runs fully locally on CUDA.

    Attributes:
        model_size (str): User-supplied size selector.
        repo_id (str): Resolved Hugging Face repo id actually loaded.
        device (str): Torch device string used for inputs (``"cuda"`` /
            ``"cpu"``). Unsloth places the model on CUDA automatically when
            ``load_in_4bit`` is enabled.
        max_seq_length (int): Maximum sequence length (caps the KV cache).
        max_new_tokens (int): Default generation cap per call.
        temperature (float): Default sampling temperature.
        top_p (float): Default nucleus-sampling parameter.
    """

    # Unsloth's generate() is not safely re-entrant across threads on the same
    # model object. Serialize concurrent requests within a single process.
    _generate_lock = threading.Lock()

    def __init__(
        self,
        model_size: str = "e4b",
        *,
        max_seq_length: int = 8192,
        load_in_4bit: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        hf_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """Load the tokenizer and model weights via Unsloth.

        Args:
            model_size: Size token (``"e2b"``, ``"e4b"``, ``"1b"``,
                ``"4b"``, ``"12b"``, ``"27b"``) or a full Hugging Face repo
                id. Defaults to ``"e4b"`` (Gemma 4 Efficient-4B, 4-bit).
            max_seq_length: Cap on context length — also caps the KV cache
                size, so keep it modest to avoid OOM on small GPUs.
                Defaults to 8192.
            load_in_4bit: When ``True`` (default) Unsloth loads the
                bitsandbytes 4-bit quantized weights. Turn off to load
                fp16/bf16 if you have the VRAM.
            max_new_tokens: Default per-call generation cap.
            temperature: Default sampling temperature.
            top_p: Default nucleus-sampling parameter.
            hf_token: Hugging Face access token for gated checkpoints.
                Falls back to ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` in
                the environment. Unsloth mirrors are ungated; a token is
                only needed when you pass a ``google/...`` repo id.
            cache_dir: Optional Hugging Face cache directory override.
            device: Torch device for the input tensors. Defaults to
                ``"cuda"`` when a GPU is visible. Unsloth itself manages
                weight placement; this only affects where we move inputs
                before generate().
        """
        # Heavy imports are deferred so `import pddl_planner.llm.llm` does
        # not pull in torch/unsloth for users staying on the OpenAI backend.
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "GemmaLLM requires `torch`. Install a CUDA-enabled torch "
                "build, then: pip install unsloth"
            ) from exc
        try:
            from unsloth import FastLanguageModel
        except ImportError as exc:
            raise ImportError(
                "GemmaLLM now loads Gemma via Unsloth's FastLanguageModel. "
                "Install with: `pip install unsloth` (Colab/Linux + CUDA)."
            ) from exc

        self.model_size = model_size
        self.repo_id = _resolve_repo_id(model_size)
        self.max_seq_length = int(max_seq_length)
        self.load_in_4bit = bool(load_in_4bit)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self._torch = torch

        # Where we move the input token tensor before generate(). Unsloth
        # puts the model on CUDA internally when load_in_4bit is True.
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        token = (
            hf_token
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGING_FACE_HUB_TOKEN")
        )

        logger.info(
            "Loading Gemma via Unsloth: %s (load_in_4bit=%s, max_seq_length=%d)",
            self.repo_id, self.load_in_4bit, self.max_seq_length,
        )
        try:
            self._model, self._tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.repo_id,
                max_seq_length=self.max_seq_length,
                load_in_4bit=self.load_in_4bit,
                token=token,
                cache_dir=cache_dir,
            )
        except ValueError as exc:
            # Keep the actionable tokenizer hint for the rare case where
            # sentencepiece is missing; otherwise re-raise unchanged.
            msg = str(exc)
            if "sentencepiece" in msg or "slow tokenizer" in msg:
                raise ImportError(
                    "Loading the Gemma tokenizer requires `sentencepiece`. "
                    "Install it with `pip install sentencepiece` and "
                    "restart your Python/kernel session."
                ) from exc
            raise

        # Switch to Unsloth's fast inference path (enables KV-cache tricks and
        # kernel fusions that speed up generate() by ~2x).
        FastLanguageModel.for_inference(self._model)

        # OpenAI-style access path: ``client.chat.completions.create(...)``.
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._chat_create)
        )

    # ------------------------------ API ------------------------------ #

    def with_options(self, timeout: Optional[float] = None, **_: Any) -> "GemmaLLM":
        """OpenAI-client compatibility shim — local inference ignores timeout."""
        return self

    # ---------------------------- internal ---------------------------- #

    def _chat_create(
        self,
        *,
        model: Optional[str] = None,  # accepted for API parity; ignored
        messages: Sequence[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        **_: Any,
    ) -> Any:
        """Apply the chat template, generate, and wrap in an OpenAI shape."""
        completion = self._generate(
            list(messages),
            temperature=self.temperature if temperature is None else temperature,
            top_p=self.top_p if top_p is None else top_p,
            max_new_tokens=self.max_new_tokens if max_tokens is None else max_tokens,
        )
        return _make_openai_response(completion, self.repo_id)

    def _normalize_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Collapse system-role messages into the first user turn.

        Gemma's chat template has no dedicated ``system`` role, so any
        system-role content is prepended to the first user message.
        """
        normalized: List[Dict[str, str]] = []
        system_chunks: List[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = str(m.get("content", ""))
            if role == "system":
                if content:
                    system_chunks.append(content)
                continue
            if role == "user" and system_chunks:
                content = "\n".join(system_chunks) + "\n\n" + content
                system_chunks = []
            normalized.append({"role": role, "content": content})
        if system_chunks and not normalized:
            normalized.append({"role": "user", "content": "\n".join(system_chunks)})
        return normalized

    def _generate(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> str:
        torch = self._torch
        normalized = self._normalize_messages(messages)
        # apply_chat_template with tokenize=True returns a token-id tensor
        # directly, matching the Unsloth quickstart pattern.
        input_ids = self._tokenizer.apply_chat_template(
            normalized,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)

        input_len = input_ids.shape[-1]
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": temperature > 0.0,
            "temperature": max(float(temperature), 1e-5),
            "top_p": float(top_p),
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        with self._generate_lock, torch.no_grad():
            output_ids = self._model.generate(input_ids=input_ids, **gen_kwargs)
        new_tokens = output_ids[0, input_len:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip()


def _make_openai_response(content: str, model: str) -> Any:
    """Wrap a raw completion string in an OpenAI ``ChatCompletion``-like shape.

    The rest of the codebase only reads ``response.choices[0].message.content``,
    so a lightweight :class:`~types.SimpleNamespace` is sufficient and avoids a
    hard dependency on the OpenAI types module.
    """
    message = SimpleNamespace(role="assistant", content=content)
    choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
    return SimpleNamespace(model=model, choices=[choice])
