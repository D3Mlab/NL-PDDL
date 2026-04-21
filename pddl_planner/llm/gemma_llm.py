"""Local Google Gemma LLM backend with an OpenAI-compatible chat interface.

This module provides :class:`GemmaLLM`, a drop-in alternative to
``openai.OpenAI`` for the rest of :mod:`pddl_planner.llm.llm`. It loads an
instruction-tuned Gemma checkpoint locally through Hugging Face Transformers
with CUDA support when a GPU is available.

The object exposes ``client.with_options(timeout=...).chat.completions.create(
model=..., messages=[...])`` and returns a response whose shape matches
``openai.types.chat.ChatCompletion`` enough for downstream code:
``response.choices[0].message.content``.

Selecting the model size
------------------------
``GemmaLLM(model_size="4b")`` is the default. Any of the canonical Gemma
instruction-tuned sizes are accepted:

    * ``"1b"``  — fastest, text-only, fits comfortably on one consumer GPU.
    * ``"4b"``  — default, balanced quality/VRAM.
    * ``"12b"`` — higher quality, needs a high-VRAM GPU.
    * ``"27b"`` — largest, multi-GPU / 48GB+ recommended.

The Gemma *generation* ("3" by default) can be overridden via the
``generation`` kwarg, or a full Hugging Face repo id (e.g.
``"google/gemma-3-4b-it"``) can be passed directly as ``model_size``.

Gemma checkpoints on Hugging Face are gated. Provide an access token via the
``hf_token`` kwarg, or set ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` in your
environment before instantiation.
"""

from __future__ import annotations

import logging
import os
import threading
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger("pddl_planner.llm.gemma")


# Canonical Gemma instruction-tuned checkpoint sizes.
_GEMMA_SIZES: Dict[str, str] = {
    "1b": "1b",
    "4b": "4b",
    "12b": "12b",
    "27b": "27b",
}
_DEFAULT_GENERATION = "3"


def available_sizes() -> List[str]:
    """Return the canonical Gemma sizes accepted by :class:`GemmaLLM`."""
    return sorted(_GEMMA_SIZES)


def _resolve_repo_id(model_size: str, generation: str = _DEFAULT_GENERATION) -> str:
    """Map a user-friendly size token to a Hugging Face repo id.

    Accepts:
        * Full repo id — e.g. ``"google/gemma-3-4b-it"`` (returned unchanged).
        * Generation-qualified name — e.g. ``"gemma-3-4b"`` or
          ``"gemma-3-4b-it"``.
        * Bare size token — e.g. ``"4b"``; combined with ``generation``.

    Args:
        model_size: User-supplied size selector.
        generation: Gemma generation to assume when only a size is given.

    Returns:
        A fully qualified Hugging Face repo id for the instruction-tuned
        (``-it``) variant.
    """
    if not isinstance(model_size, str) or not model_size.strip():
        raise ValueError("model_size must be a non-empty string.")
    if "/" in model_size:
        return model_size
    s = model_size.strip().lower().removesuffix("-it")
    if s.startswith("gemma-"):
        parts = s.split("-")
        if len(parts) >= 3:
            gen, size = parts[1], parts[2]
            if size in _GEMMA_SIZES:
                return f"google/gemma-{gen}-{_GEMMA_SIZES[size]}-it"
    if s in _GEMMA_SIZES:
        return f"google/gemma-{generation}-{_GEMMA_SIZES[s]}-it"
    raise ValueError(
        f"Unsupported Gemma model selector {model_size!r}. "
        f"Pass one of {available_sizes()} (optionally prefixed with "
        f"\"gemma-<generation>-\"), or a full Hugging Face repo id like "
        f"\"google/gemma-3-4b-it\"."
    )


class GemmaLLM:
    """Local Gemma inference client with an OpenAI-compatible chat surface.

    The object mirrors the tiny slice of the OpenAI client used by
    :class:`pddl_planner.llm.llm.LLM`: ``with_options`` (no-op here) and
    ``chat.completions.create``. Generation runs fully locally on the
    configured Torch device; CUDA is used automatically when available.

    Attributes:
        model_size (str): User-supplied size selector.
        generation (str): Gemma generation token (e.g. ``"3"``).
        repo_id (str): Resolved Hugging Face repo id.
        device (str): Torch device string (``"cuda"``, ``"cuda:0"``, ``"cpu"``).
        max_new_tokens (int): Default cap on generated tokens per call.
        temperature (float): Default sampling temperature.
        top_p (float): Default nucleus-sampling parameter.
    """

    # Transformers' generate() is not re-entrant across threads on the same
    # model object. Serialize concurrent requests within a single process.
    _generate_lock = threading.Lock()

    def __init__(
        self,
        model_size: str = "4b",
        *,
        generation: str = _DEFAULT_GENERATION,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        hf_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        """Load the tokenizer and model weights.

        Args:
            model_size: ``"1b"``, ``"4b"``, ``"12b"``, ``"27b"``, or a full
                Hugging Face repo id. Defaults to ``"4b"``.
            generation: Gemma generation (``"3"`` by default). Ignored when
                ``model_size`` is already a full repo id.
            device: Torch device string. Defaults to ``"cuda"`` when a GPU is
                visible, else ``"cpu"``.
            dtype: Torch dtype name (``"bfloat16"``, ``"float16"``,
                ``"float32"``). Defaults to ``"bfloat16"`` on CUDA and
                ``"float32"`` on CPU.
            max_new_tokens: Default generation cap.
            temperature: Default sampling temperature.
            top_p: Default nucleus-sampling parameter.
            hf_token: Hugging Face access token for gated Gemma weights.
                Falls back to ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` in
                the environment.
            cache_dir: Optional Hugging Face cache directory override.
        """
        # Heavy imports are deferred so `import pddl_planner.llm.llm` does not
        # pull in torch/transformers for users staying on the OpenAI backend.
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError as exc:
            raise ImportError(
                "GemmaLLM requires `torch` and `transformers`. "
                "Install with: pip install torch transformers accelerate"
            ) from exc

        # For multimodal-capable Gemma variants (4B/12B/27B) we prefer the
        # image-text-to-text class and feed it text-only. Fall back silently
        # when the class is unavailable (older transformers).
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError:
            AutoModelForImageTextToText = None  # type: ignore

        self.model_size = model_size
        self.generation = generation
        self.repo_id = _resolve_repo_id(model_size, generation)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        if dtype is None:
            dtype_str = "bfloat16" if device.startswith("cuda") else "float32"
        else:
            dtype_str = dtype
        try:
            torch_dtype = getattr(torch, dtype_str)
        except AttributeError as exc:
            raise ValueError(
                f"Unknown torch dtype {dtype_str!r}. Expected one of "
                f"'bfloat16', 'float16', 'float32'."
            ) from exc
        self._torch = torch

        token = (
            hf_token
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGING_FACE_HUB_TOKEN")
        )

        logger.info(
            "Loading Gemma model %s on %s (dtype=%s)", self.repo_id, self.device, dtype_str
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.repo_id, token=token, cache_dir=cache_dir
        )

        model_kwargs: Dict[str, Any] = dict(
            torch_dtype=torch_dtype,
            token=token,
            cache_dir=cache_dir,
        )
        if self.device.startswith("cuda"):
            model_kwargs["device_map"] = self.device

        # Text-only checkpoints (1B) load through AutoModelForCausalLM;
        # multimodal ones (4B+) register under AutoModelForImageTextToText.
        try:
            self._model = AutoModelForCausalLM.from_pretrained(self.repo_id, **model_kwargs)
        except (ValueError, KeyError) as exc:
            if AutoModelForImageTextToText is None:
                raise
            logger.debug(
                "AutoModelForCausalLM rejected %s (%s); retrying with "
                "AutoModelForImageTextToText", self.repo_id, exc,
            )
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.repo_id, **model_kwargs
            )

        if not self.device.startswith("cuda"):
            self._model.to(self.device)
        self._model.eval()

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
        """Render the chat template, generate, and wrap in an OpenAI shape."""
        prompt = self._apply_chat_template(list(messages))
        completion = self._generate(
            prompt,
            temperature=self.temperature if temperature is None else temperature,
            top_p=self.top_p if top_p is None else top_p,
            max_new_tokens=self.max_new_tokens if max_tokens is None else max_tokens,
        )
        return _make_openai_response(completion, self.repo_id)

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """Render OpenAI-style messages through Gemma's chat template.

        Gemma tokenizers do not expose a dedicated ``system`` role; any
        system-role content is folded into the first user turn.
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

        return self._tokenizer.apply_chat_template(
            normalized,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _generate(
        self,
        prompt: str,
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> str:
        torch = self._torch
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[-1]
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
            "temperature": max(float(temperature), 1e-5),
            "top_p": float(top_p),
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        with self._generate_lock, torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)
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
