"""Prompt template loader for LLM modules.

Templates are plain-text files in this directory with ``{placeholder}`` fields
that are filled via ``str.format_map()``.
"""

from pathlib import Path
from functools import lru_cache

_PROMPT_DIR = Path(__file__).parent


@lru_cache(maxsize=None)
def _read_template(name: str) -> str:
    """Read and cache a prompt template file by name (without extension)."""
    path = _PROMPT_DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8")


def load_prompt(name: str, **kwargs: str) -> str:
    """Load a prompt template and fill in placeholders.

    Args:
        name: Template filename without ``.txt`` extension
              (e.g. ``"entailment"``, ``"type_entailment"``).
        **kwargs: Values for ``{placeholder}`` fields in the template.

    Returns:
        The formatted prompt string.
    """
    return _read_template(name).format_map(kwargs)
