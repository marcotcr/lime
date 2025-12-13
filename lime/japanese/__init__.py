"""Japanese-specific text processing utilities for LIME.

Export a small, stable API and keep implementation details in separate
modules. Users can import `lime.japanese` when they need Japanese-specific
functionality.
"""

from .splitters import split as mecab_unidic_split, active_japanese_tokenizer  # type: ignore

__all__ = ["mecab_unidic_split", "active_japanese_tokenizer"]
