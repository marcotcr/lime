"""Singleton tokenizer instances for Japanese processing.

Use Sudachi (Python bindings) to avoid external MeCab/UniDic setup.
The tokenizer is instantiated once at import for performance.
"""

try:
    # SudachiPy API (compatible with sudachi.rs python bindings)
    from sudachipy import tokenizer as _sudachi_tokenizer  # type: ignore
    from sudachipy import dictionary as _sudachi_dictionary  # type: ignore

    _SUDACHI_TOKENIZER = _sudachi_dictionary.Dictionary().create()
    _SUDACHI_MODE = _sudachi_tokenizer.Tokenizer.SplitMode.C
except Exception:
    _SUDACHI_TOKENIZER = None
    _SUDACHI_MODE = None


def has_sudachi():
    """Return True if the Sudachi tokenizer was successfully initialized."""
    return _SUDACHI_TOKENIZER is not None


__all__ = ["_SUDACHI_TOKENIZER", "_SUDACHI_MODE", "has_sudachi"]
