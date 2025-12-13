"""
Utility functions for LIME.

This module previously contained Japanese-specific tokenizers. That
functionality has been moved to `lime.japanese`. This file now provides a
small shim to maintain backward compatibility for callers importing
`lime.utils.mecab_unidic_split`.
"""

import warnings

warnings.warn(
    "The `lime.utils` module is deprecated for Japanese tokenizer utilities. "
    "Please import from `lime.japanese` instead (e.g. `from lime.japanese import mecab_unidic_split`). "
    "`lime.utils` shim will be removed in a future release.",
    DeprecationWarning,
)

# Backwards-compatibility shim: import mecab_unidic_split from the
# new `lime.japanese` package if available.
try:
    from .japanese import mecab_unidic_split  # type: ignore
except Exception:
    def mecab_unidic_split(*args, **kwargs):
        raise ImportError("Install fugashi[unidic-lite] to use the Japanese splitter: pip install 'fugashi[unidic-lite]'")

__all__ = ["mecab_unidic_split"]
