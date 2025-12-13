"""Package shim exposing mecab_unidic_split from lime.utils module."""

try:
    from .utils import mecab_unidic_split  # type: ignore
except Exception:
    def mecab_unidic_split(*args, **kwargs):
        raise ImportError("Install fugashi[unidic-lite] to use mecab_unidic_split: pip install 'fugashi[unidic-lite]'")

__all__ = ["mecab_unidic_split"]