"""Tests for Sudachi-backed Japanese tokenization.

These tests are skipped if `sudachipy` is not installed so CI can run
without the optional dependency.
"""
import pytest


def _is_installed(pkg):
    try:
        __import__(pkg)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _is_installed('sudachipy'), reason='sudachipy not installed')
def test_active_backend_is_sudachi():
    from lime.japanese import active_japanese_tokenizer
    assert active_japanese_tokenizer() == 'sudachi'


@pytest.mark.skipif(not _is_installed('sudachipy'), reason='sudachipy not installed')
def test_sudachi_splitter_returns_tokens():
    from lime.japanese import mecab_unidic_split
    s = "今日はいい天気です。"
    tokens = mecab_unidic_split(s)
    assert isinstance(tokens, list)
    assert all(isinstance(t, str) for t in tokens)
    assert len(tokens) >= 2
