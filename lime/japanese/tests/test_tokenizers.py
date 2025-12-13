"""Tests for the `lime.japanese` package tokenizers and splitters.

These tests are skipped if `fugashi` is not installed so the repository
can be tested on systems without the dependency.
"""

import pytest


def _is_installed(pkg):
    try:
        __import__(pkg)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _is_installed('fugashi'), reason='fugashi not installed')
def test_mecab_unidic_singleton_and_splitter():
    from lime.japanese import tokenizers, splitters

    assert tokenizers.has_fugashi()

    text = "私は学生です。"
    toks = splitters.mecab_unidic_split(text)
    assert isinstance(toks, list)
    assert len(toks) >= 2
    assert all(isinstance(t, str) for t in toks)


@pytest.mark.skipif(not _is_installed('fugashi'), reason='fugashi not installed')
def test_utils_shim_works():
    # ensure old shim continues to work
    from lime import utils
    from lime.japanese import splitters

    assert getattr(utils, 'mecab_unidic_split', None) is not None

    text = "今日はいい天気です。"
    assert splitters.mecab_unidic_split(text) == utils.mecab_unidic_split(text)
