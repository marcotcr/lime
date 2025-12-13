import pytest
import numpy as np


def _is_installed(pkg_name):
    try:
        __import__(pkg_name)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _is_installed('fugashi'), reason="fugashi not installed")
def test_mecab_unidic_splitter_tokens():
    from lime import utils
    s = "私は学生です。"
    tokens = utils.mecab_unidic_split(s)
    assert isinstance(tokens, list)
    assert all(isinstance(t, str) for t in tokens)
    assert len(tokens) >= 2


@pytest.mark.skip(reason="janome removed; tests require fugashi")
def test_janome_splitter_tokens():
    pass


@pytest.mark.skipif(not _is_installed('fugashi'), reason="fugashi not installed")
def test_explainer_integration_prefers_available_splitter():
    from lime import utils
    from lime.lime_text import LimeTextExplainer

    explainer = LimeTextExplainer(lang='jp')

    # split_expression must be the fugashi splitter
    available = [getattr(utils, 'mecab_unidic_split', None)]
    assert explainer.split_expression in available

    # quick integration run with a trivial classifier
    def clf(texts):
        return np.array([[0.6, 0.4]] * len(texts))

    exp = explainer.explain_instance("今日はいい天気です。", clf, labels=(0,), num_features=2, num_samples=20)
    assert hasattr(exp, 'as_list')
    lst = exp.as_list(0)
    assert isinstance(lst, list)
