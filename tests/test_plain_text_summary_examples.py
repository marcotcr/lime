"""Simple test demonstrating the plain-English summary helper for
LimeTextExplainer.

This test avoids any tokenizer complexity by using the default English
splitter and a trivial classifier that returns fixed probabilities.
The goal is to show how to call `explain_instance` and
`explain_instance_plain_text` without long try/except blocks.
"""

import numpy as np
from lime.lime_text import LimeTextExplainer


def test_explain_instance_plain_text_generates_sentence():
    explainer = LimeTextExplainer()

    # Trivial classifier that returns two-class probabilities for any input
    def clf(texts):
        return np.array([[0.7, 0.3]] * len(texts))

    text = "This movie was great and funny, I loved the acting and the story."
    exp = explainer.explain_instance(text, clf, labels=(0,), num_features=3, num_samples=100)

    summary = explainer.explain_instance_plain_text(exp, label=0, n_words=3)

    # Basic sanity checks
    assert isinstance(summary, str)
    assert summary.startswith("In this text")
    assert '"' in summary  # quoted words should appear
