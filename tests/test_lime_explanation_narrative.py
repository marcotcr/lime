import pytest

from lime.lime_text import LimeTextExplainer, summarize_lime_explanation, generate_sentence_for_feature


class DummyClassifier:
    def __init__(self, vocab_pos=None):
        # words that will increase class 1 probability
        self.vocab_pos = set(vocab_pos or [])

    def predict_proba(self, texts):
        # Two-class probabilities based on presence of positive words
        probs = []
        for t in texts:
            score = sum(word in t for word in self.vocab_pos)
            # simple mapping to [0,1]
            p1 = min(0.9, 0.1 + 0.2 * score)
            p0 = 1.0 - p1
            probs.append([p0, p1])
        return probs


def test_explain_instance_plain_text_summary():
    clf = DummyClassifier(vocab_pos={"great", "excellent", "awesome"})
    explainer = LimeTextExplainer(class_names=["neg", "pos"]) 

    text = "This movie was great, with excellent pacing but some flaws."
    exp = explainer.explain_instance(text_instance=text, classifier_fn=clf.predict_proba, labels=(1,), num_features=5, num_samples=100)

    summary = explainer.explain_instance_plain_text(exp, label=1, n_words=3)
    assert isinstance(summary, str)
    # Ensure key words appear
    assert "great" in summary or "excellent" in summary
    assert "pos" in summary


def test_summarize_lime_explanation_sentences():
    clf = DummyClassifier(vocab_pos={"great", "excellent"})
    explainer = LimeTextExplainer(class_names=["neg", "pos"]) 
    text = "great excellent average"
    exp = explainer.explain_instance(text_instance=text, classifier_fn=clf.predict_proba, labels=(1,), num_features=5, num_samples=100)

    sentences = summarize_lime_explanation(exp, class_idx=1)
    assert isinstance(sentences, list)
    assert len(sentences) >= 1
    # first sentence is overview
    assert sentences[0].startswith("Overall,")
    # subsequent sentences reference words
    assert any("great" in s or "excellent" in s for s in sentences[1:])


def test_generate_sentence_for_feature():
    s = generate_sentence_for_feature("great", 0.12, "pos")
    assert "strongly increased" in s
    s2 = generate_sentence_for_feature("bad", -0.03, "neg")
    assert "slightly decreased" in s2
