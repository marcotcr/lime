import pytest

from lime.lime_text import LimeTextExplainer

# Try importing Japanese splitter backend to decide skip
try:
    from lime.japanese.splitters import active_japanese_tokenizer
    HAS_JP = active_japanese_tokenizer() != 'fallback'
except Exception:
    HAS_JP = False

# Removed skip to always run the test


class DummyClassifierJP:
    def predict_proba(self, texts):
        # Very simple: boost class 1 if text contains '良い' or '最高'
        probs = []
        for t in texts:
            score = 0
            for kw in ("良い", "最高"):
                if kw in t:
                    score += 1
            p1 = min(0.9, 0.1 + 0.3 * score)
            p0 = 1.0 - p1
            probs.append([p0, p1])
        return probs


def test_explain_instance_plain_text_summary_jp():
    clf = DummyClassifierJP()
    explainer = LimeTextExplainer(class_names=["neg", "pos"], lang="jp")

    text = "この映画は最高に良い。音楽も良い。"
    exp = explainer.explain_instance(text_instance=text, classifier_fn=clf.predict_proba, labels=(1,), num_features=5, num_samples=200)

    summary = explainer.explain_instance_plain_text(exp, label=1, n_words=5)
    assert isinstance(summary, str)
    # Summary should include some Japanese tokens if Sudachi works, or key characters in fallback
    assert any(ch in summary for ch in ["良", "最", "映"])  # accept partial tokens
    assert "pos" in summary

    # Also verify Japanese narrative helpers
    from lime.lime_text import summarize_lime_explanation_jp, generate_sentence_for_feature_jp
    jp_sentences = summarize_lime_explanation_jp(exp, class_idx=1)
    assert isinstance(jp_sentences, list)
    assert len(jp_sentences) >= 1
    # Overview now uses the new template
    assert jp_sentences[0].startswith("このインスタンスは")
    # Per-feature sentence contains Japanese markers
    if len(jp_sentences) > 1:
        assert any(tok in jp_sentences[1] for tok in ["言葉", "重み"])  # language markers

    # Unit test for single-feature Japanese sentence
    s_jp = generate_sentence_for_feature_jp("良い", 0.12, "pos")
    assert "大きく上げました" in s_jp or "上げました" in s_jp
