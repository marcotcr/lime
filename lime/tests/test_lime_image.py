import numpy as np
from scipy.special import binom

from lime.lime_base import LimeBase
from lime.lime_image import LimeImageExplainer


def test_standard_sampling_returns_no_weight_adjustments():
    explainer = LimeImageExplainer(random_state=42)

    image = np.zeros((4, 4, 3), dtype=np.float32)
    fudged_image = np.zeros_like(image)

    segments = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 3, 3],
        [2, 2, 3, 3],
    ])

    def classifier_fn(images):
        return np.tile(np.array([[0.4, 0.6]]), (len(images), 1))

    data, labels, weight_adjustments = explainer.data_labels(
        image=image,
        fudged_image=fudged_image,
        segments=segments,
        classifier_fn=classifier_fn,
        num_samples=20,
        batch_size=5,
        use_stratification=False,
        progress_bar=False,
    )

    assert data.shape == (20, 4)
    assert labels.shape == (20, 2)
    assert weight_adjustments is None
    assert np.all(data[0] == 1)


def test_stratified_sampling_basic_properties():
    explainer = LimeImageExplainer(random_state=42)

    image = np.zeros((4, 4, 3), dtype=np.float32)
    fudged_image = np.zeros_like(image)

    segments = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 3, 3],
        [2, 2, 3, 3],
    ])

    def classifier_fn(images):
        return np.tile(np.array([[0.4, 0.6]]), (len(images), 1))

    data, labels, weight_adjustments = explainer.data_labels(
        image=image,
        fudged_image=fudged_image,
        segments=segments,
        classifier_fn=classifier_fn,
        num_samples=20,
        batch_size=5,
        use_stratification=True,
        progress_bar=False,
    )

    assert data.shape == (20, 4)
    assert labels.shape == (20, 2)
    assert weight_adjustments.shape == (20,)

    assert np.all(data[0] == 1)
    assert np.all(np.isin(data, [0, 1]))

    assert np.all(np.isfinite(weight_adjustments))
    assert weight_adjustments[0] == 1


def test_stratified_sampling_is_reproducible():
    image = np.zeros((4, 4, 3), dtype=np.float32)
    fudged_image = np.zeros_like(image)

    segments = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 3, 3],
        [2, 2, 3, 3],
    ])

    def classifier_fn(images):
        return np.tile(np.array([[0.4, 0.6]]), (len(images), 1))

    explainer_a = LimeImageExplainer(random_state=42)
    explainer_b = LimeImageExplainer(random_state=42)

    data_a, labels_a, weights_a = explainer_a.data_labels(
        image=image,
        fudged_image=fudged_image,
        segments=segments,
        classifier_fn=classifier_fn,
        num_samples=50,
        batch_size=10,
        use_stratification=True,
        progress_bar=False,
    )

    data_b, labels_b, weights_b = explainer_b.data_labels(
        image=image,
        fudged_image=fudged_image,
        segments=segments,
        classifier_fn=classifier_fn,
        num_samples=50,
        batch_size=10,
        use_stratification=True,
        progress_bar=False,
    )

    assert np.array_equal(data_a, data_b)
    assert np.array_equal(labels_a, labels_b)
    assert np.array_equal(weights_a, weights_b)


def test_stratified_weight_adjustments_match_formula():
    explainer = LimeImageExplainer(random_state=7)

    image = np.zeros((4, 4, 3), dtype=np.float32)
    fudged_image = np.zeros_like(image)

    segments = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 3, 3],
        [2, 2, 3, 3],
    ])

    def classifier_fn(images):
        return np.tile(np.array([[0.4, 0.6]]), (len(images), 1))

    data, _, weight_adjustments = explainer.data_labels(
        image=image,
        fudged_image=fudged_image,
        segments=segments,
        classifier_fn=classifier_fn,
        num_samples=30,
        batch_size=10,
        use_stratification=True,
        progress_bar=False,
    )

    n_features = data.shape[1]

    expected = np.ones(len(data), dtype=np.float64)

    for i in range(1, len(data)):
        k = np.sum(data[i])
        expected[i] = (
            binom(n_features, k)
            / (2 ** n_features)
        ) * (n_features + 1)

    assert np.allclose(weight_adjustments, expected)


def test_weight_adjustments_multiply_kernel_weights():
    recorded = {}

    def kernel_fn(distances):
        recorded["distances"] = np.array(distances, copy=True)
        return np.exp(-(distances ** 2))

    class RecordingRegressor:
        def fit(self, X, y, sample_weight=None):
            self.sample_weight = np.array(sample_weight, copy=True)
            self.coef_ = np.zeros(X.shape[1])
            self.intercept_ = 0.0
            return self

        def score(self, X, y, sample_weight=None):
            return 1.0

        def predict(self, X):
            return np.zeros(X.shape[0])


    base = LimeBase(kernel_fn=kernel_fn, random_state=42)

    neighborhood_data = np.array([
        [1, 1],
        [1, 0],
        [0, 1],
    ], dtype=float)

    neighborhood_labels = np.array([
        [0.2, 0.8],
        [0.4, 0.6],
        [0.7, 0.3],
    ])

    distances = np.array([0.0, 1.0, 2.0])
    weight_adjustments = np.array([1.0, 0.5, 0.25])

    original_distances = distances.copy()
    regressor = RecordingRegressor()

    base.explain_instance_with_data(
        neighborhood_data=neighborhood_data,
        neighborhood_labels=neighborhood_labels,
        distances=distances,
        label=1,
        num_features=2,
        feature_selection="none",
        weight_adjustments=weight_adjustments,
        model_regressor=regressor,
    )

    expected_kernel_weights = np.exp(-(original_distances ** 2))
    expected_weights = expected_kernel_weights * weight_adjustments

    assert np.allclose(recorded["distances"], original_distances)
    assert np.allclose(regressor.sample_weight, expected_weights)


def test_stratified_explain_instance_end_to_end():
    image = np.zeros((4, 4, 3), dtype=np.float32)

    segments = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 3, 3],
        [2, 2, 3, 3],
    ])

    def segmentation_fn(_image):
        return segments

    def classifier_fn(images):
        probs = []
        for img in images:
            score = float(np.mean(img))
            score = np.clip(score, 0.0, 1.0)
            probs.append([1.0 - score, score])
        return np.array(probs)

    explainer_a = LimeImageExplainer(random_state=42)
    explainer_b = LimeImageExplainer(random_state=42)

    explanation_a = explainer_a.explain_instance(
        image=image,
        classifier_fn=classifier_fn,
        labels=(1,),
        top_labels=None,
        num_features=4,
        num_samples=50,
        batch_size=10,
        segmentation_fn=segmentation_fn,
        use_stratification=True,
        progress_bar=False,
    )

    explanation_b = explainer_b.explain_instance(
        image=image,
        classifier_fn=classifier_fn,
        labels=(1,),
        top_labels=None,
        num_features=4,
        num_samples=50,
        batch_size=10,
        segmentation_fn=segmentation_fn,
        use_stratification=True,
        progress_bar=False,
    )

    assert 1 in explanation_a.local_exp
    assert 1 in explanation_a.local_pred
    assert 1 in explanation_a.score

    feature_ids = [feature_id for feature_id, _ in explanation_a.local_exp[1]]
    weights = [weight for _, weight in explanation_a.local_exp[1]]

    assert all(0 <= feature_id < 4 for feature_id in feature_ids)
    assert np.all(np.isfinite(weights))
    assert np.all(np.isfinite(explanation_a.local_pred[1]))
    assert np.isfinite(explanation_a.score[1])

    assert explanation_a.local_exp[1] == explanation_b.local_exp[1]
    assert np.allclose(explanation_a.local_pred[1], explanation_b.local_pred[1])
    assert np.isclose(explanation_a.score[1], explanation_b.score[1])


def test_stratified_sampling_has_more_uniform_cardinality_coverage():
    n_features = 12
    num_samples = 5000

    segments = np.arange(n_features).reshape(3, 4)
    image = np.zeros((3, 4, 3), dtype=np.float32)
    fudged_image = np.zeros_like(image)

    def classifier_fn(images):
        return np.tile(np.array([[0.4, 0.6]]), (len(images), 1))

    standard = LimeImageExplainer(random_state=42)
    stratified = LimeImageExplainer(random_state=42)

    data_standard, _, _ = standard.data_labels(
        image=image,
        fudged_image=fudged_image,
        segments=segments,
        classifier_fn=classifier_fn,
        num_samples=num_samples,
        batch_size=100,
        use_stratification=False,
        progress_bar=False,
    )

    data_stratified, _, _ = stratified.data_labels(
        image=image,
        fudged_image=fudged_image,
        segments=segments,
        classifier_fn=classifier_fn,
        num_samples=num_samples,
        batch_size=100,
        use_stratification=True,
        progress_bar=False,
    )

    # The first perturbation is explicitly forced to all active features,
    # so exclude it from the statistical comparison.
    standard_counts = np.bincount(
        data_standard[1:].sum(axis=1),
        minlength=n_features + 1,
    )
    stratified_counts = np.bincount(
        data_stratified[1:].sum(axis=1),
        minlength=n_features + 1,
    )

    standard_freq = standard_counts / standard_counts.sum()
    stratified_freq = stratified_counts / stratified_counts.sum()

    uniform = np.full(n_features + 1, 1.0 / (n_features + 1))

    standard_mad = np.mean(np.abs(standard_freq - uniform))
    stratified_mad = np.mean(np.abs(stratified_freq - uniform))

    assert stratified_mad < standard_mad
    assert stratified_mad < 0.02
    assert stratified_mad < 0.25 * standard_mad
