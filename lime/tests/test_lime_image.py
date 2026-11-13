import unittest

import numpy as np

from lime.lime_image import LimeImageExplainer


class TestLimeImageExplainer(unittest.TestCase):

    def test_data_labels_masks_and_batches(self):
        image = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
        fudged_image = -image - 1
        for segment_ids in ([0, 1, 2], [0, 2, 8]):
            segments = np.tile(segment_ids, (2, 1))
            for batch_size in (1, 4):
                with self.subTest(segments=segment_ids, batch_size=batch_size):
                    batches = []

                    def classifier_fn(images):
                        batches.append(images.copy())
                        return images.mean(axis=(1, 2))

                    explainer = LimeImageExplainer(random_state=42)
                    data, labels = explainer.data_labels(
                        image, fudged_image, segments, classifier_fn,
                        num_samples=10, batch_size=batch_size,
                        progress_bar=False)
                    perturbed = np.concatenate(batches)
                    np.testing.assert_array_equal(data[0], 1)
                    for row, actual in zip(data, perturbed):
                        expected = image.copy()
                        for feature, enabled in enumerate(row):
                            if not enabled:
                                expected[segments == feature] = (
                                    fudged_image[segments == feature])
                        np.testing.assert_array_equal(actual, expected)
                    np.testing.assert_array_equal(
                        labels, perturbed.mean(axis=(1, 2)))
                    self.assertEqual(
                        [len(batch) for batch in batches],
                        [batch_size] * (10 // batch_size)
                        + ([10 % batch_size] if 10 % batch_size else []))
        np.testing.assert_array_equal(
            image, np.arange(18, dtype=np.float32).reshape(2, 3, 3))
        np.testing.assert_array_equal(fudged_image, -image - 1)
