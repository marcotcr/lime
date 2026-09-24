import unittest
from unittest import TestCase

import numpy as np

from sklearn.datasets import load_iris

from lime.discretize import QuartileDiscretizer, DecileDiscretizer, EntropyDiscretizer


class TestDiscretize(TestCase):

    def setUp(self):
        iris = load_iris()

        self.feature_names = iris.feature_names
        self.x = iris.data
        self.y = iris.target

    def check_random_state_for_discretizer_class(self, DiscretizerClass):
        # ----------------------------------------------------------------------
        # -----------Check if the same random_state produces the same-----------
        # -------------results for different discretizer instances.-------------
        # ----------------------------------------------------------------------
        discretizer = DiscretizerClass(self.x, [], self.feature_names, self.y,
                                       random_state=10)
        x_1 = discretizer.undiscretize(discretizer.discretize(self.x))

        discretizer = DiscretizerClass(self.x, [], self.feature_names, self.y,
                                       random_state=10)
        x_2 = discretizer.undiscretize(discretizer.discretize(self.x))

        self.assertEqual((x_1 == x_2).sum(), x_1.shape[0] * x_1.shape[1])

        discretizer = DiscretizerClass(self.x, [], self.feature_names, self.y,
                                       random_state=np.random.RandomState(10))
        x_1 = discretizer.undiscretize(discretizer.discretize(self.x))

        discretizer = DiscretizerClass(self.x, [], self.feature_names, self.y,
                                       random_state=np.random.RandomState(10))
        x_2 = discretizer.undiscretize(discretizer.discretize(self.x))

        self.assertEqual((x_1 == x_2).sum(), x_1.shape[0] * x_1.shape[1])

        # ----------------------------------------------------------------------
        # ---------Check if two different random_state values produces----------
        # -------different results for different discretizers instances.--------
        # ----------------------------------------------------------------------
        discretizer = DiscretizerClass(self.x, [], self.feature_names, self.y,
                                       random_state=10)
        x_1 = discretizer.undiscretize(discretizer.discretize(self.x))

        discretizer = DiscretizerClass(self.x, [], self.feature_names, self.y,
                                       random_state=20)
        x_2 = discretizer.undiscretize(discretizer.discretize(self.x))

        self.assertFalse((x_1 == x_2).sum() == x_1.shape[0] * x_1.shape[1])

        discretizer = DiscretizerClass(self.x, [], self.feature_names, self.y,
                                       random_state=np.random.RandomState(10))
        x_1 = discretizer.undiscretize(discretizer.discretize(self.x))

        discretizer = DiscretizerClass(self.x, [], self.feature_names, self.y,
                                       random_state=np.random.RandomState(20))
        x_2 = discretizer.undiscretize(discretizer.discretize(self.x))

        self.assertFalse((x_1 == x_2).sum() == x_1.shape[0] * x_1.shape[1])

    def test_random_state(self):
        self.check_random_state_for_discretizer_class(QuartileDiscretizer)

        self.check_random_state_for_discretizer_class(DecileDiscretizer)

        self.check_random_state_for_discretizer_class(EntropyDiscretizer)

    def test_feature_names_1(self):
        self.maxDiff = None
        discretizer = QuartileDiscretizer(self.x, [], self.feature_names,
                                          self.y, random_state=10)
        self.assertDictEqual(
            {0: ['sepal length (cm) <= 5.10',
                 '5.10 < sepal length (cm) <= 5.80',
                 '5.80 < sepal length (cm) <= 6.40',
                 'sepal length (cm) > 6.40'],
             1: ['sepal width (cm) <= 2.80',
                 '2.80 < sepal width (cm) <= 3.00',
                 '3.00 < sepal width (cm) <= 3.30',
                 'sepal width (cm) > 3.30'],
             2: ['petal length (cm) <= 1.60',
                 '1.60 < petal length (cm) <= 4.35',
                 '4.35 < petal length (cm) <= 5.10',
                 'petal length (cm) > 5.10'],
             3: ['petal width (cm) <= 0.30',
                 '0.30 < petal width (cm) <= 1.30',
                 '1.30 < petal width (cm) <= 1.80',
                 'petal width (cm) > 1.80']},
            discretizer.names)

    def test_feature_names_2(self):
        self.maxDiff = None
        discretizer = DecileDiscretizer(self.x, [], self.feature_names, self.y,
                                        random_state=10)
        self.assertDictEqual(
            {0: ['sepal length (cm) <= 4.80',
                 '4.80 < sepal length (cm) <= 5.00',
                 '5.00 < sepal length (cm) <= 5.27',
                 '5.27 < sepal length (cm) <= 5.60',
                 '5.60 < sepal length (cm) <= 5.80',
                 '5.80 < sepal length (cm) <= 6.10',
                 '6.10 < sepal length (cm) <= 6.30',
                 '6.30 < sepal length (cm) <= 6.52',
                 '6.52 < sepal length (cm) <= 6.90',
                 'sepal length (cm) > 6.90'],
             1: ['sepal width (cm) <= 2.50',
                 '2.50 < sepal width (cm) <= 2.70',
                 '2.70 < sepal width (cm) <= 2.80',
                 '2.80 < sepal width (cm) <= 3.00',
                 '3.00 < sepal width (cm) <= 3.10',
                 '3.10 < sepal width (cm) <= 3.20',
                 '3.20 < sepal width (cm) <= 3.40',
                 '3.40 < sepal width (cm) <= 3.61',
                 'sepal width (cm) > 3.61'],
             2: ['petal length (cm) <= 1.40',
                 '1.40 < petal length (cm) <= 1.50',
                 '1.50 < petal length (cm) <= 1.70',
                 '1.70 < petal length (cm) <= 3.90',
                 '3.90 < petal length (cm) <= 4.35',
                 '4.35 < petal length (cm) <= 4.64',
                 '4.64 < petal length (cm) <= 5.00',
                 '5.00 < petal length (cm) <= 5.32',
                 '5.32 < petal length (cm) <= 5.80',
                 'petal length (cm) > 5.80'],
             3: ['petal width (cm) <= 0.20',
                 '0.20 < petal width (cm) <= 0.40',
                 '0.40 < petal width (cm) <= 1.16',
                 '1.16 < petal width (cm) <= 1.30',
                 '1.30 < petal width (cm) <= 1.50',
                 '1.50 < petal width (cm) <= 1.80',
                 '1.80 < petal width (cm) <= 1.90',
                 '1.90 < petal width (cm) <= 2.20',
                 'petal width (cm) > 2.20']},
            discretizer.names)

    def test_feature_names_3(self):
        self.maxDiff = None
        discretizer = EntropyDiscretizer(self.x, [], self.feature_names,
                                         self.y, random_state=10)
        self.assertDictEqual(
            {0: ['sepal length (cm) <= 4.85',
                 '4.85 < sepal length (cm) <= 5.45',
                 '5.45 < sepal length (cm) <= 5.55',
                 '5.55 < sepal length (cm) <= 5.85',
                 '5.85 < sepal length (cm) <= 6.15',
                 '6.15 < sepal length (cm) <= 7.05',
                 'sepal length (cm) > 7.05'],
             1: ['sepal width (cm) <= 2.45',
                 '2.45 < sepal width (cm) <= 2.95',
                 '2.95 < sepal width (cm) <= 3.05',
                 '3.05 < sepal width (cm) <= 3.35',
                 '3.35 < sepal width (cm) <= 3.45',
                 '3.45 < sepal width (cm) <= 3.55',
                 'sepal width (cm) > 3.55'],
             2: ['petal length (cm) <= 2.45',
                 '2.45 < petal length (cm) <= 4.45',
                 '4.45 < petal length (cm) <= 4.75',
                 '4.75 < petal length (cm) <= 5.15',
                 'petal length (cm) > 5.15'],
             3: ['petal width (cm) <= 0.80',
                 '0.80 < petal width (cm) <= 1.35',
                 '1.35 < petal width (cm) <= 1.75',
                 '1.75 < petal width (cm) <= 1.85',
                 'petal width (cm) > 1.85']},
            discretizer.names)


if __name__ == '__main__':
    unittest.main()
