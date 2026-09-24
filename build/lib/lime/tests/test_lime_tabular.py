import unittest

import numpy as np
import collections
import sklearn  # noqa
import sklearn.datasets
import sklearn.ensemble
import sklearn.linear_model  # noqa
from numpy.testing import assert_array_equal
from sklearn.datasets import load_iris, make_classification, make_multilabel_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from lime.discretize import QuartileDiscretizer, DecileDiscretizer, EntropyDiscretizer


try:
    from sklearn.model_selection import train_test_split
except ImportError:
    # Deprecated in scikit-learn version 0.18, removed in 0.20
    from sklearn.cross_validation import train_test_split

from lime.lime_tabular import LimeTabularExplainer


class TestLimeTabular(unittest.TestCase):

    def setUp(self):
        iris = load_iris()

        self.feature_names = iris.feature_names
        self.target_names = iris.target_names

        (self.train,
         self.test,
         self.labels_train,
         self.labels_test) = train_test_split(iris.data, iris.target, train_size=0.80)

    def test_lime_explainer_good_regressor(self):
        np.random.seed(1)
        rf = RandomForestClassifier(n_estimators=500)
        rf.fit(self.train, self.labels_train)
        i = np.random.randint(0, self.test.shape[0])

        explainer = LimeTabularExplainer(self.train,
                                         mode="classification",
                                         feature_names=self.feature_names,
                                         class_names=self.target_names,
                                         discretize_continuous=True)

        exp = explainer.explain_instance(self.test[i],
                                         rf.predict_proba,
                                         num_features=2,
                                         model_regressor=LinearRegression())

        self.assertIsNotNone(exp)
        keys = [x[0] for x in exp.as_list()]
        self.assertEqual(1,
                         sum([1 if 'petal width' in x else 0 for x in keys]),
                         "Petal Width is a major feature")
        self.assertEqual(1,
                         sum([1 if 'petal length' in x else 0 for x in keys]),
                         "Petal Length is a major feature")

    def test_lime_explainer_good_regressor_synthetic_data(self):
        X, y = make_classification(n_samples=1000,
                                   n_features=20,
                                   n_informative=2,
                                   n_redundant=2,
                                   random_state=10)

        rf = RandomForestClassifier(n_estimators=500)
        rf.fit(X, y)
        instance = np.random.randint(0, X.shape[0])
        feature_names = ["feature" + str(i) for i in range(20)]
        explainer = LimeTabularExplainer(X,
                                         feature_names=feature_names,
                                         discretize_continuous=True)

        exp = explainer.explain_instance(X[instance], rf.predict_proba)

        self.assertIsNotNone(exp)
        self.assertEqual(10, len(exp.as_list()))

    def test_lime_explainer_sparse_synthetic_data(self):
        n_features = 20
        X, y = make_multilabel_classification(n_samples=100,
                                              sparse=True,
                                              n_features=n_features,
                                              n_classes=1,
                                              n_labels=2)
        rf = RandomForestClassifier(n_estimators=500)
        rf.fit(X, y)
        instance = np.random.randint(0, X.shape[0])
        feature_names = ["feature" + str(i) for i in range(n_features)]
        explainer = LimeTabularExplainer(X,
                                         feature_names=feature_names,
                                         discretize_continuous=True)

        exp = explainer.explain_instance(X[instance], rf.predict_proba)

        self.assertIsNotNone(exp)
        self.assertEqual(10, len(exp.as_list()))

    def test_lime_explainer_no_regressor(self):
        np.random.seed(1)

        rf = RandomForestClassifier(n_estimators=500)
        rf.fit(self.train, self.labels_train)
        i = np.random.randint(0, self.test.shape[0])

        explainer = LimeTabularExplainer(self.train,
                                         feature_names=self.feature_names,
                                         class_names=self.target_names,
                                         discretize_continuous=True)

        exp = explainer.explain_instance(self.test[i],
                                         rf.predict_proba,
                                         num_features=2)
        self.assertIsNotNone(exp)
        keys = [x[0] for x in exp.as_list()]
        self.assertEqual(1,
                         sum([1 if 'petal width' in x else 0 for x in keys]),
                         "Petal Width is a major feature")
        self.assertEqual(1,
                         sum([1 if 'petal length' in x else 0 for x in keys]),
                         "Petal Length is a major feature")

    def test_lime_explainer_entropy_discretizer(self):
        np.random.seed(1)

        rf = RandomForestClassifier(n_estimators=500)
        rf.fit(self.train, self.labels_train)
        i = np.random.randint(0, self.test.shape[0])

        explainer = LimeTabularExplainer(self.train,
                                         feature_names=self.feature_names,
                                         class_names=self.target_names,
                                         training_labels=self.labels_train,
                                         discretize_continuous=True,
                                         discretizer='entropy')

        exp = explainer.explain_instance(self.test[i],
                                         rf.predict_proba,
                                         num_features=2)
        self.assertIsNotNone(exp)
        keys = [x[0] for x in exp.as_list()]
        print(keys)
        self.assertEqual(1,
                         sum([1 if 'petal width' in x else 0 for x in keys]),
                         "Petal Width is a major feature")
        self.assertEqual(1,
                         sum([1 if 'petal length' in x else 0 for x in keys]),
                         "Petal Length is a major feature")

    def test_lime_tabular_explainer_equal_random_state(self):
        X, y = make_classification(n_samples=1000,
                                   n_features=20,
                                   n_informative=2,
                                   n_redundant=2,
                                   random_state=10)

        rf = RandomForestClassifier(n_estimators=500, random_state=10)
        rf.fit(X, y)
        instance = np.random.RandomState(10).randint(0, X.shape[0])
        feature_names = ["feature" + str(i) for i in range(20)]

        # ----------------------------------------------------------------------
        # -------------------------Quartile Discretizer-------------------------
        # ----------------------------------------------------------------------
        discretizer = QuartileDiscretizer(X, [], feature_names, y,
                                          random_state=10)
        explainer_1 = LimeTabularExplainer(X,
                                           feature_names=feature_names,
                                           discretize_continuous=True,
                                           discretizer=discretizer,
                                           random_state=10)
        exp_1 = explainer_1.explain_instance(X[instance], rf.predict_proba,
                                             num_samples=500)

        discretizer = QuartileDiscretizer(X, [], feature_names, y,
                                          random_state=10)
        explainer_2 = LimeTabularExplainer(X,
                                           feature_names=feature_names,
                                           discretize_continuous=True,
                                           discretizer=discretizer,
                                           random_state=10)
        exp_2 = explainer_2.explain_instance(X[instance], rf.predict_proba,
                                             num_samples=500)

        self.assertDictEqual(exp_1.as_map(), exp_2.as_map())

        # ----------------------------------------------------------------------
        # --------------------------Decile Discretizer--------------------------
        # ----------------------------------------------------------------------
        discretizer = DecileDiscretizer(X, [], feature_names, y,
                                        random_state=10)
        explainer_1 = LimeTabularExplainer(X,
                                           feature_names=feature_names,
                                           discretize_continuous=True,
                                           discretizer=discretizer,
                                           random_state=10)
        exp_1 = explainer_1.explain_instance(X[instance], rf.predict_proba,
                                             num_samples=500)

        discretizer = DecileDiscretizer(X, [], feature_names, y,
                                        random_state=10)
        explainer_2 = LimeTabularExplainer(X,
                                           feature_names=feature_names,
                                           discretize_continuous=True,
                                           discretizer=discretizer,
                                           random_state=10)
        exp_2 = explainer_2.explain_instance(X[instance], rf.predict_proba,
                                             num_samples=500)

        self.assertDictEqual(exp_1.as_map(), exp_2.as_map())

        # ----------------------------------------------------------------------
        # -------------------------Entropy Discretizer--------------------------
        # ----------------------------------------------------------------------
        discretizer = EntropyDiscretizer(X, [], feature_names, y,
                                         random_state=10)
        explainer_1 = LimeTabularExplainer(X,
                                           feature_names=feature_names,
                                           discretize_continuous=True,
                                           discretizer=discretizer,
                                           random_state=10)
        exp_1 = explainer_1.explain_instance(X[instance], rf.predict_proba,
                                             num_samples=500)

        discretizer = EntropyDiscretizer(X, [], feature_names, y,
                                         random_state=10)
        explainer_2 = LimeTabularExplainer(X,
                                           feature_names=feature_names,
                                           discretize_continuous=True,
                                           discretizer=discretizer,
                                           random_state=10)
        exp_2 = explainer_2.explain_instance(X[instance], rf.predict_proba,
                                             num_samples=500)

        self.assertDictEqual(exp_1.as_map(), exp_2.as_map())

    def test_lime_tabular_explainer_not_equal_random_state(self):
        X, y = make_classification(n_samples=1000,
                                   n_features=20,
                                   n_informative=2,
                                   n_redundant=2,
                                   random_state=10)

        rf = RandomForestClassifier(n_estimators=500, random_state=10)
        rf.fit(X, y)
        instance = np.random.RandomState(10).randint(0, X.shape[0])
        feature_names = ["feature" + str(i) for i in range(20)]

        # ----------------------------------------------------------------------
        # -------------------------Quartile Discretizer-------------------------
        # ----------------------------------------------------------------------

        # ---------------------------------[1]----------------------------------
        discretizer = QuartileDiscretizer(X, [], feature_names, y,
                                          random_state=20)
        explainer_1 = LimeTabularExplainer(X,
                                           feature_names=feature_names,
                                           discretize_continuous=True,
                                           discretizer=discretizer,
                                           random_state=10)
        exp_1 = explainer_1.explain_instance(X[instance], rf.predict_proba,
                                             num_samples=500)

        discretizer = QuartileDiscretizer(X, [], feature_names, y,
                                          random_state=10)
        explainer_2 = LimeTabularExplainer(X,
                                           feature_names=feature_names,
                                           discretize_continuous=True,
                                           discretizer=discretizer,
                                           random_state=10)
        exp_2 = explainer_2.explain_instance(X[instance], rf.predict_proba,
                                             num_samples=500)

        self.assertTrue(exp_1.as_map() != exp_2.as_map())

        # ---------------------------------[2]----------------------------------
        discretizer = QuartileDiscretizer(X, [], feature_names, y,
                                          random_state=20)
        explainer_1 = LimeTabularExplainer(X,
                                           feature_names=feature_names,
                                           discretize_continuous=True,
                                           discretizer=discretizer,
                                           random_state=20)
        exp_1 = explainer_1.explain_instance(X[instance], rf.predict_proba,
                                             num_samples=500)

        discretizer = QuartileDiscretizer(X, [], feature_names, y,
                                          random_state=10)
        explainer_2 = LimeTabularExplainer(X,
                                           feature_names=feature_names,
                                           discretize_continuous=True,
                                           discretizer=discretizer,
                                           random_state=10)
        exp_2 = explainer_2.explain_instance(X[instance], rf.predict_proba,
                                             num_samples=500)

        self.assertTrue(exp_1.as_map() != exp_2.as_map())

        # ---------------------------------[3]----------------------------------
        discretizer = QuartileDiscretizer(X, [], feature_names, y,
                                          random_state=20)
        explainer_1 = LimeTabularExplainer(X,
                                           feature_names=feature_names,
                                           discretize_continuous=True,
                                           discretizer=discretizer,
                                           random_state=20)
        exp_1 = explainer_1.explain_instance(X[instance], rf.predict_proba,
                                             num_samples=500)

        discretizer = QuartileDiscretizer(X, [], feature_names, y,
                                          random_state=20)
        explainer_2 = LimeTabularExplainer(X,
                                           feature_names=feature_names,
                                           discretize_continuous=True,
                                           discretizer=discretizer,
                                           random_state=10)
        exp_2 = explainer_2.explain_instance(X[instance], rf.predict_proba,
                                             num_samples=500)

        self.assertTrue(exp_1.as_map() != exp_2.as_map())

        # ---------------------------------[4]----------------------------------
        discretizer = QuartileDiscretizer(X, [], feature_names, y,
                                          random_state=20)
        explainer_1 = LimeTabularExplainer(X,
                                           feature_names=feature_names,
                                           discretize_continuous=True,
                                           discretizer=discretizer,
                                           random_state=20)
        exp_1 = explainer_1.explain_instance(X[instance], rf.predict_proba,
                                             num_samples=500)

        discretizer = QuartileDiscretizer(X, [], feature_names, y,
                                          random_state=20)
        explainer_2 = LimeTabularExplainer(X,
                                           feature_names=feature_names,
                                           discretize_continuous=True,
                                           discretizer=discretizer,
                                           random_state=20)
        exp_2 = explainer_2.explain_instance(X[instance], rf.predict_proba,
                                             num_samples=500)

        self.assertFalse(exp_1.as_map() != exp_2.as_map())

        # ----------------------------------------------------------------------
        # --------------------------Decile Discretizer--------------------------
        # ----------------------------------------------------------------------

        # ---------------------------------[1]----------------------------------
        discretizer = DecileDiscretizer(X, [], feature_names, y,
                                        random_state=20)
        explainer_1 = LimeTabularExplainer(X,
                                           feature_names=feature_names,
                                           discretize_continuous=True,
                                           discretizer=discretizer,
                                           random_state=10)
        exp_1 = explainer_1.explain_instance(X[instance], rf.predict_proba,
                                             num_samples=500)

        discretizer = DecileDiscretizer(X, [], feature_names, y,
                                        random_state=10)
        explainer_2 = LimeTabularExplainer(X,
                                           feature_names=feature_names,
                                           discretize_continuous=True,
                                           discretizer=discretizer,
                                           random_state=10)
        exp_2 = explainer_2.explain_instance(X[instance], rf.predict_proba,
                                             num_samples=500)

        self.assertTrue(exp_1.as_map() != exp_2.as_map())

        # ---------------------------------[2]----------------------------------
        discretizer = DecileDiscretizer(X, [], feature_names, y,
                                        random_state=20)
        explainer_1 = LimeTabularExplainer(X,
                                           feature_names=feature_names,
                                           discretize_continuous=True,
                                           discretizer=discretizer,
                                           random_state=20)
        exp_1 = explainer_1.explain_instance(X[instance], rf.predict_proba,
                                             num_samples=500)

        discretizer = DecileDiscretizer(X, [], feature_names, y,
                                        random_state=10)
        explainer_2 = LimeTabularExplainer(X,
                                           feature_names=feature_names,
                                           discretize_continuous=True,
                                           discretizer=discretizer,
                                           random_state=10)
        exp_2 = explainer_2.explain_instance(X[instance], rf.predict_proba,
                                             num_samples=500)

        self.assertTrue(exp_1.as_map() != exp_2.as_map())

        # ---------------------------------[3]----------------------------------
        discretizer = DecileDiscretizer(X, [], feature_names, y,
                                        random_state=20)
        explainer_1 = LimeTabularExplainer(X,
                                           feature_names=feature_names,
                                           discretize_continuous=True,
                                           discretizer=discretizer,
                                           random_state=20)
        exp_1 = explainer_1.explain_instance(X[instance], rf.predict_proba,
                                             num_samples=500)

        discretizer = DecileDiscretizer(X, [], feature_names, y,
                                        random_state=20)
        explainer_2 = LimeTabularExplainer(X,
                                           feature_names=feature_names,
                                           discretize_continuous=True,
                                           discretizer=discretizer,
                                           random_state=10)
        exp_2 = explainer_2.explain_instance(X[instance], rf.predict_proba,
                                             num_samples=500)

        self.assertTrue(exp_1.as_map() != exp_2.as_map())

        # ---------------------------------[4]----------------------------------
        discretizer = DecileDiscretizer(X, [], feature_names, y,
                                        random_state=20)
        explainer_1 = LimeTabularExplainer(X,
                                           feature_names=feature_names,
                                           discretize_continuous=True,
                                           discretizer=discretizer,
                                           random_state=20)
        exp_1 = explainer_1.explain_instance(X[instance], rf.predict_proba,
                                             num_samples=500)

        discretizer = DecileDiscretizer(X, [], feature_names, y,
                                        random_state=20)
        explainer_2 = LimeTabularExplainer(X,
                                           feature_names=feature_names,
                                           discretize_continuous=True,
                                           discretizer=discretizer,
                                           random_state=20)
        exp_2 = explainer_2.explain_instance(X[instance], rf.predict_proba,
                                             num_samples=500)

        self.assertFalse(exp_1.as_map() != exp_2.as_map())

        # ----------------------------------------------------------------------
        # --------------------------Entropy Discretizer-------------------------
        # ----------------------------------------------------------------------

        # ---------------------------------[1]----------------------------------
        discretizer = EntropyDiscretizer(X, [], feature_names, y,
                                         random_state=20)
        explainer_1 = LimeTabularExplainer(X,
                                           feature_names=feature_names,
                                           discretize_continuous=True,
                                           discretizer=discretizer,
                                           random_state=10)
        exp_1 = explainer_1.explain_instance(X[instance], rf.predict_proba,
                                             num_samples=500)

        discretizer = EntropyDiscretizer(X, [], feature_names, y,
                                         random_state=10)
        explainer_2 = LimeTabularExplainer(X,
                                           feature_names=feature_names,
                                           discretize_continuous=True,
                                           discretizer=discretizer,
                                           random_state=10)
        exp_2 = explainer_2.explain_instance(X[instance], rf.predict_proba,
                                             num_samples=500)

        self.assertTrue(exp_1.as_map() != exp_2.as_map())

        # ---------------------------------[2]----------------------------------
        discretizer = EntropyDiscretizer(X, [], feature_names, y,
                                         random_state=20)
        explainer_1 = LimeTabularExplainer(X,
                                           feature_names=feature_names,
                                           discretize_continuous=True,
                                           discretizer=discretizer,
                                           random_state=20)
        exp_1 = explainer_1.explain_instance(X[instance], rf.predict_proba,
                                             num_samples=500)

        discretizer = EntropyDiscretizer(X, [], feature_names, y,
                                         random_state=10)
        explainer_2 = LimeTabularExplainer(X,
                                           feature_names=feature_names,
                                           discretize_continuous=True,
                                           discretizer=discretizer,
                                           random_state=10)
        exp_2 = explainer_2.explain_instance(X[instance], rf.predict_proba,
                                             num_samples=500)

        self.assertTrue(exp_1.as_map() != exp_2.as_map())

        # ---------------------------------[3]----------------------------------
        discretizer = EntropyDiscretizer(X, [], feature_names, y,
                                         random_state=20)
        explainer_1 = LimeTabularExplainer(X,
                                           feature_names=feature_names,
                                           discretize_continuous=True,
                                           discretizer=discretizer,
                                           random_state=20)
        exp_1 = explainer_1.explain_instance(X[instance], rf.predict_proba,
                                             num_samples=500)

        discretizer = EntropyDiscretizer(X, [], feature_names, y,
                                         random_state=20)
        explainer_2 = LimeTabularExplainer(X,
                                           feature_names=feature_names,
                                           discretize_continuous=True,
                                           discretizer=discretizer,
                                           random_state=10)
        exp_2 = explainer_2.explain_instance(X[instance], rf.predict_proba,
                                             num_samples=500)

        self.assertTrue(exp_1.as_map() != exp_2.as_map())

        # ---------------------------------[4]----------------------------------
        discretizer = EntropyDiscretizer(X, [], feature_names, y,
                                         random_state=20)
        explainer_1 = LimeTabularExplainer(X,
                                           feature_names=feature_names,
                                           discretize_continuous=True,
                                           discretizer=discretizer,
                                           random_state=20)
        exp_1 = explainer_1.explain_instance(X[instance], rf.predict_proba,
                                             num_samples=500)

        discretizer = EntropyDiscretizer(X, [], feature_names, y,
                                         random_state=20)
        explainer_2 = LimeTabularExplainer(X,
                                           feature_names=feature_names,
                                           discretize_continuous=True,
                                           discretizer=discretizer,
                                           random_state=20)
        exp_2 = explainer_2.explain_instance(X[instance], rf.predict_proba,
                                             num_samples=500)

        self.assertFalse(exp_1.as_map() != exp_2.as_map())

    def testFeatureNamesAndCategoricalFeats(self):
        training_data = np.array([[0., 1.], [1., 0.]])

        explainer = LimeTabularExplainer(training_data=training_data)
        self.assertEqual(explainer.feature_names, ['0', '1'])
        self.assertEqual(explainer.categorical_features, [0, 1])

        explainer = LimeTabularExplainer(
            training_data=training_data,
            feature_names=np.array(['one', 'two'])
        )
        self.assertEqual(explainer.feature_names, ['one', 'two'])

        explainer = LimeTabularExplainer(
            training_data=training_data,
            categorical_features=np.array([0]),
            discretize_continuous=False
        )
        self.assertEqual(explainer.categorical_features, [0])

    def testFeatureValues(self):
        training_data = np.array([
            [0, 0, 2],
            [1, 1, 0],
            [0, 2, 2],
            [1, 3, 0]
        ])

        explainer = LimeTabularExplainer(
            training_data=training_data,
            categorical_features=[0, 1, 2]
        )

        self.assertEqual(set(explainer.feature_values[0]), {0, 1})
        self.assertEqual(set(explainer.feature_values[1]), {0, 1, 2, 3})
        self.assertEqual(set(explainer.feature_values[2]), {0, 2})

        assert_array_equal(explainer.feature_frequencies[0], np.array([.5, .5]))
        assert_array_equal(explainer.feature_frequencies[1], np.array([.25, .25, .25, .25]))
        assert_array_equal(explainer.feature_frequencies[2], np.array([.5, .5]))

    def test_lime_explainer_with_data_stats(self):
        np.random.seed(1)

        rf = RandomForestClassifier(n_estimators=500)
        rf.fit(self.train, self.labels_train)
        i = np.random.randint(0, self.test.shape[0])

        # Generate stats using a quartile descritizer
        descritizer = QuartileDiscretizer(self.train, [], self.feature_names, self.target_names,
                                          random_state=20)

        d_means = descritizer.means
        d_stds = descritizer.stds
        d_mins = descritizer.mins
        d_maxs = descritizer.maxs
        d_bins = descritizer.bins(self.train, self.target_names)

        # Compute feature values and frequencies of all columns
        cat_features = np.arange(self.train.shape[1])
        discretized_training_data = descritizer.discretize(self.train)

        feature_values = {}
        feature_frequencies = {}
        for feature in cat_features:
            column = discretized_training_data[:, feature]
            feature_count = collections.Counter(column)
            values, frequencies = map(list, zip(*(feature_count.items())))
            feature_values[feature] = values
            feature_frequencies[feature] = frequencies

        # Convert bins to list from array
        d_bins_revised = {}
        index = 0
        for bin in d_bins:
            d_bins_revised[index] = bin.tolist()
            index = index+1

        # Descritized stats
        data_stats = {}
        data_stats["means"] = d_means
        data_stats["stds"] = d_stds
        data_stats["maxs"] = d_maxs
        data_stats["mins"] = d_mins
        data_stats["bins"] = d_bins_revised
        data_stats["feature_values"] = feature_values
        data_stats["feature_frequencies"] = feature_frequencies

        data = np.zeros((2, len(self.feature_names)))
        explainer = LimeTabularExplainer(
            data, feature_names=self.feature_names, random_state=10,
            training_data_stats=data_stats, training_labels=self.target_names)

        exp = explainer.explain_instance(self.test[i],
                                         rf.predict_proba,
                                         num_features=2,
                                         model_regressor=LinearRegression())

        self.assertIsNotNone(exp)
        keys = [x[0] for x in exp.as_list()]
        self.assertEqual(1,
                         sum([1 if 'petal width' in x else 0 for x in keys]),
                         "Petal Width is a major feature")
        self.assertEqual(1,
                         sum([1 if 'petal length' in x else 0 for x in keys]),
                         "Petal Length is a major feature")


if __name__ == '__main__':
    unittest.main()
