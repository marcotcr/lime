import re
import unittest

import sklearn # noqa
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

import numpy as np

from lime.lime_text import LimeTextExplainer
from lime.lime_text import IndexedCharacters, IndexedString


class TestLimeText(unittest.TestCase):

    def test_lime_text_explainer_good_regressor(self):
        categories = ['alt.atheism', 'soc.religion.christian']
        newsgroups_train = fetch_20newsgroups(subset='train',
                                              categories=categories)
        newsgroups_test = fetch_20newsgroups(subset='test',
                                             categories=categories)
        class_names = ['atheism', 'christian']
        vectorizer = TfidfVectorizer(lowercase=False)
        train_vectors = vectorizer.fit_transform(newsgroups_train.data)
        test_vectors = vectorizer.transform(newsgroups_test.data)
        nb = MultinomialNB(alpha=.01)
        nb.fit(train_vectors, newsgroups_train.target)
        pred = nb.predict(test_vectors)
        f1_score(newsgroups_test.target, pred, average='weighted')
        c = make_pipeline(vectorizer, nb)
        explainer = LimeTextExplainer(class_names=class_names)
        idx = 83
        exp = explainer.explain_instance(newsgroups_test.data[idx],
                                         c.predict_proba, num_features=6)
        self.assertIsNotNone(exp)
        self.assertEqual(6, len(exp.as_list()))

    def test_lime_text_tabular_equal_random_state(self):
        categories = ['alt.atheism', 'soc.religion.christian']
        newsgroups_train = fetch_20newsgroups(subset='train',
                                              categories=categories)
        newsgroups_test = fetch_20newsgroups(subset='test',
                                             categories=categories)
        class_names = ['atheism', 'christian']
        vectorizer = TfidfVectorizer(lowercase=False)
        train_vectors = vectorizer.fit_transform(newsgroups_train.data)
        test_vectors = vectorizer.transform(newsgroups_test.data)
        nb = MultinomialNB(alpha=.01)
        nb.fit(train_vectors, newsgroups_train.target)
        pred = nb.predict(test_vectors)
        f1_score(newsgroups_test.target, pred, average='weighted')
        c = make_pipeline(vectorizer, nb)

        explainer = LimeTextExplainer(class_names=class_names, random_state=10)
        exp_1 = explainer.explain_instance(newsgroups_test.data[83],
                                           c.predict_proba, num_features=6)

        explainer = LimeTextExplainer(class_names=class_names, random_state=10)
        exp_2 = explainer.explain_instance(newsgroups_test.data[83],
                                           c.predict_proba, num_features=6)

        self.assertTrue(exp_1.as_map() == exp_2.as_map())

    def test_lime_text_tabular_not_equal_random_state(self):
        categories = ['alt.atheism', 'soc.religion.christian']
        newsgroups_train = fetch_20newsgroups(subset='train',
                                              categories=categories)
        newsgroups_test = fetch_20newsgroups(subset='test',
                                             categories=categories)
        class_names = ['atheism', 'christian']
        vectorizer = TfidfVectorizer(lowercase=False)
        train_vectors = vectorizer.fit_transform(newsgroups_train.data)
        test_vectors = vectorizer.transform(newsgroups_test.data)
        nb = MultinomialNB(alpha=.01)
        nb.fit(train_vectors, newsgroups_train.target)
        pred = nb.predict(test_vectors)
        f1_score(newsgroups_test.target, pred, average='weighted')
        c = make_pipeline(vectorizer, nb)

        explainer = LimeTextExplainer(
            class_names=class_names, random_state=10)
        exp_1 = explainer.explain_instance(newsgroups_test.data[83],
                                           c.predict_proba, num_features=6)

        explainer = LimeTextExplainer(
            class_names=class_names, random_state=20)
        exp_2 = explainer.explain_instance(newsgroups_test.data[83],
                                           c.predict_proba, num_features=6)

        self.assertFalse(exp_1.as_map() == exp_2.as_map())

    def test_indexed_characters_bow(self):
        s = 'Please, take your time'
        inverse_vocab = ['P', 'l', 'e', 'a', 's', ',', ' ', 't', 'k', 'y', 'o', 'u', 'r', 'i', 'm']
        positions = [[0], [1], [2, 5, 11, 21], [3, 9],
                     [4], [6], [7, 12, 17], [8, 18], [10],
                     [13], [14], [15], [16], [19], [20]]
        ic = IndexedCharacters(s)

        self.assertTrue(np.array_equal(ic.as_np, np.array(list(s))))
        self.assertTrue(np.array_equal(ic.string_start, np.arange(len(s))))
        self.assertTrue(ic.inverse_vocab == inverse_vocab)
        self.assertTrue(ic.positions == positions)

    def test_indexed_characters_not_bow(self):
        s = 'Please, take your time'

        ic = IndexedCharacters(s, bow=False)

        self.assertTrue(np.array_equal(ic.as_np, np.array(list(s))))
        self.assertTrue(np.array_equal(ic.string_start, np.arange(len(s))))
        self.assertTrue(ic.inverse_vocab == list(s))
        self.assertTrue(np.array_equal(ic.positions, np.arange(len(s))))

    def test_indexed_string_regex(self):
        s = 'Please, take your time. Please'
        tokenized_string = np.array(
            ['Please', ', ', 'take', ' ', 'your', ' ', 'time', '. ', 'Please'])
        inverse_vocab = ['Please', 'take', 'your', 'time']
        start_positions = [0, 6, 8, 12, 13, 17, 18, 22, 24]
        positions = [[0, 8], [2], [4], [6]]
        indexed_string = IndexedString(s)

        self.assertTrue(np.array_equal(indexed_string.as_np, tokenized_string))
        self.assertTrue(np.array_equal(indexed_string.string_start, start_positions))
        self.assertTrue(indexed_string.inverse_vocab == inverse_vocab)
        self.assertTrue(np.array_equal(indexed_string.positions, positions))

    def test_indexed_string_callable(self):
        s = 'aabbccddaa'

        def tokenizer(string):
            return [string[i] + string[i + 1] for i in range(0, len(string) - 1, 2)]

        tokenized_string = np.array(['aa', 'bb', 'cc', 'dd', 'aa'])
        inverse_vocab = ['aa', 'bb', 'cc', 'dd']
        start_positions = [0, 2, 4, 6, 8]
        positions = [[0, 4], [1], [2], [3]]
        indexed_string = IndexedString(s, tokenizer)

        self.assertTrue(np.array_equal(indexed_string.as_np, tokenized_string))
        self.assertTrue(np.array_equal(indexed_string.string_start, start_positions))
        self.assertTrue(indexed_string.inverse_vocab == inverse_vocab)
        self.assertTrue(np.array_equal(indexed_string.positions, positions))

    def test_indexed_string_inverse_removing_tokenizer(self):
        s = 'This is a good movie. This, it is a great movie.'

        def tokenizer(string):
            return re.split(r'(?:\W+)|$', string)

        indexed_string = IndexedString(s, tokenizer)

        self.assertEqual(s, indexed_string.inverse_removing([]))

    def test_indexed_string_inverse_removing_regex(self):
        s = 'This is a good movie. This is a great movie'
        indexed_string = IndexedString(s)

        self.assertEqual(s, indexed_string.inverse_removing([]))


if __name__ == '__main__':
    unittest.main()
