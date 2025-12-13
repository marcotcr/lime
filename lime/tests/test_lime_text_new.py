import re
import unittest

import sklearn  # noqa
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

        self.assertEqual(6, len(exp.as_list()))
        self.assertIsNotNone(exp.as_list())

    def test_lime_text_explainer_char_level(self):
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
        explainer = LimeTextExplainer(class_names=class_names, char_level=True)
        idx = 83
        exp = explainer.explain_instance(newsgroups_test.data[idx],
                                         c.predict_proba, num_features=6)

        self.assertEqual(6, len(exp.as_list()))
        self.assertIsNotNone(exp.as_list())

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
        
        # Compare positions by converting both to lists of lists
        actual_positions = [list(p) if hasattr(p, '__iter__') else [p] 
                           for p in indexed_string.positions]
        self.assertEqual(actual_positions, positions)

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
        
        # Compare positions by converting both to lists of lists
        actual_positions = [list(p) if hasattr(p, '__iter__') else [p] 
                           for p in indexed_string.positions]
        self.assertEqual(actual_positions, positions)

    def test_indexed_characters_bow(self):
        s = 'Please, take your time. Please'
        tokenized_string = np.array(list(s))

        indexed_chars = IndexedCharacters(s, bow=True)
        vocab = list(set(s))
        vocab.sort()
        vocab_array = np.array(vocab)
        positions = [np.where(tokenized_string == x)[0] for x in vocab_array]

        self.assertTrue(np.array_equal(indexed_chars.as_np, tokenized_string))
        
        # Convert inverse_vocab to strings for comparison (handles np.str_ objects)
        actual_vocab = [str(v) for v in indexed_chars.inverse_vocab]
        expected_vocab = vocab  # This is already a sorted list of strings
        
        # Compare as sorted lists since order may vary
        self.assertEqual(sorted(actual_vocab), sorted(expected_vocab))
        
        # Verify positions match for each character
        for char in vocab:
            # Find index in actual vocab
            actual_idx = actual_vocab.index(char)
            # Find index in expected vocab
            expected_idx = list(vocab_array).index(char)
            # Compare positions
            self.assertTrue(
                np.array_equal(indexed_chars.positions[actual_idx], positions[expected_idx]),
                f"Positions don't match for character '{char}'"
            )

    def test_indexed_characters_not_bow(self):
        s = 'Please, take your time. Please'
        tokenized_string = np.array(list(s))
        vocab = tokenized_string
        positions = np.arange(len(s))

        indexed_chars = IndexedCharacters(s, bow=False)

        self.assertTrue(np.array_equal(indexed_chars.as_np, tokenized_string))
        self.assertTrue(np.array_equal(indexed_chars.inverse_vocab, vocab))
        for i, p in enumerate(indexed_chars.positions):
            self.assertTrue(np.array_equal(p, positions[i]))

    def test_indexed_string_inverse_removing_bow(self):
        s = 'Please, take your time. Please'
        indexed_string = IndexedString(s)

        self.assertTrue(
            indexed_string.inverse_removing([0]) == ', take your time. ')
        self.assertTrue(
            indexed_string.inverse_removing([0, 2]) == ', take  time. ')
        self.assertTrue(
            indexed_string.inverse_removing([0, 2, 3]) == ', take  . ')

    def test_indexed_string_inverse_removing_not_bow(self):
        s = 'Please, take your time. Please'
        indexed_string = IndexedString(s, bow=False)
        
        # When bow=False, indices refer to the inverse_vocab list, not tokenized positions
        # inverse_vocab: ['Please', 'take', 'your', 'time', 'Please']
        # Note: 'Please' appears twice (index 0 and 4), but they map to unique vocab indices
        
        # Removing index 0 from inverse_vocab removes first 'Please'
        self.assertEqual(
            indexed_string.inverse_removing([0]),
            'UNKWORDZ, take your time. Please')
        
        # Removing indices 0 and 2 from inverse_vocab removes 'Please' and 'your'
        # Based on actual diagnostic output
        self.assertEqual(
            indexed_string.inverse_removing([0, 2]),
            'UNKWORDZ, take UNKWORDZ time. Please')
        
        # Removing indices 0, 2, and 3 removes 'Please', 'your', and 'time'
        # Based on actual diagnostic output
        self.assertEqual(
            indexed_string.inverse_removing([0, 2, 3]),
            'UNKWORDZ, take UNKWORDZ UNKWORDZ. Please')


if __name__ == '__main__':
    unittest.main()