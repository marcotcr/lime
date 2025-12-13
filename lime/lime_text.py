"""
Functions for explaining text classifiers.
"""
from functools import partial
import itertools
import json
import re

import numpy as np
import scipy as sp
import sklearn
from sklearn.utils import check_random_state

from . import explanation
from . import lime_base


class TextDomainMapper(explanation.DomainMapper):
    """Maps feature ids to words or word-positions"""

    def __init__(self, indexed_string):
        """Initializer.

        Args:
            indexed_string: lime_text.IndexedString, original string
        """
        self.indexed_string = indexed_string

    def map_exp_ids(self, exp, positions=False):
        """Maps ids to words or word-position strings.

        Args:
            exp: list of tuples [(id, weight), (id,weight)]
            positions: if True, also return word positions

        Returns:
            list of tuples (word, weight), or (word_positions, weight) if
            examples: ('bad', 1) or ('bad_3-6-12', 1)
        """
        if positions:
            exp = [('%s_%s' % (
                self.indexed_string.word(x[0]),
                '-'.join(
                    map(str,
                        self.indexed_string.string_position(x[0])))), x[1])
                   for x in exp]
        else:
            exp = [(self.indexed_string.word(x[0]), x[1]) for x in exp]
        return exp

    def visualize_instance_html(self, exp, label, div_name, exp_object_name,
                                text=True, opacity=True):
        """Adds text with highlighted words to visualization.

        Args:
             exp: list of tuples [(id, weight), (id,weight)]
             label: label id (integer)
             div_name: name of div object to be used for rendering(in js)
             exp_object_name: name of js explanation object
             text: if False, return empty
             opacity: if True, fade colors according to weight
        """
        if not text:
            return u''
        text = (self.indexed_string.raw_string()
                .encode('utf-8', 'xmlcharrefreplace').decode('utf-8'))
        text = re.sub(r'[<>&]', '|', text)
        exp = [(self.indexed_string.word(x[0]),
                self.indexed_string.string_position(x[0]),
                x[1]) for x in exp]
        all_occurrences = list(itertools.chain.from_iterable(
            [itertools.product([x[0]], x[1], [x[2]]) for x in exp]))
        all_occurrences = [(x[0], int(x[1]), x[2]) for x in all_occurrences]
        ret = '''
            %s.show_raw_text(%s, %d, %s, %s, %s);
            ''' % (exp_object_name, json.dumps(all_occurrences), label,
                   json.dumps(text), div_name, json.dumps(opacity))
        return ret


class IndexedString(object):
    """String with various indexes."""

    def __init__(self, raw_string, split_expression=r'\W+', bow=True,
                 mask_string=None):
        """Initializer.

        Args:
            raw_string: string with raw text in it
            split_expression: Regex string or callable. If regex string, will be used with re.split.
                If callable, the function should return a list of tokens.
            bow: if True, a word is the same everywhere in the text - i.e. we
                 will index multiple occurrences of the same word. If False,
                 order matters, so that the same word will have different ids
                 according to position.
            mask_string: If not None, replace words with this if bow=False
                if None, default value is UNKWORDZ
        """
        self.raw = raw_string
        self.mask_string = 'UNKWORDZ' if mask_string is None else mask_string

        if callable(split_expression):
            tokens = split_expression(self.raw)
            self.as_list = self._segment_with_tokens(self.raw, tokens)
            tokens = set(tokens)

            def non_word(string):
                return string not in tokens

        else:
            # with the split_expression as a non-capturing group (?:), we don't need to filter out
            # the separator character from the split results.
            splitter = re.compile(r'(%s)|$' % split_expression)
            self.as_list = [s for s in splitter.split(self.raw) if s]
            non_word = splitter.match

        self.as_np = np.array(self.as_list)
        self.string_start = np.hstack(
            ([0], np.cumsum([len(x) for x in self.as_np[:-1]])))
        vocab = {}
        self.inverse_vocab = []
        self.positions = []
        self.bow = bow
        non_vocab = set()
        for i, word in enumerate(self.as_np):
            if word in non_vocab:
                continue
            if non_word(word):
                non_vocab.add(word)
                continue
            if bow:
                if word not in vocab:
                    vocab[word] = len(vocab)
                    self.inverse_vocab.append(word)
                    self.positions.append([])
                idx_word = vocab[word]
                self.positions[idx_word].append(i)
            else:
                self.inverse_vocab.append(word)
                self.positions.append(i)
        if not bow:
            self.positions = np.array(self.positions)

    def raw_string(self):
        """Returns the original raw string"""
        return self.raw

    def num_words(self):
        """Returns the number of tokens in the vocabulary for this document."""
        return len(self.inverse_vocab)

    def word(self, id_):
        """Returns the word that corresponds to id_ (int)"""
        return self.inverse_vocab[id_]

    def string_position(self, id_):
        """Returns a np array with indices to id_ (int) occurrences"""
        if self.bow:
            return self.string_start[self.positions[id_]]
        else:
            return self.string_start[[self.positions[id_]]]

    def inverse_removing(self, words_to_remove):
        """Returns a string after removing the appropriate words.

        If self.bow is false, replaces word with UNKWORDZ instead of removing
        it.

        Args:
            words_to_remove: list of ids (ints) to remove

        Returns:
            original raw string with appropriate words removed.
        """
        mask = np.ones(self.as_np.shape[0], dtype='bool')
        mask[self.__get_idxs(words_to_remove)] = False
        if not self.bow:
            return ''.join(
                [self.as_list[i] if mask[i] else self.mask_string
                 for i in range(mask.shape[0])])
        return ''.join([self.as_list[v] for v in mask.nonzero()[0]])

    @staticmethod
    def _segment_with_tokens(text, tokens):
        """Segment a string around the tokens created by a passed-in tokenizer"""
        list_form = []
        text_ptr = 0
        for token in tokens:
            inter_token_string = []
            while not text[text_ptr:].startswith(token):
                inter_token_string.append(text[text_ptr])
                text_ptr += 1
                if text_ptr >= len(text):
                    raise ValueError("Tokenization produced tokens that do not belong in string!")
            text_ptr += len(token)
            if inter_token_string:
                list_form.append(''.join(inter_token_string))
            list_form.append(token)
        if text_ptr < len(text):
            list_form.append(text[text_ptr:])
        return list_form

    def __get_idxs(self, words):
        """Returns indexes to appropriate words."""
        if self.bow:
            return list(itertools.chain.from_iterable(
                [self.positions[z] for z in words]))
        else:
            return self.positions[words]


class IndexedCharacters(object):
    """String with various indexes."""

    def __init__(self, raw_string, bow=True, mask_string=None):
        """Initializer.

        Args:
            raw_string: string with raw text in it
            bow: if True, a char is the same everywhere in the text - i.e. we
                 will index multiple occurrences of the same character. If False,
                 order matters, so that the same word will have different ids
                 according to position.
            mask_string: If not None, replace characters with this if bow=False
                if None, default value is chr(0)
        """
        self.raw = raw_string
        self.as_list = list(self.raw)
        self.as_np = np.array(self.as_list)
        self.mask_string = chr(0) if mask_string is None else mask_string
        self.string_start = np.arange(len(self.raw))
        vocab = {}
        self.inverse_vocab = []
        self.positions = []
        self.bow = bow
        non_vocab = set()
        for i, char in enumerate(self.as_np):
            if char in non_vocab:
                continue
            if bow:
                if char not in vocab:
                    vocab[char] = len(vocab)
                    self.inverse_vocab.append(char)
                    self.positions.append([])
                idx_char = vocab[char]
                self.positions[idx_char].append(i)
            else:
                self.inverse_vocab.append(char)
                self.positions.append(i)
        if not bow:
            self.positions = np.array(self.positions)

    def raw_string(self):
        """Returns the original raw string"""
        return self.raw

    def num_words(self):
        """Returns the number of tokens in the vocabulary for this document."""
        return len(self.inverse_vocab)

    def word(self, id_):
        """Returns the word that corresponds to id_ (int)"""
        return self.inverse_vocab[id_]

    def string_position(self, id_):
        """Returns a np array with indices to id_ (int) occurrences"""
        if self.bow:
            return self.string_start[self.positions[id_]]
        else:
            return self.string_start[[self.positions[id_]]]

    def inverse_removing(self, words_to_remove):
        """Returns a string after removing the appropriate words.

        If self.bow is false, replaces word with UNKWORDZ instead of removing
        it.

        Args:
            words_to_remove: list of ids (ints) to remove

        Returns:
            original raw string with appropriate words removed.
        """
        mask = np.ones(self.as_np.shape[0], dtype='bool')
        mask[self.__get_idxs(words_to_remove)] = False
        if not self.bow:
            return ''.join(
                [self.as_list[i] if mask[i] else self.mask_string
                 for i in range(mask.shape[0])])
        return ''.join([self.as_list[v] for v in mask.nonzero()[0]])

    def __get_idxs(self, words):
        """Returns indexes to appropriate words."""
        if self.bow:
            return list(itertools.chain.from_iterable(
                [self.positions[z] for z in words]))
        else:
            return self.positions[words]


class LimeTextExplainer(object):
    """Explains text classifiers.
       Currently, we are using an exponential kernel on cosine distance, and
       restricting explanations to words that are present in documents."""

    def __init__(self,
                 kernel_width=25,
                 kernel=None,
                 verbose=False,
                 class_names=None,
                 feature_selection='auto',
                 split_expression=r'\W+',
                 bow=True,
                 mask_string=None,
                 random_state=None,
                 char_level=False,
                 lang="en"):
        """Init function.

        Args:
            kernel_width: kernel width for the exponential kernel.
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            class_names: list of class names, ordered according to whatever the
                classifier is using. If not present, class names will be '0',
                '1', ...
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            split_expression: Regex string or callable. If regex string, will be used with re.split.
                If callable, the function should return a list of tokens.
            bow: if True (bag of words), will perturb input data by removing
                all occurrences of individual words or characters.
                Explanations will be in terms of these words. Otherwise, will
                explain in terms of word-positions, so that a word may be
                important the first time it appears and unimportant the second.
                Only set to false if the classifier uses word order in some way
                (bigrams, etc), or if you set char_level=True.
            mask_string: String used to mask tokens or characters if bow=False
                if None, will be 'UNKWORDZ' if char_level=False, chr(0)
                otherwise.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
            char_level: an boolean identifying that we treat each character
                as an independent occurence in the string
        """

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.random_state = check_random_state(random_state)
        self.base = lime_base.LimeBase(kernel_fn, verbose,
                                       random_state=self.random_state)
        self.class_names = class_names
        self.vocabulary = None
        self.feature_selection = feature_selection
        self.bow = bow
        self.mask_string = mask_string
        self.split_expression = split_expression
        self.char_level = char_level
        self.lang = lang

        # after parsing args in __init__, before using split_expression
        if self.lang == "jp" and not char_level:
            try:
                from .japanese import mecab_unidic_split, active_japanese_tokenizer
                # Use Sudachi when available; otherwise allow fallback tokens
                self.split_expression = mecab_unidic_split
            except Exception:
                # Keep regex split if japanese package import fails
                pass

    def explain_instance(self,
                         text_instance,
                         classifier_fn,
                         labels=(1,),
                         top_labels=None,
                         num_features=10,
                         num_samples=5000,
                         distance_metric='cosine',
                         model_regressor=None):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly hiding features from
        the instance (see __data_labels_distance_mapping). We then learn
        locally weighted linear models on this neighborhood data to explain
        each of the classes in an interpretable way (see lime_base.py).

        Args:
            text_instance: raw text string to be explained.
            classifier_fn: classifier prediction probability function, which
                takes a list of d strings and outputs a (d, k) numpy array with
                prediction probabilities, where k is the number of classes.
                For ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for sample weighting,
                defaults to cosine similarity
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()
        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """

        indexed_string = (IndexedCharacters(
            text_instance, bow=self.bow, mask_string=self.mask_string)
                          if self.char_level else
                          IndexedString(text_instance, bow=self.bow,
                                        split_expression=self.split_expression,
                                        mask_string=self.mask_string))
        domain_mapper = TextDomainMapper(indexed_string)
        data, yss, distances = self.__data_labels_distances(
            indexed_string, classifier_fn, num_samples,
            distance_metric=distance_metric)
        if self.class_names is None:
            self.class_names = [str(x) for x in range(yss[0].shape[0])]
        ret_exp = explanation.Explanation(domain_mapper=domain_mapper,
                                          class_names=self.class_names,
                                          random_state=self.random_state)
        ret_exp.predict_proba = yss[0]
        if top_labels:
            labels = np.argsort(yss[0])[-top_labels:]
            ret_exp.top_labels = list(labels)
            ret_exp.top_labels.reverse()
        for label in labels:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score[label],
             ret_exp.local_pred[label]) = self.base.explain_instance_with_data(
                data, yss, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)
        return ret_exp

    def explain_instance_plain_text(self, exp, label=None, n_words=3):
        """Generate a short plain-English summary for a text explanation.

        Args:
            exp: an Explanation object returned by `explain_instance`.
            label: integer label index or label name. If None, use top label from
                the explanation (if available) or 0.
            n_words: number of top words to include in the summary.

        Returns:
            str: A one-sentence English summary describing the label and its
                 most important words.
        """
        # Determine label index
        if label is None:
            if hasattr(exp, 'top_labels') and getattr(exp, 'top_labels'):
                label_idx = exp.top_labels[0]
            else:
                label_idx = 0
        else:
            # allow passing label name or index
            if isinstance(label, int):
                label_idx = label
            else:
                try:
                    label_idx = exp.class_names.index(label)
                except Exception:
                    try:
                        label_idx = int(label)
                    except Exception:
                        label_idx = 0

        # Retrieve local explanation for the label
        features = []
        try:
            features = exp.local_exp.get(label_idx, []) if isinstance(exp.local_exp, dict) else exp.local_exp[label_idx]
        except Exception:
            # fallback: try to access as attribute or index
            try:
                features = exp.local_exp[label_idx]
            except Exception:
                features = []

        # Keep top n_words by absolute weight
        if features:
            try:
                # features is list of (feature_id, weight)
                features_sorted = sorted(features, key=lambda x: -abs(x[1]))[:n_words]
            except Exception:
                features_sorted = features[:n_words]
        else:
            features_sorted = []

        # Map feature ids to words using the domain mapper
        words = []
        try:
            mapped = exp.domain_mapper.map_exp_ids(features_sorted, positions=False)
            words = [w for w, _ in mapped]
        except Exception:
            try:
                words = [str(x[0]) for x in features_sorted]
            except Exception:
                words = []

        label_name = exp.class_names[label_idx] if getattr(exp, 'class_names', None) is not None else str(label_idx)

        if len(words) == 0:
            return f"In this text, the model {label_name} did not return any explanatory words."

        quoted = ', '.join(words)
        return f'In this text, the overall probability we can see that the model {label_name} is characterized by the words such as "{quoted}".'

    def __data_labels_distances(self,
                                indexed_string,
                                classifier_fn,
                                num_samples,
                                distance_metric='cosine'):
        """Generates a neighborhood around a prediction.

        Generates neighborhood data by randomly removing words from
        the instance, and predicting with the classifier. Uses cosine distance
        to compute distances between original and perturbed instances.
        Args:
            indexed_string: document (IndexedString) to be explained,
            classifier_fn: classifier prediction probability function, which
                takes a string and outputs prediction probabilities. For
                ScikitClassifier, this is classifier.predict_proba.
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for sample weighting,
                defaults to cosine similarity.


        Returns:
            A tuple (data, labels, distances), where:
                data: dense num_samples * K binary matrix, where K is the
                    number of tokens in indexed_string. The first row is the
                    original instance, and thus a row of ones.
                labels: num_samples * L matrix, where L is the number of target
                    labels
                distances: cosine distance between the original instance and
                    each perturbed instance (computed in the binary 'data'
                    matrix), times 100.
        """

        def distance_fn(x):
            return sklearn.metrics.pairwise.pairwise_distances(
                x, x[0], metric=distance_metric).ravel() * 100

        doc_size = indexed_string.num_words()
        sample = self.random_state.randint(1, doc_size + 1, num_samples - 1)
        data = np.ones((num_samples, doc_size))
        data[0] = np.ones(doc_size)
        features_range = range(doc_size)
        inverse_data = [indexed_string.raw_string()]
        for i, size in enumerate(sample, start=1):
            inactive = self.random_state.choice(features_range, size,
                                                replace=False)
            data[i, inactive] = 0
            inverse_data.append(indexed_string.inverse_removing(inactive))
        labels = classifier_fn(inverse_data)
        # Ensure labels are a numpy array of shape (num_samples, n_classes)
        labels = np.asarray(labels)
        distances = distance_fn(sp.sparse.csr_matrix(data))
        return data, labels, distances


def generate_sentence_for_feature(word, weight, class_name):
    """
    Converts a single LIME word-weight pair into a natural-language sentence.
    """
    direction = "increased" if weight > 0 else "decreased"
    weight_abs = abs(weight)

    if weight_abs > 0.10:
        strength = "strongly"
    elif weight_abs > 0.05:
        strength = "moderately"
    else:
        strength = "slightly"

    return (
        f'The word "{word}" {strength} {direction} '
        f'the predicted probability of {class_name} (weight = {weight:.3f}).'
    )


def summarize_lime_explanation(explanation_obj, class_idx=1):
    """
    Takes a LIME explanation object and returns a list of natural-language sentences.
    Works with LimeTextExplainer().
    """
    # Extract the explanation for the class of interest
    try:
        exp_list = explanation_obj.as_list(label=class_idx)
    except Exception:
        # Fallback: build list from local_exp
        local = explanation_obj.local_exp.get(class_idx, []) if isinstance(explanation_obj.local_exp, dict) else explanation_obj.local_exp[class_idx]
        # Map ids to words using the domain mapper
        mapped = explanation_obj.domain_mapper.map_exp_ids(local, positions=False)
        exp_list = mapped

    sentences = []
    class_names = getattr(explanation_obj, 'class_names', None)
    class_name = class_names[class_idx] if class_names and class_idx < len(class_names) else str(class_idx)

    # Generate sentence per feature
    for word, weight in exp_list:
        sentences.append(generate_sentence_for_feature(word, weight, class_name))

    if not exp_list:
        return sentences

    # Summary overview sentence
    highest_word, highest_weight = max(exp_list, key=lambda x: abs(x[1]))
    overview_sentence = (
        f'Overall, "{highest_word}" had the largest impact on the prediction '
        f'with a weight of {highest_weight:.3f}, making it the most influential term.'
    )

    return [overview_sentence] + sentences


def print_lime_narrative(explanation_obj, class_idx=1):
    """
    Prints a clean, readable explanation block.
    """
    narrative = summarize_lime_explanation(explanation_obj, class_idx=class_idx)

    print("\nNatural-Language Explanation of LIME Output")
    print("--------------------------------------------------")
    for sent in narrative:
        print("• " + sent)


def generate_sentence_for_feature_jp(word, weight, class_name):
    """日本語の1特徴語と重みから、自然言語の文を生成します。"""
    direction = "上げました" if weight > 0 else "下げました"
    weight_abs = abs(weight)

    if weight_abs > 0.10:
        strength = "大きく"
    elif weight_abs > 0.05:
        strength = "中程度に"
    else:
        strength = "わずかに"

    return (
        f'単語「{word}」は{strength}{direction} '
        f'クラス「{class_name}」の予測確率（重み = {weight:.3f}）。'
    )


def summarize_lime_explanation_jp(explanation_obj, class_idx=1):
    """LIMEの説明オブジェクトから日本語のプレーン文を生成します。"""
    probs = getattr(explanation_obj, 'predict_proba', None)
    class_names = getattr(explanation_obj, 'class_names', None)
    if probs is None:
        return ["予測確率が取得できませんでした。"]

    order = np.argsort(probs)
    class_1_idx = int(order[-1])
    class_2_idx = int(order[-2]) if len(probs) > 1 else (1 - class_1_idx)

    class_1 = class_names[class_1_idx] if class_names else str(class_1_idx)
    class_2 = class_names[class_2_idx] if class_names else str(class_2_idx)

    p0 = float(probs[class_1_idx])
    p1 = float(probs[class_2_idx])

    def _get_feats(idx):
        try:
            return explanation_obj.local_exp.get(idx, []) if isinstance(explanation_obj.local_exp, dict) else explanation_obj.local_exp[idx]
        except Exception:
            return []

    feats_1 = _get_feats(class_1_idx)
    feats_2 = _get_feats(class_2_idx)

    mapper = getattr(explanation_obj, 'domain_mapper', None)
    def _map(feats):
        if not feats:
            return []
        try:
            return mapper.map_exp_ids(feats, positions=False)
        except Exception:
            return [(str(fid), w) for fid, w in feats]

    mapped_1 = _map(feats_1)
    mapped_2 = _map(feats_2)

    pos_1 = sorted([(w, wt) for (w, wt) in mapped_1 if wt > 0], key=lambda x: -abs(x[1]))
    pos_2 = sorted([(w, wt) for (w, wt) in mapped_2 if wt > 0], key=lambda x: -abs(x[1]))

    def _take(items, n):
        take = items[:n]
        if len(take) < n:
            take += [("—", 0.0)] * (n - len(take))
        return take

    top3_1 = _take(pos_1, 3)
    next3_1 = _take(pos_1[3:], 3)
    top3_2 = _take(pos_2, 3)

    sent1 = (
        f"このインスタンスは{p0:.3f}対{p1:.3f}で{class_1}と分類されました。"
        f"{class_1}への分類に最も強い影響を与えた言葉は{top3_1[0][0]}, {top3_1[1][0]}, {top3_1[2][0]}で、"
        f"それぞれの重みは{top3_1[0][1]:.3f}, {top3_1[1][1]:.3f}, {top3_1[2][1]:.3f}となっています。"
    )

    sent2 = (
        f"他に{class_1}への分類の確率を上げた言葉として{next3_1[0][0]} (重み = {next3_1[0][1]:.3f})、"
        f"{next3_1[1][0]} (重み = {next3_1[1][1]:.3f})、{next3_1[2][0]} (重み = {next3_1[2][1]:.3f})などが挙げられます。"
        f"{class_2}への分類への確率を上げた言葉として、{top3_2[0][0]} (重み = {top3_2[0][1]:.3f})、"
        f"{top3_2[1][0]} (重み = {top3_2[1][1]:.3f})、{top3_2[2][0]} (重み = {top3_2[2][1]:.3f})などが挙げられます。"
    )

    return [sent1, sent2]


def print_lime_narrative_jp(explanation_obj, class_idx=1):
    """日本語の説明ブロックを整形して出力します。"""
    narrative = summarize_lime_explanation_jp(explanation_obj, class_idx=class_idx)

    print("\nLIME出力の自然言語による説明")
    print("--------------------------------------------------")
    for sent in narrative:
        print("・ " + sent)
