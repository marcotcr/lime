import numpy as np
from skimage.io import imread
from skimage.segmentation import mark_boundaries
import sklearn
from sklearn.utils import check_random_state
from explanation import Explanation
from lime_base import LimeBase
from lime_text import (
    IndexedCharacters,
    IndexedString,
    LimeTextExplainer,
    TextDomainMapper,
)
from lime_image import ImageExplanation, LimeImageExplainer
import matplotlib.pyplot as plt
from functools import partial
import scipy as sp

class ITLIME:
    """
    Image Text Lime CLass
    """

    def __init__(
        self,
        class_names,
        kernel_width=0.25,
        kernel=None,
        feature_selection="auto",
        kernel_fn=None,
        verbose=False,
        random_state=None,
        random_seed=None,
        split_expression=r"\W+",
        bow=True,
        mask_string=None,
        char_level=False,
    ) -> None:

        self.text_explainer = LimeTextExplainer()
        self.image_explainer = LimeImageExplainer()
        kernel_width = float(kernel_width)
        kernel_fn1 = kernel_fn

        if kernel is None:

            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d**2) / kernel_width**2))

            kernel_fn1 = partial(kernel, kernel_width=kernel_width)

        self.random_state = check_random_state(random_state)
        self.feature_selection = feature_selection
        self.base = LimeBase(kernel_fn1, verbose, random_state=self.random_state)
        if random_seed is None:
            self.random_seed = self.random_state.randint(0, high=1000)
        self.class_names = class_names
        self.vocabulary = None
        self.feature_selection = feature_selection
        self.bow = bow
        self.mask_string = mask_string
        self.split_expression = split_expression
        self.char_level = char_level

    def __calc_dist__text(self, data, metric="cosine"):
        data = sp.sparse.csr_matrix(data)
        return (
            sklearn.metrics.pairwise.pairwise_distances(
                data, data[0], metric=metric
            ).ravel()
            * 100
        )

    def __calc_dist__img(self, data, metric="cosine"):
        return sklearn.metrics.pairwise_distances(
            data, data[0].reshape(1, -1), metric=metric
        ).ravel()

    def explain_instance(
        self,
        img,
        sentence,
        predict_fun,
        num_features=100000,
        num_samples=15,
        top_labels=5,
        model_regressor=None,
        text_preprocessing=None,
        image_preprocessing=None,
    ):
        sentence = text_preprocessing(sentence) if text_preprocessing else sentence
        img = image_preprocessing(img) if image_preprocessing else img

        indexed_string = (
            IndexedCharacters(sentence, bow=self.bow, mask_string=self.mask_string)
            if self.char_level
            else IndexedString(
                sentence,
                bow=self.bow,
                split_expression=self.split_expression,
                mask_string=self.mask_string,
            )
        )
        domain_mapper = TextDomainMapper(indexed_string)
        text_data, sentences = self.text_explainer.generat_text_data(
            indexed_string, num_samples
        )
        text_distances = self.__calc_dist__text(text_data)

        segments = self.image_explainer.get_segments(img)
        imgs_data, imgs = self.image_explainer.generate_imgs(img, segments, num_samples)
        image_distances = self.__calc_dist__img(imgs_data)

        labels = predict_fun(imgs,sentences)

        img_exp = ImageExplanation(img, segments)
        text_exp = Explanation(
            domain_mapper=domain_mapper,
            class_names=self.class_names,
            random_state=self.random_state,
        )
        text_exp.predict_proba = labels[0]
        if top_labels:
            top = np.argsort(labels[0])[-top_labels:]
            text_exp.top_labels = list(top)
            text_exp.top_labels.reverse()
            img_exp.top_labels = list(top)
            img_exp.top_labels.reverse()

        for label in top:
            (
                text_exp.intercept[label],
                text_exp.local_exp[label],
                text_exp.score[label],
                text_exp.local_pred[label],
            ) = self.base.explain_instance_with_data(
                text_data,
                labels,
                text_distances,
                label,
                num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection,
            )

        for label in top:
            (
                img_exp.intercept[label],
                img_exp.local_exp[label],
                img_exp.score[label],
                img_exp.local_pred[label],
            ) = self.base.explain_instance_with_data(
                imgs_data,
                labels,
                image_distances,
                label,
                num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection,
            )
        return img_exp, text_exp


