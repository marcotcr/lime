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
        classes,
        kernel_width=0.25,
        kernel=None,
        feature_selection="auto",
        verbose=False,
        random_state=None,
        split_expression=r"\W+",
        bow=True,
        mask_string=None,
        char_level=False,
    ) -> None:
        """Initial Image Text Class

        Args:
            classes (List): list of class names, ordered according to whatever the
                classifier is using.
            kernel_width (float, optional):  kernel width for the exponential kernel. Defaults to 0.25.
            kernel (fun, optional): similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel. Defaults to None.
            feature_selection (str, optional): feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options doesyt  . Defaults to "auto".
            verbose (bool, optional): if true, print local prediction values from linear model. Defaults to False.
            random_state (integer or numpy.RandomState, optional): will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed. Defaults to None.
            split_expression (regexp, optional): Regex string or callable. If regex string, will be used with re.split.
                If callable, the function should return a list of tokens. Defaults to r"\W+".
            bow (bool, optional): if True (bag of words), will perturb input data by removing
                all occurrences of individual words or characters.
                Explanations will be in terms of these words. Otherwise, will
                explain in terms of word-positions, so that a word may be
                important the first time it appears and unimportant the second.
                Only set to false if the classifier uses word order in some way
                (bigrams, etc), or if you set char_level=True. Defaults to True.
            mask_string (str, optional): String used to mask tokens or characters if bow=False
                if None, will be 'UNKWORDZ' if char_level=False, chr(0)
                otherwise. Defaults to None.
            char_level (bool, optional): an boolean identifying that we treat each character
                as an independent occurence in the string. Defaults to False.
        """
        self.text_explainer = LimeTextExplainer()
        self.image_explainer = LimeImageExplainer()
        kernel_width = float(kernel_width)
        kernel_fn = kernel
        if kernel is None:

            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d**2) / kernel_width**2))

            kernel_fn = partial(kernel, kernel_width=kernel_width)
        self.random_state = check_random_state(random_state)
        self.feature_selection = feature_selection
        self.base = LimeBase(kernel_fn, verbose, random_state=self.random_state)
        self.random_seed = self.random_state.randint(0, high=1000)
        self.class_names = classes
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
        """Explain Image And Text Togather For Fusion Models!

        Args:
            img (ndarray): Image that will be used to make prediction.
            sentence (str): Text that will be used to make prediction.
            predict_fun (fun): classifier prediction probability function, which
                takes an image and text and outputs prediction probabilities.
            num_features: maximum number of features present in explanation. Defaults to 100000.
            num_samples: size of the neighborhood to learn the linear model. Defualts to 15.
            top_labels (int, optional): if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter. Defaults to 5.
            model_regressor (sklearn regressor, optional): sklearn regressor to use in explanation. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit(). Defaults
            to Ridge regression in LimeBase if None.
            text_preprocessing (fun, optional): Text Preprocessing Function That apply on Text. Defaults to None.
            image_preprocessing (fun, optional): Image Preprocessing Function That apply on Image. Defaults to None.

        Returns:
            tuple: Image Explanation, Text Explanation
        """
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

        labels = predict_fun(imgs, sentences)

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

#DEMO ************************************************************
# Just For Testing !!!!
def pred(imgs, sentences):
    labels = np.random.random((len(imgs), 3))
    return labels


# Just For Testing !!!!
if __name__ == "__main__":
    image_path = "../../man5.jpg"
    sentenct = "Happy Smiley Face Man In Blue Suit With White Background"
    CLASS_NAMES = ["Positive", "Negative", "Neutral"]
    image_text_lime = ITLIME(classes=CLASS_NAMES)
    img = imread(image_path)
    img_exp, text_exp = image_text_lime.explain_instance(
        img,
        sentenct,
        predict_fun=pred,
        num_samples=15,
    )
    fig = text_exp.as_pyplot_figure(0)
    plt.figure()
    temp_img, mask = img_exp.get_image_and_mask(img_exp.top_labels[0])
    plt.imshow(mark_boundaries(temp_img, mask))
    plt.show()
#*****************************************************************