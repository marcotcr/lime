import numpy as np
from skimage.io import imread
import sklearn
from sklearn.utils import check_random_state
from lime_base import LimeBase
from lime_text import LimeTextExplainer
from lime_image import ImageExplanation, LimeImageExplainer
import matplotlib.pyplot as plt
from functools import partial
from skimage.segmentation import mark_boundaries

class ITLIME:
    """
    Image Text Lime CLass
    """

    def __init__(self,
        kernel_width=0.25,
        kernel=None,
        feature_selection='auto',
        kernal_fn=None,
        verbose=False,
        random_state=None,
        random_seed=None) -> None:
        self.text_explainer = LimeTextExplainer()
        self.image_explainer = LimeImageExplainer()
        kernel_width = float(kernel_width)

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d**2) / kernel_width**2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)
        self.random_state = check_random_state(random_state)
        self.feature_selection = feature_selection
        self.base = LimeBase(
            kernel_fn, verbose, random_state=self.random_state
        )
        if random_seed is None:
            self.random_seed = self.random_state.randint(0, high=1000)

    def explain_instance(
        self,
        img,
        sentence,
        num_features=100000,
        num_samples=15,
        top_labels=5,
        model_regressor=None,
    ):
        segments = self.image_explainer.get_segments(img)
        data, imgs = self.image_explainer.generate_imgs(img, segments, num_samples)
        labels = np.random.random((num_samples, 3))
        distances = sklearn.metrics.pairwise_distances(
            data, data[0].reshape(1, -1), metric="cosine"
        ).ravel()
        
        ret_exp = ImageExplanation(img, segments)
        if top_labels:
            top = np.argsort(labels[0])[-top_labels:]
            ret_exp.top_labels = list(top)
            ret_exp.top_labels.reverse()
        for label in top:
            (
                ret_exp.intercept[label],
                ret_exp.local_exp[label],
                ret_exp.score[label],
                ret_exp.local_pred[label],
            ) = self.base.explain_instance_with_data(
                data,
                labels,
                distances,
                label,
                num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection,
            )
        temp, mask = ret_exp.get_image_and_mask(ret_exp.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
        plt.imshow(mark_boundaries(temp , mask))
        # _, axs = plt.subplots(2, 2, figsize=(12, 12))
        # axs = axs.flatten()
        # for ax, img in zip(axs, imgs):
        #     ax.imshow(img)
        plt.show()

    def predict(self, input):
        pass



