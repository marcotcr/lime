Quickstart
==========

Explaining an image
-------------------

``classifier_fn`` takes a batch of images of shape ``(n, h, w, 3)`` and
returns an ``(n, n_classes)`` array of class probabilities, as in the original
LIME. Stratified sampling is enabled with a single flag:

.. code-block:: python

   from lime import lime_image

   explainer = lime_image.LimeImageExplainer(random_state=1234)
   explanation = explainer.explain_instance(
       image,                    # H x W x 3 numpy array
       classifier_fn,            # batch of images -> class probabilities
       top_labels=1,
       num_samples=1000,         # n, size of the synthetic neighborhood
       use_stratification=True,  # the method of the paper
   )

   label = explanation.top_labels[0]
   img, mask = explanation.get_image_and_mask(label, positive_only=True,
                                              num_features=5, hide_rest=True)

With ``use_stratification=False`` (the default), the explainer behaves exactly
like the original LIME Image with Monte Carlo sampling. You can run both on the
same image and compare the results.

Controlling the number of superpixels
-------------------------------------

The advantage of stratified sampling grows with the number of superpixels
:math:`k` (see :doc:`method`). The paper uses quick shift with
``kernel_size=4, ratio=0.2`` and tunes ``max_dist`` to reach
:math:`k \in \{50, 100, 150, 200\}`. You can pass the segmentation explicitly:

.. code-block:: python

   from lime.wrappers.scikit_image import SegmentationAlgorithm

   segmenter = SegmentationAlgorithm("quickshift", kernel_size=4,
                                     max_dist=7, ratio=0.2, random_seed=0)
   explanation = explainer.explain_instance(image, classifier_fn,
                                            top_labels=1,
                                            segmentation_fn=segmenter,
                                            use_stratification=True)
   k = len(set(explanation.segments.ravel()))   # number of superpixels

Smaller ``max_dist`` values give more superpixels.

.. _diagnostics:

Diagnosing an explanation: CV and RC
------------------------------------

The paper uses two diagnostics to detect "confused" explanations (see
:ref:`metrics`).

**Coefficient of variation** :math:`CV(\beta)` of the superpixel importances.
Values close to zero mean that the explanation is nearly uniform:

.. code-block:: python

   import numpy as np

   beta = np.array([w for _, w in explanation.local_exp[label]])
   cv = beta.std() / beta.mean()

**Range coverage** :math:`RC(Y)` of the black-box scores in the synthetic
neighborhood. ``explain_instance`` does not return the neighborhood, so
generate it with
:meth:`~lime.lime_image.LimeImageExplainer.data_labels`, using the same
mean-filled replacement image that ``explain_instance`` builds by default:

.. code-block:: python

   segments = explanation.segments
   fudged = image.copy()
   for s in np.unique(segments):
       fudged[segments == s] = image[segments == s].mean(axis=0)

   data, Y, adj = explainer.data_labels(image, fudged, segments,
                                        classifier_fn, num_samples=1000,
                                        use_stratification=True,
                                        progress_bar=False)
   y = Y[:, label]          # y[0] is f(xi): row 0 is the unmasked image
   rc = (np.percentile(y, 99) - np.percentile(y, 1)) / y[0]

Values of :math:`RC(Y)` close to 0 mean that the neighborhood never left a
narrow band of scores, which leaves the regressor nothing to fit.

More examples
-------------

All experiments of the paper, with notebooks and pinned requirements, are in
`lime-stratified-examples <https://github.com/rashidrao-pk/lime-stratified-examples>`_.
