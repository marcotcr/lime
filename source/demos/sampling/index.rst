Inside LIME Stratified: Where the Samples Land
==============================================

Why does LIME Image sometimes return a flat, meaningless explanation, and what does stratified
sampling change? This demo answers with the paper's formulas and with the library itself, run on
the five "low CV" photos of the paper's Fig. 7 with the same ResNet50. Monte Carlo sampling and
stratified sampling are always shown side by side.

.. lime-stratified-demo:: sampling
   :anchor: sampling-demo
   :fallback: demos/sampling/fallback.png
   :alt: The paper's Fig. 7 recomputed: for five photos, the Monte Carlo and the stratified LIME explanations at k = 50 and k = 200.

What you see
------------

**1 · Where samples land.** LIME Image explains a photo split into :math:`k` superpixels by
asking the model about random masks :math:`x \in \{0,1\}^k`. Each mask keeps or removes each
superpixel. The original LIME keeps each superpixel with probability 0.5 (Eq. 1 of the paper), so
the number of kept superpixels :math:`|x|` follows a binomial distribution, drawn in orange. With
realistic values of :math:`k`, almost every mask keeps about half of the photo, and the masks
close to the photo (the shaded band) practically never occur. Stratified sampling draws
:math:`q \sim U(0,1)` first and then keeps each superpixel with probability :math:`q`, so every
:math:`|x|` is equally likely (in green). Press ▶ to draw 1000 masks with each sampler. Move the
slider to change :math:`k`, and switch to the log scale to see how fast the binomial tails vanish.
Point at the chart to read each stratum's adjustment factor
:math:`\mathrm{adj}(i) = (k+1)\binom{k}{i}/2^k` (Eq. 12). It scales the sample's distance to the
photo before LIME's kernel (see :doc:`../../method`).

**2 · The neighbourhood.** The 1000 perturbed photos that each sampler generated for the selected
photo and segmentation, and the model's probability for each of them, side by side. This is the
"dependent variable distribution" of the paper (Fig. 1F, 4C, 5C and 7), drawn the same way:
:math:`f(\xi_x)` against the share of kept superpixels :math:`|x|/k`, the photo itself as a
magenta cross and :math:`f(\xi)` as a dashed green line. Point at a dot to see that perturbed
photo: removed superpixels are filled with their mean colour, the library's default. Monte
Carlo's dots crowd around :math:`|x|/k = 0.5`, where the model has usually lost the object, so
their scores are squashed near zero and the range coverage :math:`RC(Y)` (Eq. 7) is small.
Stratified sampling's dots spread from the empty photo to the full one, and cover the whole range
:math:`[0, f(\xi)]`.

**3 · The explanation.** One row of the paper's Fig. 7 for the selected photo and :math:`k`: for
each sampler, *CV | RC* above, then the explanation :math:`\beta` in the paper's colours (red
raises the probability, blue lowers it, white is zero, in a symmetric scale per heatmap) next to
the scores of its neighbourhood. A *confused* explanation, with :math:`CV(\beta) < 1` (in red),
gives every superpixel about the same importance and points at no region. You can also show
:math:`\beta` over the photo, or the five superpixels that ``get_image_and_mask`` would select.
Point at a superpixel to compare its importance under the two samplers. The table summarizes all
four segmentations, with how much the two explanations agree: the Spearman rank correlation of
:math:`\beta` and :math:`\widehat{\beta}`, and the superpixels shared by their top-5 regions.

**4 · All photos (Fig. 7).** The low-CV rows of the paper's Fig. 7, recomputed: every photo,
Monte Carlo and stratified, at :math:`k \approx 50` and :math:`k \approx 200`. Click a row to
explore that photo in the other tabs.

What the five photos show
-------------------------

On these photos the difference between the two samplers is large:

- **Monte Carlo is confused in 17 of the 20 settings** (5 photos × 4 values of :math:`k`), with
  :math:`CV(\beta) < 1`. **Stratified sampling is confused in none**: its :math:`CV(\widehat{\beta})`
  is between 1.71 and 5.06.
- Monte Carlo's neighbourhood covers only 0.7% to 33% of the score range at :math:`k \ge 100`
  (:math:`RC(Y)` from 0.007 to 0.44). Stratified sampling covers all of it
  (:math:`RC(\widehat{Y})` from 0.99 to 1.71).
- The surrogate model fits better with stratified sampling in all 20 settings. For example,
  :math:`R^2` on the wardrobe at :math:`k = 50` rises from 0.11 to 0.38.
- The two explanations are **not** the same. Their rank correlation falls from 0.15–0.79 at
  :math:`k = 50` to 0.03–0.30 at :math:`k = 200`. On the wardrobe at :math:`k = 200`, Monte
  Carlo's and stratified sampling's top-5 regions share no superpixel.

The numbers match the paper's Fig. 7 to within a few hundredths. For example, the orange at
:math:`k = 50` gives 0.392 | 0.084 for Monte Carlo and 2.11 | 1.00 for stratified sampling, against
0.387 | 0.0833 and 2.12 | 1.0 in the paper. The exception is the milk can's Monte Carlo cell at
:math:`k = 200` (0.354 | 0.076 here, 0.571 | 0.145 in the paper), which is confused either way.

How it was computed
-------------------

The setup is the one of the paper's Fig. 7, from
`lime-stratified-examples <https://github.com/rashidrao-pk/lime-stratified-examples>`_
(``Paper_Figures.ipynb``):

- the model is Keras' ``ResNet50(weights="imagenet")`` with ``preprocess_input``, and the
  explained output is the probability of its top class;
- the photos are ImageNet test images 114, 147, 60, 144 and 66, resized to 224 × 224;
- the superpixels come from quick shift (``kernel_size=4, ratio=0.2, random_seed=1234``), with
  ``max_dist`` found by the paper's dichotomic search for :math:`k` = 50, 100, 150 and 200;
- for each :math:`k` and each sampler, one explainer explains the photo 10 times, and the
  explanation is the average of the 10 coefficient vectors. :math:`CV` is computed on that
  average, while :math:`RC` and :math:`R^2` are averaged over the 10 runs.

.. code-block:: python

   from lime import lime_image

   explainer = lime_image.LimeImageExplainer(random_state=1234)
   for repeat in range(10):
       explanation = explainer.explain_instance(
           image, classifier_fn, labels=(label,), top_labels=None, hide_color=None,
           num_samples=1000, batch_size=500, segmentation_fn=lambda _: segments,
           use_stratification=stratified,          # False, then True
       )

The demo replays these calls. The compute script wraps ``LimeBase.explain_instance_with_data`` to
log what the unchanged explainer passed to its regressor. The page shows the masks and scores of
the first run of each setting, and the averaged coefficients, as the paper's figure does. Only the
first tab is computed in the browser, from the formulas of the paper.

Run it again
------------

The scripts and the five photos are in ``source/demos/sampling/compute``. ``config.json`` lists
the photos, the values of :math:`k`, the number of samples and of repeats. To explain other
photos, change the file and run:

.. code-block:: bash

   PYTHONPATH=../../../.. python explain.py   # the LIME runs, out/<photo>.npz (needs tensorflow, opencv)
   python pack.py                             # the demo's data, ../../../_static/demos/sampling/

With 10 repeats, the five photos take about 40 minutes on a laptop GPU.
