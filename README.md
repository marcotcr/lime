# lime

[![Build Status](https://travis-ci.org/marcotcr/lime.svg?branch=master)](https://travis-ci.org/marcotcr/lime)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/marcotcr/lime/master)

This project is about explaining what machine learning classifiers (or models) are doing.
At the moment, we support explaining individual predictions for text classifiers or classifiers that act on tables (numpy arrays of numerical or categorical data) or images, with a package called lime (short for local interpretable model-agnostic explanations).
Lime is based on the work presented in [this paper](https://arxiv.org/abs/1602.04938) ([bibtex here for citation](https://github.com/marcotcr/lime/blob/master/citation.bib)). Here is a link to the promo video:

<a href="https://www.youtube.com/watch?v=hUnRCxnydCc" target="_blank"><img src="https://raw.githubusercontent.com/marcotcr/lime/master/doc/images/video_screenshot.png" width="450" alt="KDD promo video"/></a>

Our plan is to add more packages that help users understand and interact meaningfully with machine learning.

Lime is able to explain any black box classifier, with two or more classes. All we require is that the classifier implements a function that takes in raw text or a numpy array and outputs a probability for each class. Support for scikit-learn classifiers is built-in.

## Installation

The lime package is on [PyPI](https://pypi.python.org/pypi/lime). Simply run:

```sh
pip install lime
```

Or clone the repository and run:

```sh
pip install .
```

We dropped python2 support in `0.2.0`, `0.1.1.37` was the last version before that.

## Screenshots

Below are some screenshots of lime explanations. These are generated in html, and can be easily produced and embedded in ipython notebooks. We also support visualizations using matplotlib, although they don't look as nice as these ones.

#### Two class case, text

Negative (blue) words indicate atheism, while positive (orange) words indicate christian. The way to interpret the weights by applying them to the prediction probabilities. For example, if we remove the words Host and NNTP from the document, we expect the classifier to predict atheism with probability 0.58 - 0.14 - 0.11 = 0.31.

![twoclass](doc/images/twoclass.png)

#### Multiclass case

![multiclass](doc/images/multiclass.png)

#### Tabular data

![tabular](doc/images/tabular.png)

#### Images (explaining prediction of 'Cat' in pros and cons)

<img src="https://raw.githubusercontent.com/marcotcr/lime/master/doc/images/images.png" width=200 />

This image visually explains why the model predicted ‘cat’.

Positive contribution pixels (green): Regions that are important for predicting the cat class. In the image, the cat’s body and face are highlighted.

Negative contribution pixels (red): Regions that decrease the probability of the cat class. In the image, part of a dog’s face and body are included.

LIME divides the input image into small regions (superpixels) and measures how the model’s output changes when each region is removed or altered.

## Tutorials and API

For example usage for text classifiers, take a look at the following two tutorials (generated from ipython notebooks):

- [Basic usage, two class. We explain random forest classifiers.](https://marcotcr.github.io/lime/tutorials/Lime%20-%20basic%20usage%2C%20two%20class%20case.html)
- [Multiclass case](https://marcotcr.github.io/lime/tutorials/Lime%20-%20multiclass.html)

For classifiers that use numerical or categorical data, take a look at the following tutorial (this is newer, so please let me know if you find something wrong):

- [Tabular data](https://marcotcr.github.io/lime/tutorials/Tutorial%20-%20continuous%20and%20categorical%20features.html)
- [Tabular data with H2O models](https://marcotcr.github.io/lime/tutorials/Tutorial_H2O_continuous_and_cat.html)
- [Latin Hypercube Sampling](doc/notebooks/Latin%20Hypercube%20Sampling.ipynb)

For image classifiers:

- [Images - basic](https://marcotcr.github.io/lime/tutorials/Tutorial%20-%20images.html)
- [Images - Faces](https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20Faces%20and%20GradBoost.ipynb)
- [Images with Keras](https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20Image%20Classification%20Keras.ipynb)
- [MNIST with random forests](https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20MNIST%20and%20RF.ipynb)
- [Images with PyTorch](https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20images%20-%20Pytorch.ipynb)

For regression:

- [Simple regression](https://marcotcr.github.io/lime/tutorials/Using%2Blime%2Bfor%2Bregression.html)

Submodular Pick:

- [Submodular Pick](https://github.com/marcotcr/lime/tree/master/doc/notebooks/Submodular%20Pick%20examples.ipynb)

The raw (non-html) notebooks for these tutorials are available [here](https://github.com/marcotcr/lime/tree/master/doc/notebooks).

The API reference is available [here](https://lime-ml.readthedocs.io/en/latest/).

## What are explanations?

Intuitively, an explanation is a local linear approximation of the model's behaviour.
While the model may be very complex globally, it is easier to approximate it around the vicinity of a particular instance.
While treating the model as a black box, we perturb the instance we want to explain and learn a sparse linear model around it, as an explanation.
The figure below illustrates the intuition for this procedure. The model's decision function is represented by the blue/pink background, and is clearly nonlinear.
The bright red cross is the instance being explained (let's call it X).
We sample instances around X, and weight them according to their proximity to X (weight here is indicated by size).
We then learn a linear model (dashed line) that approximates the model well in the vicinity of X, but not necessarily globally. For more information, [read our paper](https://arxiv.org/abs/1602.04938), or take a look at [this blog post](https://www.oreilly.com/learning/introduction-to-local-interpretable-model-agnostic-explanations-lime).

<img src="https://raw.githubusercontent.com/marcotcr/lime/master/doc/images/lime.png" width=300px />

## Contributing

Please read [this](CONTRIBUTING.md).
