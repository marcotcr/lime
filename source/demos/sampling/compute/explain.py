"""Explain the demo's photos with LIME Image, Monte Carlo and stratified.

This reproduces the setup of the paper's Fig. 7 (the code of
https://github.com/rashidrao-pk/lime-stratified-examples, Paper_Figures.ipynb):

- model: Keras ResNet50 with ImageNet weights and ``preprocess_input``;
- photos: ImageNet test images, resized to 224 x 224 (no crop);
- segmentation: quick shift (``kernel_size=4, ratio=0.2, random_seed=1234``),
  with ``max_dist`` found by the paper's dichotomic search for k superpixels;
- for each k and each sampler, one ``LimeImageExplainer(random_state=1234)``
  explains the photo 10 times (``num_samples=1000``, mean-filled), and the
  explanation is the average of the 10 coefficient vectors. CV is computed on
  that average, and RC and R^2 are averaged over the 10 runs, as in the paper.

The library's ``explain_instance`` is not changed. A wrapper around
``LimeBase.explain_instance_with_data`` records what each run fed to its
regressor; the demo shows the neighbourhood (masks, scores, sample weights) of
the first run.

Output: one ``out/<photo id>.npz`` per photo, read by pack.py.

    python explain.py                  # all photos in config.json
    python explain.py --only orange    # some of them
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import (ResNet50, decode_predictions,
                                                    preprocess_input)

from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm

HERE = Path(__file__).resolve().parent
SIZE = 224


def load_photo(path):
    """As read_process_image in the paper's code: RGB, resized to SIZE x SIZE."""
    img = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return cv2.resize(img, [SIZE, SIZE])


def segment_count(image, max_dist):
    seg = SegmentationAlgorithm("quickshift", kernel_size=4, max_dist=max_dist,
                                ratio=0.2, random_seed=1234)
    return np.unique(seg(image)).shape[0]


def search_max_dist(image, target, init_max_dist=100):
    """The paper's search_segment_number: dichotomic search on max_dist."""
    lmd, rmd = 0.0, float(init_max_dist)
    lsn, rsn = segment_count(image, lmd), segment_count(image, rmd)
    for _ in range(20):
        if rsn == target:
            break
        mmd = (lmd + rmd) / 2.0
        msn = segment_count(image, mmd)
        if msn <= target <= lsn:
            rsn, rmd = msn, mmd
        else:
            lsn, lmd = msn, mmd
    return rmd


def explain(image, classifier_fn, label, segments, stratified, num_samples, repeats):
    """`repeats` unchanged explain_instance calls with one explainer, as in the paper."""
    explainer = lime_image.LimeImageExplainer(random_state=1234)
    fit = explainer.base.explain_instance_with_data
    seen = []

    def logged(data, labels, distances, *args, weight_adjustments=None, **kwargs):
        # the sample weights, as LimeBase computes them: kernel(distance x adjustment)
        scaled = distances if weight_adjustments is None else distances * weight_adjustments
        weights = explainer.base.kernel_fn(scaled)
        seen.append(dict(masks=data.astype(bool), y=labels[:, label], weights=weights))
        return fit(data, labels, distances, *args,
                   weight_adjustments=weight_adjustments, **kwargs)

    explainer.base.explain_instance_with_data = logged
    k = int(segments.max()) + 1
    betas, rcs, r2s = [], [], []
    for _ in range(repeats):
        exp = explainer.explain_instance(
            image, classifier_fn, labels=(label,), top_labels=None, hide_color=None,
            num_features=100000, num_samples=num_samples, batch_size=500,
            segmentation_fn=lambda _: segments, use_stratification=stratified,
            progress_bar=False)
        beta = np.zeros(k)
        for feature, weight in exp.local_exp[label]:
            beta[feature] = weight
        y = seen[-1]["y"]
        betas.append(beta)
        rcs.append((np.quantile(y, 0.99) - np.quantile(y, 0.01)) / y[0])
        r2s.append(exp.score[label])
    beta = np.mean(betas, axis=0)
    return dict(seen[0], beta=beta, cv=float(beta.std() / beta.mean()),
                rc=float(np.mean(rcs)), r2=float(np.mean(r2s)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=HERE / "config.json", type=Path)
    parser.add_argument("--only", nargs="*", help="photo ids to (re)compute")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())

    model = ResNet50(weights="imagenet")

    def classifier_fn(images):
        x = preprocess_input(np.array(images, dtype=np.float32))
        return model.predict(x, verbose=0, batch_size=500)

    out = HERE / "out"
    out.mkdir(exist_ok=True)
    for photo in config["photos"]:
        if args.only and photo["id"] not in args.only:
            continue
        image = load_photo(HERE / photo["file"])
        probs = classifier_fn(image[None])
        label = int(probs[0].argmax())
        category = decode_predictions(probs, top=1)[0][0][1]
        record = {"image": image, "label": label, "prob": float(probs[0, label]),
                  "category": category, "ks": []}
        for target in config["k"]:
            max_dist = search_max_dist(image, target)
            segments = SegmentationAlgorithm("quickshift", kernel_size=4, max_dist=max_dist,
                                             ratio=0.2, random_seed=1234)(image)
            segments = np.unique(segments, return_inverse=True)[1].reshape(segments.shape)
            k = int(segments.max()) + 1
            record["ks"].append(k)
            record[f"seg{k}"] = segments
            record[f"maxdist{k}"] = max_dist
            for name, stratified in (("mc", False), ("st", True)):
                r = explain(image, classifier_fn, label, segments, stratified,
                            config["num_samples"], config["repeats"])
                for key, value in r.items():
                    record[f"{name}{k}_{key}"] = value
                print(f"{photo['id']:>15} k={k:3d} {name}: CV={r['cv']:.3f} "
                      f"RC={r['rc']:.3f} R2={r['r2']:.3f}", flush=True)
        record["ks"] = np.array(record["ks"])
        np.savez_compressed(out / f"{photo['id']}.npz",
                            **{k: np.asarray(v) for k, v in record.items()})
        print(f"{photo['id']}: {category} ({record['prob']:.1%})", flush=True)


if __name__ == "__main__":
    main()
