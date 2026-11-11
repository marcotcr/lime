"""Pack the logs of explain.py into the demo's data files.

Writes, into source/_static/demos/sampling/:

- ``data.js``: the list of photos and the run settings (small, loaded first);
- ``<id>.js``: one photo's segmentations and logged LIME runs (loaded on demand);
- ``<id>.jpg``: the 224 x 224 photo the model saw;
- ``fallback.png``: the figure shown when JavaScript is off.

Arrays are base64 strings: masks are bit-packed rows (``np.packbits``),
segmentations are run-length encoded ``(label, length)`` uint16 pairs, and
scores, weights and coefficients are float32.

    python pack.py
"""

import base64
import json
from pathlib import Path

import numpy as np
from scipy.special import gammaln
from PIL import Image

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parents[2] / "_static" / "demos" / "sampling"


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def rle(segments):
    flat = segments.ravel()
    starts = np.r_[0, np.flatnonzero(np.diff(flat)) + 1]
    lengths = np.diff(np.r_[starts, flat.size])
    pairs = np.empty(2 * len(starts), dtype=np.uint16)
    pairs[0::2] = flat[starts]
    pairs[1::2] = lengths
    return b64(pairs)


def sample_weights(masks, stratified, kernel_width=0.25):
    """The weights LimeBase gives the samples: LIME's kernel of the cosine distance to the
    unmasked photo, multiplied (for stratified sampling) by the adjustment factor first."""
    k = masks.shape[1]
    kept = masks.sum(axis=1).astype(float)
    distances = 1 - np.sqrt(kept / k)  # cosine distance between x and the all-ones mask
    if stratified:
        adj = (k + 1) * np.exp(gammaln(k + 1) - gammaln(kept + 1) - gammaln(k - kept + 1) - k * np.log(2))
        adj[0] = 1.0
        distances = distances * adj
    return np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))


def run(d, name, k):
    p = f"{name}{k}_"
    out = {
        "masks": b64(np.packbits(d[p + "masks"], axis=1)),
        "y": b64(d[p + "y"].astype(np.float32)),
        "w": b64(sample_weights(d[p + "masks"], name == "st").astype(np.float32)),
        "beta": b64(d[p + "beta"].astype(np.float32)),
                "r2": round(float(d[p + "r2"]), 4),
        "cv": round(float(d[p + "cv"]), 4),
        "rc": round(float(d[p + "rc"]), 4),
    }
    return out


def bwr(segments, beta):
    """The paper's heatmap: matplotlib's bwr colour map, symmetric scale ±max|beta|."""
    t = np.clip(beta[segments] / (np.abs(beta).max() or 1.0), -1, 1)[..., None]
    pos = np.concatenate([np.ones_like(t), 1 - t, 1 - t], axis=2)
    neg = np.concatenate([1 + t, 1 + t, np.ones_like(t)], axis=2)
    return np.where(t >= 0, pos, neg) * 255


def fallback(records):
    """Fig. 7 without JavaScript: per photo, the photo, then MC and stratified at the smallest
    and largest k."""
    gap = np.full((224, 8, 3), 255.0)
    rows = []
    for d in records:
        ks = [int(k) for k in d["ks"]]
        tiles = [d["image"].astype(float)]
        for m in ("mc", "st"):
            for k in (ks[0], ks[-1]):
                tiles.append(bwr(d[f"seg{k}"], d[f"{m}{k}_beta"]))
        row = np.concatenate([x for t in tiles for x in (t, gap)][:-1], axis=1)
        rows += [row, np.full((8, row.shape[1], 3), 255.0)]
    img = Image.fromarray(np.concatenate(rows[:-1], axis=0).astype(np.uint8))
    img.thumbnail((900, 2000))
    img.save(ASSETS / "fallback.png", optimize=True)


def main():
    config = json.loads((HERE / "config.json").read_text())
    ASSETS.mkdir(parents=True, exist_ok=True)
    for old in list(ASSETS.glob("*.js")) + list(ASSETS.glob("*.jpg")):
        if old.name not in ("demo.js",):
            old.unlink()
    photos, records = [], []
    for photo in config["photos"]:
        d = np.load(HERE / "out" / f"{photo['id']}.npz")
        records.append(d)
        ks = [int(k) for k in d["ks"]]
        Image.fromarray(d["image"]).save(ASSETS / f"{photo['id']}.jpg", quality=92)
        runs = {}
        for k in ks:
            runs[str(k)] = {
                "seg": rle(d[f"seg{k}"].astype(np.uint16)),
                "maxDist": round(float(d[f"maxdist{k}"]), 3),
                "mc": run(d, "mc", k),
                "st": run(d, "st", k),
            }
        payload = json.dumps({"runs": runs}, separators=(",", ":"))
        (ASSETS / f"{photo['id']}.js").write_text(
            "// One photo of the LIME Stratified demo, written by compute/pack.py.\n"
            "window.LIMESTRAT_DEMO_PHOTOS = window.LIMESTRAT_DEMO_PHOTOS || {};\n"
            f"window.LIMESTRAT_DEMO_PHOTOS[{json.dumps(photo['id'])}] = {payload};\n")
        photos.append({
            "id": photo["id"],
            "title": photo["title"],
            "label": str(d["category"]),
            "prob": round(float(d["prob"]), 4),
            "photo": f"{photo['id']}.jpg",
            "file": f"{photo['id']}.js",
            "ks": ks,
        })
    fallback(records)
    meta = {
        "model": "ResNet50 (Keras, ImageNet weights)",
        "n": config["num_samples"],
        "repeats": config["repeats"],
        "seed": config["seed"],
        "size": 224,
        "photos": photos,
    }
    (ASSETS / "data.js").write_text(
        "// The LIME Stratified demo's data (see demo.js), written by compute/pack.py.\n"
        f"window.LIMESTRAT_DEMO = {json.dumps(meta, separators=(',', ':'))};\n")
    for f in sorted(ASSETS.iterdir()):
        print(f"{f.name:>20} {f.stat().st_size / 1024:8.1f} KiB")


if __name__ == "__main__":
    main()
