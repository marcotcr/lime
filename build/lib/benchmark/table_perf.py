"""
A helper script for evaluating performance of changes to the tabular explainer, in this case different
implementations and methods for distance calculation.
"""

import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from lime.lime_tabular import LimeTabularExplainer


def interpret_data(X, y, func):
    explainer = LimeTabularExplainer(X, discretize_continuous=False, kernel_width=3)
    times, scores = [], []
    for r_idx in range(100):
        start_time = time.time()
        explanation = explainer.explain_instance(X[r_idx, :], func)
        times.append(time.time() - start_time)
        scores.append(explanation.score)
        print('...')

    return times, scores


if __name__ == '__main__':
    X_raw, y_raw = make_classification(n_classes=2, n_features=1000, n_samples=1000)
    clf = RandomForestClassifier()
    clf.fit(X_raw, y_raw)
    y_hat = clf.predict_proba(X_raw)

    times, scores = interpret_data(X_raw, y_hat, clf.predict_proba)
    print('%9.4fs %9.4fs %9.4fs' % (min(times), sum(times) / len(times), max(times)))
    print('%9.4f %9.4f% 9.4f' % (min(scores), sum(scores) / len(scores), max(scores)))