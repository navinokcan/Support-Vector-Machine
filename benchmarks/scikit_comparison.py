"""Compare KKTSVM with scikit-learn's SVC (LIBSVM) on held-out test data.

    python benchmarks/scikit_comparison.py

Prints a markdown table. Both models get the same C, kernel, gamma and stopping tolerance.
"""

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from svm_kkt import KKTSVM  # noqa: E402

X_digits, y_digits = load_digits(return_X_y=True)
keep = np.isin(y_digits, (3, 8))
DATASETS = [
    ("moons", "rbf", make_moons(1000, noise=0.25, random_state=0)),
    ("circles", "rbf", make_circles(1000, noise=0.1, factor=0.5, random_state=0)),
    ("breast cancer", "rbf", load_breast_cancer(return_X_y=True)),
    ("breast cancer", "linear", load_breast_cancer(return_X_y=True)),
    ("digits 3 vs 8", "rbf", (X_digits[keep], y_digits[keep])),
]


def timed(fit):
    start = time.perf_counter()
    model = fit()
    return model, 1000 * (time.perf_counter() - start)


print("| Dataset | Kernel | n (train) | Test acc. (ours / sklearn) | Same predictions | "
      "Support vectors (ours / sklearn) | Same SVs | max abs diff in f(x) | "
      "Iterations (ours / sklearn) | Fit ms (ours / sklearn) |")
print("|---|---|--:|:-:|:-:|:-:|:-:|--:|:-:|:-:|")

for name, kernel, (X, y) in DATASETS:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

    ours, t_ours = timed(lambda: KKTSVM(C=1.0, kernel=kernel, tol=1e-3).fit(X_tr, y_tr))
    ref, t_ref = timed(lambda: SVC(C=1.0, kernel=kernel, tol=1e-3).fit(X_tr, y_tr))

    same_pred = np.mean(ours.predict(X_te) == ref.predict(X_te))
    same_svs = "yes" if set(ours.support_) == set(ref.support_) else "no"
    diff = np.abs(ours.decision_function(X_te) - ref.decision_function(X_te)).max()
    print(f"| {name} | {kernel} | {len(y_tr)} | {ours.score(X_te, y_te):.3f} / {ref.score(X_te, y_te):.3f} | "
          f"{100 * same_pred:.0f}% | {ours.support_.size} / {ref.support_.size} | {same_svs} | {diff:.1e} | "
          f"{ours.n_iter_} / {ref.n_iter_[0]} | {t_ours:.0f} / {t_ref:.0f} |")
