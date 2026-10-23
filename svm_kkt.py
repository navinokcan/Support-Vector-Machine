"""A soft-margin support vector machine whose training is driven by the KKT conditions.

Primal problem (r_i = 1 for the usual SVM; other values are used for the shadow-price check):

    min_{w, b, xi}  1/2 ||w||^2 + C * sum_i xi_i
    s.t.            y_i f(x_i) >= r_i - xi_i,   xi_i >= 0,      f(x) = w.phi(x) + b

Dual problem, written as a minimisation:

    min_alpha  F(alpha) = 1/2 alpha^T Q alpha - r^T alpha,   Q_ij = y_i y_j K(x_i, x_j)
    s.t.       sum_i y_i alpha_i = 0,   0 <= alpha_i <= C

KKT conditions in "gap" form (Keerthi et al., 2001). Let G = Q alpha - r be the gradient and

    I_up  = {t : alpha_t < C, y_t = +1}  U  {t : alpha_t > 0, y_t = -1}
    I_low = {t : alpha_t < C, y_t = -1}  U  {t : alpha_t > 0, y_t = +1}

Each point t "asks for" the bias -y_t G_t, the bias that would put it exactly on its margin.
Points in I_up need b at least that large, points in I_low need b at most that large, so

    alpha is optimal  <=>  m(alpha) = max_{I_up} -y_t G_t  <=  min_{I_low} -y_t G_t = M(alpha).

m - M is the KKT gap. The solver (SMO) uses it to pick which two multipliers to update,
to decide when to stop, and to read off the bias b.
"""

import warnings

import numpy as np


def smo(K, y, C, tol=1e-3, r=None, max_iter=100_000):
    """Solve the SVM dual with Sequential Minimal Optimisation.

    K : (n, n) kernel matrix, y : labels in {-1, +1}, C : box constraint.
    r : required margin for each point (default 1).
    Returns (alpha, b, info) where info holds the KKT gap and dual objective per iteration.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    r = np.ones(n) if r is None else np.asarray(r, dtype=float)
    diag = np.diag(K)
    alpha = np.zeros(n)
    G = -r.copy()  # gradient of F at alpha = 0
    gaps, duals = [], []

    for it in range(max_iter + 1):
        # 1. Check the KKT conditions: find the most violating point i and the gap m - M.
        up = ((y > 0) & (alpha < C)) | ((y < 0) & (alpha > 0))
        low = ((y < 0) & (alpha < C)) | ((y > 0) & (alpha > 0))
        score = -y * G
        i = np.flatnonzero(up)[np.argmax(score[up])]
        m, M = score[i], score[low].min()
        gaps.append(m - M)
        duals.append(0.5 * alpha @ (r - G))  # dual objective r^T alpha - 1/2 alpha^T Q alpha
        if m - M <= tol:
            break
        if it == max_iter:
            warnings.warn(f"SMO hit max_iter={max_iter} with KKT gap {m - M:.2e} > tol={tol}")
            break

        # 2. Choose the partner j among the points that violate the KKT conditions together
        #    with i. Taking j = argmin of score over I_low would give the "maximal violating
        #    pair"; LIBSVM's second-order rule instead picks the j with the largest guaranteed
        #    decrease of F, violation^2 / curvature (Fan, Chen & Lin, 2005).
        cand = np.flatnonzero(low & (score < m))
        violation = m - score[cand]
        curvature = np.maximum(diag[i] + diag[cand] - 2 * K[i, cand], 1e-12)
        j = cand[np.argmax(violation**2 / curvature)]

        # 3. Move along d_i = y_i, d_j = -y_j (this keeps sum_i y_i alpha_i = 0).
        #    F is a 1-D quadratic along that line, so the best step is slope / curvature,
        #    clipped so that both multipliers stay inside [0, C].
        a = max(diag[i] + diag[j] - 2 * K[i, j], 1e-12)
        room_i = C - alpha[i] if y[i] > 0 else alpha[i]
        room_j = C - alpha[j] if y[j] < 0 else alpha[j]
        t = min((score[i] - score[j]) / a, room_i, room_j)
        alpha[i] = (C if y[i] > 0 else 0.0) if t == room_i else alpha[i] + y[i] * t
        alpha[j] = (C if y[j] < 0 else 0.0) if t == room_j else alpha[j] - y[j] * t

        # 4. Update the gradient: G += Q[:, i] * y_i t - Q[:, j] * y_j t = t * y * (K_i - K_j).
        G += t * y * (K[:, i] - K[:, j])

    # 5. The bias is the KKT multiplier of sum_i y_i alpha_i = 0. For a point strictly
    #    inside the box (on the margin) complementary slackness gives b = -y_i G_i exactly.
    free = (alpha > 0) & (alpha < C)
    b = score[free].mean() if free.any() else 0.5 * (m + M)
    return alpha, b, {"n_iter": it, "kkt_gap": np.array(gaps), "dual_objective": np.array(duals)}


class KKTSVM:
    """Binary SVM classifier trained with KKT-driven SMO.

    Parameters
    ----------
    C : float, soft-margin penalty.
    kernel : "rbf" or "linear".
    gamma : "scale" (= 1 / (n_features * X.var()), as in scikit-learn) or a float.
    tol : stop when the KKT gap m(alpha) - M(alpha) is at most tol.
    """

    def __init__(self, C=1.0, kernel="rbf", gamma="scale", tol=1e-3, max_iter=100_000):
        if kernel not in ("rbf", "linear"):
            raise ValueError("kernel must be 'rbf' or 'linear'")
        self.C, self.kernel, self.gamma, self.tol, self.max_iter = C, kernel, gamma, tol, max_iter

    def _kernel(self, A, B):
        if self.kernel == "linear":
            return A @ B.T
        sq_dist = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * A @ B.T
        return np.exp(-self.gamma_ * np.maximum(sq_dist, 0))

    def _labels(self, y):
        return np.where(np.asarray(y) == self.classes_[1], 1.0, -1.0)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(f"KKTSVM is a binary classifier, but y has {len(self.classes_)} classes")
        y_pm = self._labels(y)
        self.gamma_ = 1 / (X.shape[1] * X.var()) if self.gamma == "scale" else float(self.gamma)

        alpha, b, info = smo(self._kernel(X, X), y_pm, self.C, self.tol, max_iter=self.max_iter)

        # Complementary slackness sorts the points: alpha = 0 (outside the margin),
        # 0 < alpha < C (exactly on the margin), alpha = C (inside the margin).
        self.alpha_, self.b_ = alpha, b
        self.support_ = np.flatnonzero(alpha > 0)
        self.margin_support_ = np.flatnonzero((alpha > 0) & (alpha < self.C))
        self.bound_support_ = np.flatnonzero(alpha == self.C)
        self.support_vectors_ = X[self.support_]
        self.dual_coef_ = alpha[self.support_] * y_pm[self.support_]
        self.n_iter_, self.history_ = info["n_iter"], info
        return self

    def decision_function(self, X):
        """f(x) = sum over support vectors of alpha_i y_i K(x_i, x) + b."""
        return self._kernel(np.asarray(X, dtype=float), self.support_vectors_) @ self.dual_coef_ + self.b_

    def predict(self, X):
        return self.classes_[(self.decision_function(X) > 0).astype(int)]

    def score(self, X, y):
        return np.mean(self.predict(X) == np.asarray(y))


def kkt_report(model, X, y):
    """Recompute every KKT condition of the primal from the fitted model, without the solver."""
    y_pm = model._labels(y)
    a, C = model.alpha_, model.C
    margin = y_pm * model.decision_function(X)
    w_sq = model.dual_coef_ @ model._kernel(model.support_vectors_, model.support_vectors_) @ model.dual_coef_
    primal = 0.5 * w_sq + C * np.maximum(0, 1 - margin).sum()
    dual = a.sum() - 0.5 * w_sq

    non, free, bound = a == 0, (a > 0) & (a < C), a == C
    violation = np.zeros_like(a)
    violation[non] = np.maximum(0, 1 - margin[non])  # alpha = 0      =>  y f(x) >= 1
    violation[free] = np.abs(margin[free] - 1)  # 0 < alpha < C  =>  y f(x) == 1
    violation[bound] = np.maximum(0, margin[bound] - 1)  # alpha = C      =>  y f(x) <= 1

    return {
        "n_non_sv": int(non.sum()),
        "n_margin_sv": int(free.sum()),
        "n_bound_sv": int(bound.sum()),
        "sum_alpha_y": float(a @ y_pm),  # stationarity in b
        "box_violation": float(max(0.0, (-a).max(), (a - C).max())),  # dual feasibility
        "max_violation": float(violation.max()),  # complementary slackness
        "primal": float(primal),
        "dual": float(dual),
        "duality_gap": float(primal - dual),
        "margins": margin,
    }


def shadow_prices(model, X, y, h=1e-4, tol=1e-10):
    """Check the envelope theorem: dV*/dr_i = alpha_i.

    V*(r) is the optimal objective when point i must reach margin r_i. Re-solving with each r_i
    nudged by +-h gives a finite-difference derivative to compare with the multiplier alpha_i.
    Returns (alpha, dV_dr), both of shape (n,).
    """
    X = np.asarray(X, dtype=float)
    y_pm = model._labels(y)
    K = model._kernel(X, X)
    n = len(y_pm)

    def optimal_value(r):
        _, _, info = smo(K, y_pm, model.C, tol, r=r, max_iter=10 * model.max_iter)
        return info["dual_objective"][-1]  # strong duality: optimal dual = optimal primal

    alpha, _, _ = smo(K, y_pm, model.C, tol, max_iter=10 * model.max_iter)
    dV = np.empty(n)
    for i in range(n):
        r = np.ones(n)
        r[i] = 1 + h
        up = optimal_value(r)
        r[i] = 1 - h
        dV[i] = (up - optimal_value(r)) / (2 * h)
    return alpha, dV
