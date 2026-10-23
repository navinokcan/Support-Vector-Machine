import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from svm_kkt import KKTSVM, kkt_report, shadow_prices, smo


@pytest.fixture(scope="module")
def moons():
    return make_moons(150, noise=0.25, random_state=7)


@pytest.fixture(scope="module")
def cancer():
    X, y = load_breast_cancer(return_X_y=True)
    return StandardScaler().fit_transform(X), y


def test_two_points_give_the_textbook_answer():
    # x = -1 vs x = +1 with a hard margin: w = 1, b = 0, alpha = (1/2, 1/2), solved in one step
    X = np.array([[-1.0, 0.0], [1.0, 0.0]])
    model = KKTSVM(kernel="linear", C=100.0, tol=1e-12).fit(X, [0, 1])
    np.testing.assert_allclose(model.alpha_, [0.5, 0.5])
    assert model.b_ == pytest.approx(0.0)
    assert model.n_iter_ == 1


def test_far_away_point_is_not_a_support_vector():
    # complementary slackness: the point at x = 3 has margin 3 > 1, so alpha = 0
    X = np.array([[-1.0], [1.0], [3.0]])
    model = KKTSVM(kernel="linear", C=100.0, tol=1e-12).fit(X, [-1, 1, 1])
    np.testing.assert_allclose(model.alpha_, [0.5, 0.5, 0.0])
    np.testing.assert_array_equal(model.support_, [0, 1])


@pytest.mark.parametrize("kernel", ["rbf", "linear"])
def test_trained_model_satisfies_every_kkt_condition(moons, kernel):
    X, y = moons
    tol = 1e-3
    report = kkt_report(KKTSVM(C=1.0, kernel=kernel, tol=tol).fit(X, y), X, y)
    assert abs(report["sum_alpha_y"]) < 1e-10  # stationarity
    assert report["box_violation"] == 0  # dual feasibility
    assert report["max_violation"] <= tol  # complementary slackness, bounded by the stopping rule
    assert 0 <= report["duality_gap"] < 1e-3 * report["primal"]


def test_complementary_slackness_sorts_the_points(moons):
    X, y = moons
    model = KKTSVM(C=3.0, gamma=2.0, tol=1e-6).fit(X, y)
    margins = kkt_report(model, X, y)["margins"]
    non_sv = np.setdiff1d(np.arange(len(y)), model.support_)
    assert np.all(margins[non_sv] >= 1 - 1e-6)
    np.testing.assert_allclose(margins[model.margin_support_], 1.0, atol=1e-6)
    assert np.all(margins[model.bound_support_] <= 1 + 1e-6)


def test_tighter_tolerance_gives_a_tighter_certificate(cancer):
    X, y = cancer
    loose = kkt_report(KKTSVM(tol=1e-2).fit(X, y), X, y)
    tight = kkt_report(KKTSVM(tol=1e-6).fit(X, y), X, y)
    assert tight["max_violation"] < loose["max_violation"]
    assert tight["duality_gap"] < loose["duality_gap"]


def test_dual_objective_increases_every_step(moons):
    X, y = moons
    steps = np.diff(KKTSVM(C=1.0, tol=1e-6).fit(X, y).history_["dual_objective"])
    assert np.all(steps >= -1e-12)


@pytest.mark.parametrize("kernel", ["rbf", "linear"])
@pytest.mark.parametrize("dataset", ["moons", "cancer"])
def test_matches_scikit_learn(dataset, kernel, request):
    X, y = request.getfixturevalue(dataset)
    ours = KKTSVM(kernel=kernel, tol=1e-6).fit(X, y)
    ref = SVC(kernel=kernel, tol=1e-6).fit(X, y)
    np.testing.assert_array_equal(np.sort(ours.support_), np.sort(ref.support_))
    np.testing.assert_allclose(ours.decision_function(X), ref.decision_function(X), atol=1e-4)
    np.testing.assert_array_equal(ours.predict(X), ref.predict(X))


def test_alpha_is_the_shadow_price_of_each_margin_constraint(moons):
    X, y = moons[0][:60], moons[1][:60]
    model = KKTSVM(C=1.0, gamma=2.0).fit(X, y)
    alpha, dV = shadow_prices(model, X, y)
    np.testing.assert_allclose(dV, alpha, atol=1e-6)


def test_labels_can_be_anything(moons):
    X, y = moons
    labels = np.array(["no", "yes"])[y]
    model = KKTSVM().fit(X, labels)
    assert set(model.predict(X)) <= {"no", "yes"}
    assert model.score(X, labels) > 0.9


def test_rejects_more_than_two_classes():
    with pytest.raises(ValueError):
        KKTSVM().fit(np.random.rand(6, 2), [0, 1, 2, 0, 1, 2])


def test_warns_if_max_iter_is_too_small(moons):
    X, y = moons
    with pytest.warns(UserWarning):
        KKTSVM(max_iter=3).fit(X, y)


def test_smo_keeps_the_equality_constraint(moons):
    X, y = moons
    y_pm = np.where(y == 1, 1.0, -1.0)
    K = X @ X.T
    alpha, _, _ = smo(K, y_pm, C=0.5)
    assert abs(alpha @ y_pm) < 1e-10
    assert np.all((alpha >= 0) & (alpha <= 0.5))
