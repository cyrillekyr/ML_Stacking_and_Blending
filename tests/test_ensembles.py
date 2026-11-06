import numpy as np
from sklearn.datasets import load_breast_cancer, make_regression
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from stacking_lib import Blending, Stacking


def _reg_data():
    return make_regression(n_samples=300, n_features=8, noise=10, random_state=0)


def _reg_models():
    return [("ridge", Ridge(1.0)),
            ("tree", DecisionTreeRegressor(max_depth=5, random_state=0)),
            ("knn", make_pipeline(StandardScaler(), KNeighborsRegressor(7)))]


def test_stacking_matches_sklearn():
    """Наша реализация даёт те же предсказания, что StackingRegressor."""
    X, y = _reg_data()
    ours = Stacking(_reg_models(), LinearRegression(), n_splits=5, random_state=42).fit(X[:240], y[:240])
    ref = StackingRegressor(_reg_models(), LinearRegression(),
                            cv=KFold(5, shuffle=True, random_state=42)).fit(X[:240], y[:240])
    np.testing.assert_allclose(ours.predict(X[240:]), ref.predict(X[240:]), rtol=1e-6, atol=1e-6)


def test_oof_has_no_leakage():
    """1-NN в in-sample предсказывает train идеально, а OOF - нет."""
    X, y = _reg_data()
    st = Stacking([("nn1", KNeighborsRegressor(1))], LinearRegression(), n_splits=5).fit(X, y)
    assert np.allclose(st.base_predictions(X)[:, 0], y)        # in-sample: утечка
    assert np.sqrt(np.mean((st.oof_[:, 0] - y) ** 2)) > 10     # OOF: честная ошибка


def test_blending_uses_holdout_only():
    X, y = _reg_data()
    bl = Blending(_reg_models(), LinearRegression(), holdout_size=0.25, random_state=1).fit(X, y)
    assert bl.holdout_features_.shape == (75, 3)
    assert len(set(bl.holdout_idx_)) == 75
    assert set(bl.meta_weights_) == {"ridge", "tree", "knn"}


def test_classification_stacking_and_blending():
    X, y = load_breast_cancer(return_X_y=True)
    base = [("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
            ("tree", DecisionTreeClassifier(max_depth=4, random_state=0)),
            ("knn", make_pipeline(StandardScaler(), KNeighborsClassifier(9)))]
    for ens in (Stacking(base, LogisticRegression(), task="classification"),
                Blending(base, LogisticRegression(), holdout_size=0.3, task="classification")):
        ens.fit(X[:400], y[:400])
        proba = ens.predict_proba(X[400:])
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)
        assert (ens.predict(X[400:]) == y[400:]).mean() > 0.9
