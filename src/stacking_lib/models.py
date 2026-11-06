"""Наборы базовых и мета-моделей для экспериментов."""
from __future__ import annotations

from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_base_models(seed: int = 0):
    """Пять разных по природе моделей: линейная, соседи, 2 леса, бустинг."""
    imp = lambda: SimpleImputer(strategy="median")  # noqa: E731
    return [
        ("ridge", make_pipeline(imp(), StandardScaler(), Ridge(alpha=1.0))),
        ("knn", make_pipeline(imp(), StandardScaler(),
                              KNeighborsRegressor(n_neighbors=10, weights="distance"))),
        ("rf", make_pipeline(imp(), RandomForestRegressor(
            n_estimators=100, min_samples_leaf=2, max_features=0.5, random_state=seed))),
        ("et", make_pipeline(imp(), ExtraTreesRegressor(
            n_estimators=100, min_samples_leaf=2, max_features=0.7, random_state=seed))),
        ("hgb", HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.08, l2_regularization=1.0, random_state=seed)),
    ]


def make_meta_nnls():
    """Линейная регрессия с неотрицательными весами - надёжный мета-алгоритм."""
    return LinearRegression(positive=True)
