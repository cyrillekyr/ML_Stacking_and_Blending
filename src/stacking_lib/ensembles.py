"""Собственная реализация Stacking и Blending (совместима с scikit-learn).

Обе реализации написаны «с нуля», чтобы было видно, где именно возникают
мета-признаки и почему нет утечки данных:

* ``Stacking``  - мета-признаки = out-of-fold (OOF) предсказания по K фолдам.
* ``Blending``  - мета-признаки = предсказания на одной отложенной выборке (holdout).

Поддерживаются регрессия (``task="regression"``) и классификация
(``task="classification"``, базовые модели должны уметь ``predict_proba``).
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


def _take(X, idx):
    """Выбрать строки и из DataFrame, и из ndarray."""
    return X.iloc[idx] if hasattr(X, "iloc") else np.asarray(X)[idx]


def _base_output(model, X, task: str) -> np.ndarray:
    """Предсказание базовой модели в виде матрицы (n, d) - мета-признаки."""
    if task == "regression":
        return np.asarray(model.predict(X), dtype=float).reshape(-1, 1)
    proba = np.asarray(model.predict_proba(X), dtype=float)
    # Вероятности суммируются в 1 - один столбец лишний (коллинеарность).
    return proba[:, 1:]


class _BaseEnsemble(BaseEstimator):
    """Общая часть: хранение моделей, предсказание, веса мета-модели."""

    def _check_task(self) -> None:
        if self.task not in ("regression", "classification"):
            raise ValueError("task должен быть 'regression' или 'classification'")

    def base_predictions(self, X) -> np.ndarray:
        """Мета-признаки для X от базовых моделей, обученных в fit()."""
        return np.hstack([_base_output(m, X, self.task) for _, m in self.base_models_])

    def predict(self, X) -> np.ndarray:
        return self.meta_model_.predict(self.base_predictions(X))

    def predict_proba(self, X) -> np.ndarray:
        if self.task != "classification":
            raise AttributeError("predict_proba доступен только для классификации")
        return self.meta_model_.predict_proba(self.base_predictions(X))

    @property
    def meta_weights_(self) -> dict[str, float]:
        """Веса линейной мета-модели по базовым моделям (регрессия)."""
        coef = np.ravel(getattr(self.meta_model_, "coef_", []))
        names = [n for n, _ in self.base_models_]
        if self.task != "regression" or len(coef) != len(names):
            raise AttributeError("веса доступны для линейной мета-модели в регрессии")
        return dict(zip(names, coef.tolist()))


class Stacking(_BaseEnsemble):
    """Stacked generalization (Wolpert, 1992) с OOF-предсказаниями.

    Алгоритм fit(X, y):
      1. Разбить train на K фолдов.
      2. Для каждой базовой модели и каждого фолда k: обучить на K-1 фолдах,
         предсказать оставшийся фолд -> получаем OOF-столбец (по одному на модель).
      3. Обучить мета-модель на матрице OOF-предсказаний Z (n x M).
      4. Переобучить каждую базовую модель на ВСЕМ train - они нужны для predict().

    Параметры
    ---------
    base_models : list[(str, estimator)]  модели уровня 0
    meta_model  : estimator               модель уровня 1
    n_splits    : int                     число фолдов K
    task        : "regression" | "classification"
    passthrough : bool                    добавить исходные признаки к мета-признакам
                                          (X должен быть числовым)
    """

    def __init__(self, base_models, meta_model, n_splits=5, task="regression",
                 passthrough=False, random_state=0):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_splits = n_splits
        self.task = task
        self.passthrough = passthrough
        self.random_state = random_state

    def _oof_matrix(self, X, y) -> np.ndarray:
        n = len(y)
        cv = (KFold if self.task == "regression" else StratifiedKFold)(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        folds = list(cv.split(np.zeros(n), y))
        columns = []
        for _, est in self.base_models:
            oof = None
            for tr_idx, va_idx in folds:
                model = clone(est).fit(_take(X, tr_idx), y[tr_idx])
                out = _base_output(model, _take(X, va_idx), self.task)
                if oof is None:
                    oof = np.empty((n, out.shape[1]))
                oof[va_idx] = out
            columns.append(oof)
        return np.hstack(columns)

    def fit(self, X, y):
        self._check_task()
        y = np.asarray(y)
        if self.task == "classification":
            self.classes_ = np.unique(y)
        self.oof_ = self._oof_matrix(X, y)                      # шаги 1-2
        Z = self.oof_
        if self.passthrough:
            Z = np.hstack([Z, np.asarray(X, dtype=float)])
        self.meta_model_ = clone(self.meta_model).fit(Z, y)     # шаг 3
        self.base_models_ = [(n, clone(m).fit(X, y))            # шаг 4
                             for n, m in self.base_models]
        return self

    def base_predictions(self, X) -> np.ndarray:
        Z = super().base_predictions(X)
        return np.hstack([Z, np.asarray(X, dtype=float)]) if self.passthrough else Z


class Blending(_BaseEnsemble):
    """Blending: мета-модель учится на отложенной выборке (holdout).

    Алгоритм fit(X, y):
      1. Разбить train на две части: train' (1 - h) и holdout (h).
      2. Обучить базовые модели ТОЛЬКО на train'.
      3. Получить их предсказания на holdout -> матрица Z (h*n x M).
      4. Обучить мета-модель на (Z, y_holdout).
    Базовые модели не переобучаются на всём train: мета-модель настроена на
    предсказания именно этих моделей.
    """

    def __init__(self, base_models, meta_model, holdout_size=0.2, task="regression",
                 random_state=0):
        self.base_models = base_models
        self.meta_model = meta_model
        self.holdout_size = holdout_size
        self.task = task
        self.random_state = random_state

    def fit(self, X, y):
        self._check_task()
        y = np.asarray(y)
        if self.task == "classification":
            self.classes_ = np.unique(y)
        tr_idx, ho_idx = train_test_split(
            np.arange(len(y)), test_size=self.holdout_size,
            stratify=y if self.task == "classification" else None,
            random_state=self.random_state)
        self.base_models_ = [(n, clone(m).fit(_take(X, tr_idx), y[tr_idx]))
                             for n, m in self.base_models]
        self.holdout_idx_ = ho_idx
        self.holdout_features_ = super().base_predictions(_take(X, ho_idx))
        self.meta_model_ = clone(self.meta_model).fit(self.holdout_features_, y[ho_idx])
        return self
