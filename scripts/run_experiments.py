"""Полный прогон экспериментов: results/metrics.json + рисунки в docs/figures.

Запуск:  python scripts/run_experiments.py [--seeds 3]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.datasets import load_breast_cancer  # noqa: E402
from sklearn.ensemble import RandomForestClassifier, StackingRegressor  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge  # noqa: E402
from sklearn.metrics import (accuracy_score, log_loss, mean_absolute_error,  # noqa: E402
                             mean_squared_error, r2_score)
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, train_test_split  # noqa: E402
from sklearn.naive_bayes import GaussianNB  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from stacking_lib import Blending, Stacking  # noqa: E402
from stacking_lib.data import load_housing  # noqa: E402
from stacking_lib.models import make_base_models, make_meta_nnls  # noqa: E402

FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5]


def metrics(y, p):
    return {"rmse": float(np.sqrt(mean_squared_error(y, p))),
            "mae": float(mean_absolute_error(y, p)),
            "r2": float(r2_score(y, p))}


def run_regression_seed(X, y, seed, with_sklearn_ref):
    t0 = time.time()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
    y_tr, y_te = y_tr.to_numpy(), y_te.to_numpy()
    res = {}

    # --- Stacking (OOF, K=5, мета = NNLS) ---------------------------------
    st = Stacking(make_base_models(seed), make_meta_nnls(), n_splits=5, random_state=seed)
    st.fit(X_tr, y_tr)
    names = [n for n, _ in st.base_models_]
    P_te = st.base_predictions(X_te)           # предсказания базовых моделей на test
    P_tr_in = st.base_predictions(X_tr)        # in-sample предсказания (для демо утечки)
    res["base"] = {n: {**metrics(y_te, P_te[:, i]),
                       "train_rmse": float(np.sqrt(mean_squared_error(y_tr, P_tr_in[:, i]))),
                       "oof_rmse": float(np.sqrt(mean_squared_error(y_tr, st.oof_[:, i])))}
                   for i, n in enumerate(names)}
    res["mean_ensemble"] = metrics(y_te, P_te.mean(axis=1))
    res["stacking_nnls"] = metrics(y_te, st.predict(X_te))
    res["weights_oof"] = st.meta_weights_
    res["intercept_oof"] = float(st.meta_model_.intercept_)

    # --- Другие мета-модели на тех же OOF-признаках (без пересчёта OOF) ---
    for key, meta in [("stacking_ridge", Ridge(alpha=1.0)),
                      ("stacking_hgb", HistGradientBoostingRegressor(
                          max_iter=200, learning_rate=0.05, random_state=seed))]:
        res[key] = metrics(y_te, meta.fit(st.oof_, y_tr).predict(P_te))

    # --- Демо утечки: мета-модель на in-sample предсказаниях --------------
    leak = make_meta_nnls().fit(P_tr_in, y_tr)
    res["leaky_stacking"] = {**metrics(y_te, leak.predict(P_te)),
                             "meta_train_rmse": float(np.sqrt(mean_squared_error(y_tr, leak.predict(P_tr_in)))),
                             "weights": dict(zip(names, leak.coef_.tolist()))}

    # --- Корреляция остатков OOF (разнообразие моделей) -------------------
    resid = st.oof_ - y_tr[:, None]
    res["oof_resid_corr"] = np.corrcoef(resid.T).tolist()
    res["names"] = names

    # --- Blending -----------------------------------------------------------
    sweep = {}
    for h in FRACTIONS:
        bl = Blending(make_base_models(seed), make_meta_nnls(), holdout_size=h,
                      random_state=seed).fit(X_tr, y_tr)
        sweep[str(h)] = {**metrics(y_te, bl.predict(X_te)), "weights": bl.meta_weights_}
    res["blend_sweep"] = sweep
    res["blending_0.2"] = {k: v for k, v in sweep["0.2"].items() if k != "weights"}

    # --- Референс: StackingRegressor из sklearn (только seed 0) ------------
    if with_sklearn_ref:
        ref = StackingRegressor(make_base_models(seed), LinearRegression(positive=True),
                                cv=KFold(5, shuffle=True, random_state=seed)).fit(X_tr, y_tr)
        p_ref = ref.predict(X_te)
        res["sklearn_stacking"] = metrics(y_te, p_ref)
        res["sklearn_vs_ours_max_abs_diff"] = float(np.max(np.abs(p_ref - st.predict(X_te))))
    res["seconds"] = round(time.time() - t0, 1)
    print(f"[housing seed={seed}] {res['seconds']}s  stacking RMSE={res['stacking_nnls']['rmse']:.0f}",
          flush=True)
    return res


def run_classification(n_repeats=2):
    X, y = load_breast_cancer(return_X_y=True)

    def base():
        return [("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
                ("svm", make_pipeline(StandardScaler(), SVC(probability=True, random_state=0))),
                ("knn", make_pipeline(StandardScaler(), KNeighborsClassifier(11))),
                ("rf", RandomForestClassifier(n_estimators=100, random_state=0)),
                ("nb", GaussianNB())]

    rows = {k: [] for k in ["lr", "svm", "knn", "rf", "nb", "mean_proba", "stacking", "blending"]}
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=0)
    for i, (tr, te) in enumerate(cv.split(X, y)):
        meta = lambda: LogisticRegression(max_iter=2000)  # noqa: E731
        st = Stacking(base(), meta(), task="classification", random_state=i).fit(X[tr], y[tr])
        bl = Blending(base(), meta(), holdout_size=0.3, task="classification",
                      random_state=i).fit(X[tr], y[tr])
        P = st.base_predictions(X[te])
        for j, n in enumerate(["lr", "svm", "knn", "rf", "nb"]):
            p1 = np.clip(P[:, j], 1e-6, 1 - 1e-6)
            rows[n].append((log_loss(y[te], p1), accuracy_score(y[te], p1 > 0.5)))
        pm = np.clip(P.mean(axis=1), 1e-6, 1 - 1e-6)
        rows["mean_proba"].append((log_loss(y[te], pm), accuracy_score(y[te], pm > 0.5)))
        for key, ens in [("stacking", st), ("blending", bl)]:
            pr = ens.predict_proba(X[te])
            rows[key].append((log_loss(y[te], pr), accuracy_score(y[te], ens.predict(X[te]))))
    out = {k: {"log_loss_mean": float(np.mean([a for a, _ in v])),
               "log_loss_std": float(np.std([a for a, _ in v])),
               "acc_mean": float(np.mean([b for _, b in v])),
               "acc_std": float(np.std([b for _, b in v]))} for k, v in rows.items()}
    print("[breast_cancer] done", flush=True)
    return out


def summarize(per_seed):
    keys = ["mean_ensemble", "stacking_nnls", "stacking_ridge", "stacking_hgb",
            "blending_0.2", "leaky_stacking"]
    summ = {}
    names = per_seed[0]["names"]
    for k in names:
        summ[k] = {m: [float(np.mean([s["base"][k][m] for s in per_seed])),
                       float(np.std([s["base"][k][m] for s in per_seed]))]
                   for m in ["rmse", "mae", "r2"]}
    for k in keys:
        summ[k] = {m: [float(np.mean([s[k][m] for s in per_seed])),
                       float(np.std([s[k][m] for s in per_seed]))] for m in ["rmse", "mae", "r2"]}
    summ["blend_sweep"] = {str(h): {"rmse": [float(np.mean([s["blend_sweep"][str(h)]["rmse"] for s in per_seed])),
                                             float(np.std([s["blend_sweep"][str(h)]["rmse"] for s in per_seed]))]}
                           for h in FRACTIONS}
    return summ


def make_figures(per_seed, summ, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    out_dir.mkdir(parents=True, exist_ok=True)
    names = per_seed[0]["names"]

    # 1. Сравнение RMSE
    order = names + ["mean_ensemble", "blending_0.2", "stacking_nnls"]
    labels = {"ridge": "Ridge", "knn": "KNN", "rf": "RandomForest", "et": "ExtraTrees",
              "hgb": "HistGB", "mean_ensemble": "Среднее", "blending_0.2": "Blending",
              "stacking_nnls": "Stacking"}
    vals = [summ[k]["rmse"][0] for k in order]
    errs = [summ[k]["rmse"][1] for k in order]
    colors = ["#9AA5B8"] * len(names) + ["#6EA8FE", "#F4B63F", "#E8574B"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar([labels[k] for k in order], vals, yerr=errs, color=colors, capsize=4)
    for i, v in enumerate(vals):
        ax.text(i, v + 900, f"{v/1000:.1f}k", ha="center", fontsize=10)
    ax.set_ylabel("RMSE на test, $ (меньше - лучше)")
    ax.set_ylim(min(vals) * 0.8, max(vals) * 1.08)
    ax.set_title("California Housing: качество моделей (среднее по сидам ± std)")
    plt.xticks(rotation=20)
    fig.tight_layout(); fig.savefig(out_dir / "fig_rmse_comparison.png", dpi=160); plt.close(fig)

    # 2. Корреляция остатков
    C = np.array(per_seed[0]["oof_resid_corr"])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(C, vmin=0.4, vmax=1, cmap="YlOrRd")
    ax.set_xticks(range(len(names)), [labels[n] for n in names], rotation=35)
    ax.set_yticks(range(len(names)), [labels[n] for n in names])
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if C[i, j] > 0.88 else "black")
    ax.set_title("Корреляция OOF-ошибок моделей")
    fig.colorbar(im, fraction=0.046); fig.tight_layout()
    fig.savefig(out_dir / "fig_residual_corr.png", dpi=160); plt.close(fig)

    # 3. Веса мета-модели: OOF vs in-sample
    w_oof = [per_seed[0]["weights_oof"][n] for n in names]
    w_in = [per_seed[0]["leaky_stacking"]["weights"][n] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(x - 0.19, w_oof, 0.38, label="мета-модель на OOF (верно)", color="#3CC1B0")
    ax.bar(x + 0.19, w_in, 0.38, label="мета-модель на in-sample (утечка)", color="#E8574B")
    ax.set_xticks(x, [labels[n] for n in names]); ax.set_ylabel("вес в мета-модели")
    ax.legend(frameon=False); ax.set_title("Утечка: мета-модель слепо доверяет переобученным моделям")
    fig.tight_layout(); fig.savefig(out_dir / "fig_weights_leakage.png", dpi=160); plt.close(fig)

    # 4. Размер holdout в blending
    hs = FRACTIONS
    m = [summ["blend_sweep"][str(h)]["rmse"][0] for h in hs]
    s = [summ["blend_sweep"][str(h)]["rmse"][1] for h in hs]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar([int(h * 100) for h in hs], m, yerr=s, marker="o", color="#F4B63F", capsize=4,
                label="Blending")
    ax.axhline(summ["stacking_nnls"]["rmse"][0], color="#E8574B", ls="--", label="Stacking (K=5)")
    ax.axhline(summ["hgb"]["rmse"][0], color="#9AA5B8", ls=":", label="лучшая одиночная модель")
    ax.set_xlabel("доля holdout, % от train"); ax.set_ylabel("RMSE на test, $")
    ax.legend(frameon=False); ax.set_title("Blending: чем больше holdout, тем слабее базовые модели")
    fig.tight_layout(); fig.savefig(out_dir / "fig_blend_holdout.png", dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    X, y = load_housing()
    per_seed = [run_regression_seed(X, y, s, with_sklearn_ref=(s == 0)) for s in range(args.seeds)]
    summ = summarize(per_seed)
    clf = run_classification()
    out = {"config": {"dataset": "California Housing (20640 rows)", "seeds": args.seeds,
                      "test_size": 0.2, "n_splits": 5, "fractions": FRACTIONS},
           "per_seed": per_seed, "summary": summ, "classification": clf}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    make_figures(per_seed, summ, ROOT / "docs" / "figures")
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
