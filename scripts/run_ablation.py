"""Абляция: как состав пула базовых моделей влияет на выигрыш от stacking.

Одно обучение Stacking на сид; подмножества моделей берутся как срезы столбцов
OOF-матрицы, поэтому дополнительные обучения не нужны.
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from stacking_lib import Stacking
from stacking_lib.data import load_housing
from stacking_lib.models import make_base_models, make_meta_nnls

POOLS = {"все 5 моделей": ["ridge", "knn", "rf", "et", "hgb"],
         "без HistGB": ["ridge", "knn", "rf", "et"],
         "только деревья": ["rf", "et", "hgb"],
         "RF + KNN": ["rf", "knn"],
         "RF + ET (похожие)": ["rf", "et"]}
rmse = lambda y, p: float(np.sqrt(mean_squared_error(y, p)))
X, y = load_housing()
out = {k: {"stack": [], "best_single": [], "best_name": []} for k in POOLS}
for seed in range(3):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    ytr, yte = ytr.to_numpy(), yte.to_numpy()
    st = Stacking(make_base_models(seed), make_meta_nnls(), random_state=seed).fit(Xtr, ytr)
    names = [n for n, _ in st.base_models_]
    P = st.base_predictions(Xte)
    for pool, members in POOLS.items():
        idx = [names.index(m) for m in members]
        meta = make_meta_nnls().fit(st.oof_[:, idx], ytr)
        singles = {m: rmse(yte, P[:, names.index(m)]) for m in members}
        b = min(singles, key=singles.get)
        out[pool]["stack"].append(rmse(yte, meta.predict(P[:, idx])))
        out[pool]["best_single"].append(singles[b]); out[pool]["best_name"].append(b)
    print("seed", seed, flush=True)
(ROOT / "results" / "ablation.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
for k, v in out.items():
    print(f"{k:20s} stack {np.mean(v['stack']):8.0f}  best single {np.mean(v['best_single']):8.0f} ({v['best_name'][0]})")
