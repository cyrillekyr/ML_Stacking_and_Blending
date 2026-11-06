# Stacked Generalization: Stacking и Blending

Доклад по машинному обучению (СПбГУ, 2026): идея stacked generalization, схемы **Stacking** (OOF-предсказания) и **Blending** (holdout), собственная реализация и эксперименты.

* Доклад-отчёт: [`docs/report.md`](docs/report.md)
* Презентация: [`docs/presentation.pptx`](docs/presentation.pptx)
* Пошаговый разбор: [`notebooks/main.ipynb`](notebooks/main.ipynb)

## Что внутри

```
src/stacking_lib/   Stacking и Blending с нуля (ensembles.py), загрузка данных, наборы моделей
scripts/            run_experiments.py (основные эксперименты), run_ablation.py (состав пула)
tests/              проверки: совпадение со sklearn StackingRegressor, отсутствие утечки, классификация
notebooks/main.ipynb  пошаговый разбор на одном разбиении + сводка из results/
results/            metrics.json, ablation.json - результаты прогонов
docs/               отчёт, презентация, рисунки
```

## Датасет

California Housing (цены на жильё), 20 640 строк. Файл лежит в `data/housing.csv`; если его нет, он скачивается автоматически из
[ageron/handson-ml2](https://github.com/ageron/handson-ml2/tree/master/datasets/housing).
Цель - `median_house_value`.

## Установка и запуск

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m pytest                    # тесты (несколько секунд)
python scripts/run_experiments.py   # эксперименты, около 6 минут на 1 ядре
python scripts/run_ablation.py      # абляция, около 5 минут
jupyter lab                         # открыть notebooks/main.ipynb
```

Минимальный пример:

```python
from stacking_lib import Stacking, Blending
from stacking_lib.models import make_base_models, make_meta_nnls

stack = Stacking(make_base_models(), make_meta_nnls(), n_splits=5).fit(X_train, y_train)
print(stack.meta_weights_)           # веса моделей в мета-модели
pred = stack.predict(X_test)
```

## Основные результаты (среднее по 3 разбиениям)

| Модель | RMSE на test, $ |
|---|---|
| Ridge | 68 251 |
| KNN | 59 934 |
| RandomForest | 48 544 |
| ExtraTrees | 49 341 |
| HistGradientBoosting | 44 519 |
| Простое среднее | 50 112 |
| Blending (holdout 20 %) | 45 668 |
| **Stacking (K=5, NNLS)** | 44 544 |
| Stacking на in-sample предсказаниях (утечка) | 59 934 |

Stacking сравнялся с лучшей одиночной моделью и превзошёл Blending и простое усреднение; мета-модель, обученная на предсказаниях train, «выбирает» KNN с нулевой ошибкой на train и теряет треть качества. Подробности и ограничения - в [отчёте](docs/report.md).
