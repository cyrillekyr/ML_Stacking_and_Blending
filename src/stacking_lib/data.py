"""Загрузка и подготовка датасета California Housing (цены на жильё)."""
from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

URL = ("https://raw.githubusercontent.com/ageron/handson-ml2/master/"
       "datasets/housing/housing.csv")
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "housing.csv"
OCEAN = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
TARGET = "median_house_value"


def load_housing(path: str | Path = DEFAULT_PATH, download: bool = True):
    """Вернуть (X, y). Если файла нет - скачать его по URL."""
    path = Path(path)
    if not path.exists():
        if not download:
            raise FileNotFoundError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(URL, path)
    df = pd.read_csv(path)
    y = df.pop(TARGET)
    # Признаки-отношения: детерминированные, поэтому утечки данных нет.
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]
    # One-hot с фиксированным набором категорий (одинаковые столбцы в train/test).
    cat = pd.Categorical(df.pop("ocean_proximity"), categories=OCEAN)
    df = pd.concat([df, pd.get_dummies(cat, prefix="ocean", dtype=float)], axis=1)
    return df, y
