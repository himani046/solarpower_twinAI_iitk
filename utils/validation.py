from __future__ import annotations

from typing import Iterable

import pandas as pd


def validate_columns(df: pd.DataFrame, required: Iterable[str]) -> tuple[bool, list[str]]:
    required = list(required)
    missing = [column for column in required if column not in df.columns]
    return not missing, missing


def load_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame()
    return numeric.describe().T
