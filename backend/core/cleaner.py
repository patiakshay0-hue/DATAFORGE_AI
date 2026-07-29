"""Missing-value handling for the EDA view.

Reports per-column missing stats with a type-aware suggested strategy, and
applies a user-chosen strategy per column to produce a cleaned DataFrame.
"""

import pandas as pd
import numpy as np

# method → columns it's valid for
NUMERIC_METHODS = ["median", "mean", "zero", "ffill", "bfill", "constant", "drop_rows", "drop_column"]
CATEGORICAL_METHODS = ["mode", "constant", "ffill", "bfill", "drop_rows", "drop_column"]


def _kind(series) -> str:
    return "numeric" if pd.api.types.is_numeric_dtype(series) else "categorical"


def missing_report(df: pd.DataFrame) -> dict:
    """Per-column missing stats for every column that has missing values."""
    n = len(df)
    columns = []
    for col in df.columns:
        miss = int(df[col].isnull().sum())
        if miss == 0:
            continue
        kind = _kind(df[col])
        columns.append({
            "column": col,
            "missing": miss,
            "pct": round(100 * miss / n, 1) if n else 0,
            "kind": kind,
            "suggested": "median" if kind == "numeric" else "mode",
            "methods": NUMERIC_METHODS if kind == "numeric" else CATEGORICAL_METHODS,
        })
    return {
        "columns": columns,
        "total_missing": int(df.isnull().sum().sum()),
        "rows": n,
        "n_affected": len(columns),
    }


def handle_missing(df: pd.DataFrame, strategies: list) -> tuple[pd.DataFrame, dict]:
    """Apply per-column strategies. `strategies` is a list of
    {column, method, value?}. Returns (cleaned_df, change_summary)."""
    df = df.copy()
    rows_before = len(df)
    cells_filled = 0
    dropped_cols = []
    errors = []

    for st in strategies:
        col = st.get("column")
        method = st.get("method")
        if not col or col not in df.columns or not method:
            continue
        try:
            mask = df[col].isnull()
            cnt = int(mask.sum())

            if method == "drop_column":
                df = df.drop(columns=[col])
                dropped_cols.append(col)
                continue
            if method == "drop_rows":
                df = df[~mask]
                continue
            if cnt == 0:
                continue

            if method == "mean":
                fill = df[col].mean()
            elif method == "median":
                fill = df[col].median()
            elif method == "zero":
                fill = 0
            elif method == "mode":
                modes = df[col].mode()
                fill = modes.iloc[0] if len(modes) else ""
            elif method == "constant":
                fill = st.get("value", "Unknown")
            elif method in ("ffill", "bfill"):
                # chain both directions so no NaN survives at the edges
                df[col] = df[col].ffill().bfill() if method == "ffill" else df[col].bfill().ffill()
                cells_filled += cnt
                continue
            else:
                continue

            df[col] = df[col].fillna(fill)
            cells_filled += cnt
        except Exception as e:
            errors.append(f"{col}: {str(e)[:80]}")

    summary = {
        "rows_before": rows_before,
        "rows_after": len(df),
        "rows_removed": rows_before - len(df),
        "cells_filled": cells_filled,
        "columns_dropped": dropped_cols,
        "remaining_missing": int(df.isnull().sum().sum()),
        "errors": errors,
    }
    return df, summary
