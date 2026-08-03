import pandas as pd
import numpy as np

SAMPLE_CAP = 5000


def _sample(df: pd.DataFrame) -> pd.DataFrame:
    return df if len(df) <= SAMPLE_CAP else df.sample(n=SAMPLE_CAP, random_state=42)


def perform_eda(df: pd.DataFrame):
    eda_df = _sample(df)

    summary = df.describe(include='all').transpose()
    summary = summary.fillna("N/A").to_dict(orient="index")

    numeric_df = eda_df.select_dtypes(include=[np.number])
    corr_matrix = {}
    if not numeric_df.empty:
        corr = numeric_df.corr().fillna(0)
        corr_matrix = corr.to_dict()

    missing_values = df.isnull().sum().to_dict()

    charts = {}

    for col in numeric_df.columns[:10]:
        vals = eda_df[col].dropna()
        if len(vals):
            hist, bin_edges = np.histogram(vals, bins=20)
            charts[f"dist_{col}"] = {
                "type": "bar",
                "data": [{"bin": float(bin_edges[i]), "count": int(hist[i])} for i in range(len(hist))]
            }

    # Exclude-based selection so this keeps working across pandas 2/3/4, where
    # the default string dtype changes name ("object" -> "str").
    categorical_cols = eda_df.select_dtypes(
        exclude=[np.number, "datetime", "datetimetz", "timedelta"]
    ).columns
    for col in categorical_cols[:5]:
        counts = eda_df[col].value_counts().head(10).to_dict()
        charts[f"count_{col}"] = {
            "type": "pie",
            "data": [{"name": k, "value": v} for k, v in counts.items()]
        }

    return {
        "summary": summary,
        "correlation": corr_matrix,
        "missing": missing_values,
        "charts": charts,
        "rows": len(df),
        "columns": len(df.columns)
    }
