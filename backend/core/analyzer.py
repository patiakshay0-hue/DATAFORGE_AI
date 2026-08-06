import pandas as pd
import numpy as np

SAMPLE_CAP = 5000

# Correlations are display values in a heatmap; full float64 precision just
# inflates the JSON payload (a 100-column frame is 10k numbers).
CORR_DECIMALS = 4


def _sample(df: pd.DataFrame) -> pd.DataFrame:
    return df if len(df) <= SAMPLE_CAP else df.sample(n=SAMPLE_CAP, random_state=42)


def _describe_categorical(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """count/unique/top/freq for text and boolean columns, one pass per column.

    A single value_counts yields all three: its length is the distinct count,
    and its first entry is the most frequent value and that value's count.
    Modestly cheaper than letting describe() derive them separately.

    Runs on the full frame, not the EDA sample: `unique` is also reported by
    get_schema(), where it is exact, and a sampled estimate here would
    contradict it — while `freq` reads as an absolute row count.
    """
    rows = {}
    for col in cols:
        counts = df[col].value_counts(dropna=True)
        rows[col] = {
            "count": int(df[col].count()),
            "unique": int(len(counts)),
            "top": counts.index[0] if len(counts) else np.nan,
            "freq": int(counts.iloc[0]) if len(counts) else np.nan,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    """describe(), split by dtype so each half takes the cheaper path.

    Numeric and datetime columns keep pandas' describe — it is already a tight
    vectorised pass, and datetimes need its min/max/quantiles to show a date
    range. Text and boolean columns go through _describe_categorical. Both
    halves run on the full frame, so every reported statistic is exact and the
    output matches describe(include='all') stat for stat.
    """
    stat_cols = set(df.select_dtypes(include=[np.number, "datetime", "datetimetz"]).columns)
    other_cols = [c for c in df.columns if c not in stat_cols]

    frames = []
    if stat_cols:
        stat_frame = df[[c for c in df.columns if c in stat_cols]]
        # include="all" so a datetime-only selection still gets described rather
        # than coming back empty for want of a numeric column.
        frames.append(stat_frame.describe(include="all").transpose())
    if other_cols:
        frames.append(_describe_categorical(df, other_cols))

    if not frames:
        return pd.DataFrame()

    summary = pd.concat(frames)
    # Restore original column order — the UI shows the first N rows.
    return summary.reindex([c for c in df.columns if c in summary.index])


def perform_eda(df: pd.DataFrame):
    eda_df = _sample(df)

    summary_df = _summary(df)
    summary = {} if summary_df.empty else summary_df.fillna("N/A").to_dict(orient="index")

    numeric_df = eda_df.select_dtypes(include=[np.number])
    corr_matrix = {}
    if not numeric_df.empty:
        corr = numeric_df.corr().fillna(0).round(CORR_DECIMALS)
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
        "columns": len(df.columns),
        "sampled": len(df) > SAMPLE_CAP,
    }
