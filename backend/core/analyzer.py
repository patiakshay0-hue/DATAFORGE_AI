import pandas as pd
import numpy as np

def perform_eda(df: pd.DataFrame):
    # Summary Statistics
    summary = df.describe(include='all').transpose()
    summary = summary.fillna("N/A").to_dict(orient="index")
    
    # Correlation Matrix (numeric columns only)
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = {}
    if not numeric_df.empty:
        corr = numeric_df.corr().fillna(0)
        corr_matrix = corr.to_dict()
    
    # Missing Values
    missing_values = df.isnull().sum().to_dict()
    
    # Data for charts
    charts = {}
    
    # 1. Distribution of numeric columns
    for col in numeric_df.columns[:10]: # Limit to first 10 for performance
        hist, bin_edges = np.histogram(df[col].dropna(), bins=20)
        charts[f"dist_{col}"] = {
            "type": "bar",
            "data": [{"bin": float(bin_edges[i]), "count": int(hist[i])} for i in range(len(hist))]
        }
    
    # 2. Categorical counts
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols[:5]:
        counts = df[col].value_counts().head(10).to_dict()
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
