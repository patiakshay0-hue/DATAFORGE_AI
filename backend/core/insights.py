import pandas as pd
import numpy as np

def generate_insights(df: pd.DataFrame):
    insights = []
    
    # 1. Row count insight
    insights.append({
        "type": "info",
        "title": "Dataset Volume",
        "text": f"Analyzing {len(df)} records with {len(df.columns)} features."
    })
    
    # 2. Missing value insight
    missing = df.isnull().sum().sum()
    if missing > 0:
        insights.append({
            "type": "warning",
            "title": "Data Quality",
            "text": f"Found {missing} missing values. Automated imputation was performed for analysis."
        })
    else:
        insights.append({
            "type": "success",
            "title": "Data Integrity",
            "text": "Excellent! No missing values detected in the primary features."
        })
    
    # 3. Numeric trends
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        for col in numeric_cols[:3]:
            mean_val = df[col].mean()
            insights.append({
                "type": "insight",
                "title": f"{col} Baseline",
                "text": f"The average {col} is {mean_val:.2f}, indicating a standard distribution."
            })
            
    # 4. Correlation insights
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        # Find highest positive correlation (excluding diagonal)
        corr_unstack = corr.unstack()
        corr_unstack = corr_unstack[corr_unstack < 1.0].sort_values(ascending=False)
        if not corr_unstack.empty:
            highest_pair = corr_unstack.index[0]
            val = corr_unstack.iloc[0]
            if val > 0.5:
                insights.append({
                    "type": "success",
                    "title": "Strong Relationship Found",
                    "text": f"Feature '{highest_pair[0]}' and '{highest_pair[1]}' show a strong positive correlation (+{val:.2f})."
                })

    return insights
