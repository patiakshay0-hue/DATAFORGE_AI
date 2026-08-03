import pandas as pd
import io
import json

def load_data(file_content: bytes, filename: str):
    ext = filename.split(".")[-1].lower()
    
    if ext == "csv":
        df = pd.read_csv(io.BytesIO(file_content))
    elif ext in ["xlsx", "xls"]:
        df = pd.read_excel(io.BytesIO(file_content))
    elif ext == "json":
        df = pd.read_json(io.BytesIO(file_content))
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    return df

def get_schema(df: pd.DataFrame):
    schema = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing = int(df[col].isnull().sum())
        unique = int(df[col].nunique())
        
        # Use the pandas type API rather than substring-matching the dtype name:
        # pandas 3 renamed the default string dtype from "object" to "str", so a
        # plain `"object" in dtype` check silently labels text columns numeric.
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            col_type = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(series):
            col_type = "datetime"
        elif pd.api.types.is_numeric_dtype(series):
            col_type = "numeric"
        else:
            col_type = "categorical"

        schema.append({
            "name": col,
            "type": col_type,
            "dtype": dtype,
            "missing": missing,
            "unique": unique,
            "sample": df[col].dropna().head(3).tolist()
        })
    return schema
