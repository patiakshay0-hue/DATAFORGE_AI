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
        
        col_type = "numeric"
        if "object" in dtype or "category" in dtype:
            col_type = "categorical"
        elif "datetime" in dtype:
            col_type = "datetime"
        elif "bool" in dtype:
            col_type = "boolean"
            
        schema.append({
            "name": col,
            "type": col_type,
            "dtype": dtype,
            "missing": missing,
            "unique": unique,
            "sample": df[col].dropna().head(3).tolist()
        })
    return schema
