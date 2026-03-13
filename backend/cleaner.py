"""
cleaner.py — Data cleaning, summary, anomaly detection
"""

import pandas as pd


def clean_dataframe(df: pd.DataFrame):
    """Clean CSV data and return (cleaned_df, change_log)."""
    log = []

    # Fix column names — remove spaces & special chars
    old_cols = list(df.columns)
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(" ", "_")
                  .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    if old_cols != list(df.columns):
        log.append(f"Fixed column names: {old_cols} → {list(df.columns)}")

    # Remove duplicate rows
    dupes = int(df.duplicated().sum())
    if dupes > 0:
        df = df.drop_duplicates()
        log.append(f"Removed {dupes} duplicate rows")

    # Remove fully empty rows
    empty = int(df.isnull().all(axis=1).sum())
    if empty > 0:
        df = df.dropna(how="all")
        log.append(f"Removed {empty} fully empty rows")

    # Fill missing values
    for col in df.columns:
        missing = int(df[col].isnull().sum())
        if missing > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(0)
                log.append(f"Filled {missing} missing numbers in '{col}' with 0")
            else:
                df[col] = df[col].fillna("Unknown")
                log.append(f"Filled {missing} missing text in '{col}' with 'Unknown'")

    # Strip whitespace from text columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Auto-detect and convert date columns
    for col in df.columns:
        if "date" in col or "time" in col:
            try:
                df[col] = pd.to_datetime(df[col])
                log.append(f"Converted '{col}' to datetime")
            except Exception:
                pass

    if not log:
        log.append("Data was already clean — no changes needed")

    return df, log


def get_summary(df: pd.DataFrame) -> dict:
    """Build column-level stats."""
    summary = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            summary[col] = {
                "type":    "numeric",
                "min":     float(df[col].min()),
                "max":     float(df[col].max()),
                "mean":    round(float(df[col].mean()), 2),
                "sum":     float(df[col].sum()),
                "missing": int(df[col].isnull().sum()),
            }
        else:
            top = df[col].value_counts().head(5).to_dict()
            summary[col] = {
                "type":    "text",
                "unique":  int(df[col].nunique()),
                "top_5":   {str(k): int(v) for k, v in top.items()},
                "missing": int(df[col].isnull().sum()),
            }
    return summary


def detect_anomalies(df: pd.DataFrame) -> list:
    """Detect outliers using IQR method."""
    results = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            Q1    = df[col].quantile(0.25)
            Q3    = df[col].quantile(0.75)
            IQR   = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower) | (df[col] > upper)]
            if not outliers.empty:
                results.append({
                    "column":         col,
                    "count":          len(outliers),
                    "normal_range":   {
                        "min": round(float(lower), 2),
                        "max": round(float(upper), 2)
                    },
                    "outlier_values": [float(v) for v in outliers[col].tolist()],
                })
    return results


def suggest_chart_type(question: str) -> str:
    """Suggest best chart type based on question."""
    q = question.lower()
    if any(w in q for w in ["trend", "over time", "monthly", "daily", "growth", "timeline"]):
        return "line"
    elif any(w in q for w in ["percentage", "share", "proportion", "breakdown", "distribution"]):
        return "pie"
    return "bar"
