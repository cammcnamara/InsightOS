import pandas as pd


def generate_summary(df: pd.DataFrame) -> tuple:
    summary = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isnull().sum().values,
            "Unique Values": df.nunique().values,
        }
    )

    # Summary Stats
    num_rows = len(df)
    num_cols = df.shape[1]
    num_null_cols = (df.isnull().sum() > 0).sum()
    num_categorical = (df.dtypes == "object").sum()

    stats = {
        "rows": num_rows,
        "columns": num_cols,
        "null_cols": num_null_cols,
        "categorical_cols": num_categorical,
    }

    # Warnings
    warnings = []
    for _, row in summary.iterrows():
        if row["Missing Values"] > 0.5 * num_rows:
            warnings.append(f"⚠️ Column '{row['Column']}' has over 50% missing values.")
        if row["Unique Values"] == num_rows:
            warnings.append(
                f"🔎 Column '{row['Column']}' has unique values per row (possible ID)."
            )
        if row["Unique Values"] > 100 and row["Data Type"] == "object":
            warnings.append(f"📛 Column '{row['Column']}' has very high cardinality.")

    return summary, stats, warnings
