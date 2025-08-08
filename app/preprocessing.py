from sklearn.preprocessing import LabelEncoder


def clean_and_encode(df):
    df = df.dropna()
    label_encoders = {}
    original_values = {}
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        original_values[col] = dict(zip(le.fit_transform(df[col]), df[col]))
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders, original_values
