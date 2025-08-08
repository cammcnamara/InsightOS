# model.py
import shap
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split


def train_model(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    model = XGBClassifier() if y.nunique() <= 10 else XGBRegressor()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test, check_additivity=False)
    return model, shap_values, X_test
