import pandas as pd
from mysql_connect import get_sql
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

df = get_sql()

X = df.drop("ChurnValue", axis=1)
y = df["ChurnValue"]
categorical_cols = X.select_dtypes(include=["object"]).columns
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=300),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.7,
        colsample_bytree=0.7,
        eval_metric="logloss"
    )
}

for name, model in models.items():
    clf = Pipeline(steps=[("preprocess", preprocess),
                          ("model", model)])
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    prob = clf.predict_proba(X_test)[:, 1]

    print("\n", name)
    print(classification_report(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, prob))



