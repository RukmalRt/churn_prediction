import pandas as pd
from mysql_connect import get_sql
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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

model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.7,
        colsample_bytree=0.7,
        eval_metric="logloss"
    )

clf = Pipeline(steps = [('preprocess', preprocess), ('model', model)])

clf.fit(X, y)

df["ChurnProbability"] = clf.predict_proba(X)[:,1]
df["ChurnRiskPercent"] = (df["ChurnProbability"] * 100).round(2)

df.to_csv("churn_risk_output.csv", index=False)