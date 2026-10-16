import os
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
MODEL_DIR = "models"

TARGET = "ProdTaken"

os.makedirs(MODEL_DIR, exist_ok=True)

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET].astype(int)

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET].astype(int)

numeric_features = X_train.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

categorical_features = [
    col for col in X_train.columns
    if col not in numeric_features
]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

models = {
    "LogisticRegression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42
    )
}

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Tourism_Purchase_Prediction")

results = []
best_f1 = -1
best_model = None
best_name = None
best_metrics = None

for name, classifier in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    with mlflow.start_run(run_name=name):

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(
                precision_score(y_test, predictions, zero_division=0)
            ),
            "recall": float(
                recall_score(y_test, predictions, zero_division=0)
            ),
            "f1_score": float(
                f1_score(y_test, predictions, zero_division=0)
            ),
            "roc_auc": float(roc_auc_score(y_test, probabilities))
        }

        mlflow.log_param("model_name", name)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.20)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model"
        )

        results.append({
            "model": name,
            **metrics
        })

        print(f"\nModel: {name}")
        print(json.dumps(metrics, indent=2))

        print("\nClassification report:")
        print(classification_report(
            y_test,
            predictions,
            zero_division=0
        ))

        print("Confusion matrix:")
        print(confusion_matrix(y_test, predictions))

        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_model = pipeline
            best_name = name
            best_metrics = metrics

results_df = pd.DataFrame(results)
results_df.to_csv("model_comparison.csv", index=False)

joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")

with open(f"{MODEL_DIR}/best_metrics.json", "w") as f:
    json.dump({
        "best_model": best_name,
        "selection_metric": "f1_score",
        "metrics": best_metrics
    }, f, indent=4)

print("\nModel comparison:")
print(results_df.to_string(index=False))

print(f"\nBest model: {best_name}")
print(f"Best F1 score: {best_f1:.4f}")
print(f"Saved to: {MODEL_DIR}/best_model.pkl")
