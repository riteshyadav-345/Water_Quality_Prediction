"""Train and evaluate a reproducible baseline water potability classifier."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "water_potability.csv"
MODEL_PATH = ROOT / "artifacts" / "water_potability_model.joblib"
FEATURES = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate", "Conductivity",
    "Organic_carbon", "Trihalomethanes", "Turbidity",
]
TARGET = "Potability"


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    missing_columns = sorted(set(FEATURES + [TARGET]) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    data = data[FEATURES + [TARGET]].copy()
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data[data[TARGET].isin([0, 1])]
    if data.empty or data[TARGET].nunique() != 2:
        raise ValueError("Potability must contain both classes 0 and 1.")

    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURES], data[TARGET].astype(int),
        test_size=0.2, random_state=42, stratify=data[TARGET].astype(int),
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, n_jobs=1
        )),
    ])
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    print(f"Rows: {len(data)} | Train: {len(x_train)} | Test: {len(x_test)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("Classification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix (rows=true, columns=predicted; labels 0,1):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
