from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "service_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "service_model.joblib"
REPORT_PATH = BASE_DIR / "model_evaluation.txt"
MATRIX_PATH = BASE_DIR / "confusion_matrix.csv"

LABELS = [
    "barangay_clearance",
    "business_clearance",
    "residency",
    "indigency",
    "lupon_complaint",
    "vaw_report",
    "child_protection",
    "disaster_report",
]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "data/service_dataset.csv"
        )

    df = pd.read_csv(DATA_PATH)
    required_columns = {"text", "label"}
    if not required_columns.issubset(df.columns):
        raise ValueError("CSV must contain 'text' and 'label' columns.")

    df = df[["text", "label"]].copy()
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["label"] != "")]

    # Remove duplicate message text without changing the original wording.
    df["normalized_text"] = df["text"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    df = df.drop_duplicates(subset=["normalized_text"]).drop(columns=["normalized_text"])

    found_labels = set(df["label"])
    expected_labels = set(LABELS)
    if found_labels != expected_labels:
        missing = sorted(expected_labels - found_labels)
        unexpected = sorted(found_labels - expected_labels)
        raise ValueError(f"Label mismatch. Missing: {missing}; unexpected: {unexpected}")

    counts = df["label"].value_counts()
    if (counts < 2).any():
        raise ValueError("Each label needs at least two examples for a stratified train/test split.")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.20,
        random_state=42,
        stratify=df["label"],
    )

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, labels=LABELS, average="macro", zero_division=0)
    report = classification_report(
        y_test, predictions, labels=LABELS, zero_division=0, digits=3
    )
    matrix = confusion_matrix(y_test, predictions, labels=LABELS)
    matrix_df = pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in LABELS],
        columns=[f"predicted_{label}" for label in LABELS],
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    matrix_df.to_csv(MATRIX_PATH)

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        file.write("MARITESS SERVICE CLASSIFICATION MODEL EVALUATION\n")
        file.write("=" * 48 + "\n")
        file.write(f"Dataset: {DATA_PATH.name}\n")
        file.write(f"Unique examples used: {len(df)}\n")
        file.write(f"Training examples: {len(X_train)}\n")
        file.write(f"Test examples: {len(X_test)}\n")
        file.write("\nExamples per label:\n")
        file.write(counts.sort_index().to_string())
        file.write(f"\n\nAccuracy: {accuracy:.4f}\n")
        file.write(f"Macro F1-score: {macro_f1:.4f}\n\n")
        file.write("Classification report:\n")
        file.write(report)
        file.write("\n\nConfusion matrix:\n")
        file.write(matrix_df.to_string())
        file.write(
            "\n\nLIMITATION: This dataset is synthetic. Scores measure performance on this "
            "held-out split only and do not establish real-world performance. Review examples "
            "and test with representative, consented messages before any real deployment.\n"
        )

    print("\nDATASET SUMMARY")
    print("-" * 40)
    print("Unique examples used:", len(df))
    print(counts.sort_index().to_string())
    print("\nMODEL EVALUATION")
    print("-" * 40)
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Macro F1-score: {macro_f1:.3f}")
    print("\nClassification report:")
    print(report)
    print("Saved model:", MODEL_PATH)
    print("Saved evaluation report:", REPORT_PATH)
    print("Saved confusion matrix:", MATRIX_PATH)
    print("\nReminder: synthetic-data results are preliminary, not real-world validation.")


if __name__ == "__main__":
    main()
