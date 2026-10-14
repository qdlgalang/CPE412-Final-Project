"""Train baseline service and urgency classifiers from the supplied synthetic dataset."""
from pathlib import Path
import json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data.csv"
OUT = ROOT / "models"
OUT.mkdir(exist_ok=True)
df = pd.read_csv(DATA).dropna(subset=["text_input", "extracted_service", "urgency_label"])
df["text_input"] = df["text_input"].astype(str)

reports = {}
for target, filename in [("extracted_service", "service_model.joblib"), ("urgency_label", "urgency_model.joblib")]:
    X_train, X_test, y_train, y_test = train_test_split(
        df["text_input"], df[target].astype(str), test_size=0.2, random_state=42,
        stratify=df[target].astype(str)
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, strip_accents="unicode")),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    joblib.dump(model, OUT / filename)
    reports[target] = {
        "test_rows": len(y_test),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0)
    }
    print(f"\n{target}: accuracy={reports[target]['accuracy']:.4f} (test rows={len(y_test)})")
    print(classification_report(y_test, pred, zero_division=0))
(OUT / "baseline_metrics.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
print("\nModels and baseline_metrics.json saved in ./models")
print("IMPORTANT: This is a preliminary random split, not independent real-world validation.")
