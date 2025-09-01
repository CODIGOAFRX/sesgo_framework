import os, joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)

SEED = 42

# Rutas
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(ROOT, "data", "cv_train.csv")
MODEL_PATH = os.path.join(ROOT, "models", "modelo_selector_cv.joblib")
REPORT_PATH = os.path.join(ROOT, "reports", "metrics.txt")

# Carga y limpieza mínima
df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates(subset=["id_cv"]).dropna(subset=["texto_cv", "label"])

X = df["texto_cv"].astype(str)
y = df["label"].astype(int)

# Split
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

# Pipeline simple: TF-IDF + Regresión Logística
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=50000, min_df=5)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs"))
])

# Entrenar
pipe.fit(Xtr, ytr)

# Evaluación en test
probs = pipe.predict_proba(Xte)[:, 1]
preds = (probs >= 0.5).astype(int)

print("TEST AUC-ROC:", roc_auc_score(yte, probs))
print("TEST AUC-PR :", average_precision_score(yte, probs))
print(classification_report(yte, preds, digits=3))
print("Matriz de confusión:\n", confusion_matrix(yte, preds))

# Guardar artefactos y métricas
os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
joblib.dump(pipe, MODEL_PATH)

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(f"TEST AUC-ROC: {roc_auc_score(yte, probs):.3f}\n")
    f.write(f"TEST AUC-PR : {average_precision_score(yte, probs):.3f}\n")
    f.write(str(classification_report(yte, preds, digits=3)))
    f.write("\nMatriz de confusión:\n")
    f.write(np.array2string(confusion_matrix(yte, preds)))
