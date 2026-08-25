# -*- coding: utf-8 -*-
"""
Experiment: Hilft Multilabel-Trainingsdaten dem Classifier?

Vergleicht FAIR auf denselben Cross-Validation-Folds:
  (0) Regex A                      – Baseline, kein Training
  (A) Classifier "single-only"     – trainiert NUR auf reinen Einzel-Label-Bildern
  (B) Classifier "single+multi"    – trainiert auf Einzel- UND Multilabel-Bildern

Wichtig: Beide Classifier haben dieselbe Architektur (TF-IDF + One-vs-Rest
Logistic Regression) und werden auf demselben Test-Set ausgewertet (inkl.
Multilabel-Bildern). Der EINZIGE Unterschied ist, welche Zeilen ins Training
des jeweiligen Folds dürfen. So misst man sauber den Effekt der Trainingsdaten.

Benötigt:  pip install scikit-learn pandas numpy
Aufruf:    python compare_classifiers.py  annotation_task_1000.csv
"""

import os
import re
import sys
import numpy as np
import pandas as pd

# Windows-Konsole UTF-8-sicher machen (verhindert Crash bei Umlauten/Sonderzeichen)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score, classification_report


# ======================================================================
#  Konfiguration  (identisch zu den anderen Skripten halten!)
# ======================================================================
TEXT_COL  = "caption"
CLASSES   = ["plot", "diagram", "photo"]
FLAG_COLS = {"plot": "manual_plot", "diagram": "manual_diagram", "photo": "manual_photo"}
N_FOLDS      = 5
RANDOM_STATE = 42

REGEX_KEYWORDS = {
    "plot": [
        r"bar ?chart", r"bar ?graph", r"scatter", r"line ?(chart|graph|plot)",
        r"box ?plot", r"violin", r"histogram", r"heat ?map", r"dot ?plot",
        r"\bcurve", r"\bplot(ted|s)?\b", r"\baxis\b", r"\baxes\b",
        r"x-axis", r"y-axis", r"quantif", r"distribution of", r"\bvs\.?\b",
    ],
    "diagram": [
        r"schematic", r"workflow", r"\bdiagram", r"illustrat", r"pipeline",
        r"architecture", r"flow ?chart", r"cartoon", r"\bmodel of\b",
        r"overview of", r"timeline", r"\bsetup\b", r"experimental design",
    ],
    "photo": [
        r"microscop", r"fluoresc", r"immunofluoresc", r"confocal",
        r"micrograph", r"stain", r"\bblot", r"western blot", r"histolog",
        r"\bimage", r"\bphoto", r"brightfield", r"\bdapi\b", r"\bgel\b",
        r"electron microscop", r"\bsem\b", r"\btem\b", r"representative image",
    ],
}
_RX = {c: [re.compile(p, re.I) for p in pats] for c, pats in REGEX_KEYWORDS.items()}


def flag(v):
    return 1 if str(v).strip() in ("1", "1.0", "x", "X", "true", "True") else 0


def make_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                  sublinear_tf=True, stop_words="english")),
        ("clf", OneVsRestClassifier(
            LogisticRegression(max_iter=2000, class_weight="balanced"))),
    ])


def regex_predict(texts):
    P = np.zeros((len(texts), len(CLASSES)), dtype=int)
    for i, t in enumerate(texts):
        t = t or ""
        for j, c in enumerate(CLASSES):
            if any(rx.search(t) for rx in _RX[c]):
                P[i, j] = 1
    return P


def metrics(y_true, y_pred, multi_mask=None):
    out = {}
    for j, c in enumerate(CLASSES):
        out[c] = f1_score(y_true[:, j], y_pred[:, j], zero_division=0)
    out["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    out["exact"] = accuracy_score(y_true, y_pred)
    if multi_mask is not None and multi_mask.sum() > 0:
        out["exact_multi"] = accuracy_score(y_true[multi_mask], y_pred[multi_mask])
    else:
        out["exact_multi"] = float("nan")
    return out


def pick_csv():
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        return sys.argv[1]
    if os.path.isfile("annotation_task_1000.csv"):
        return "annotation_task_1000.csv"
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk(); root.withdraw()
    path = filedialog.askopenfilename(
        title="Annotierte CSV wählen",
        filetypes=[("CSV", "*.csv"), ("Alle Dateien", "*.*")])
    root.destroy()
    return path or None


def main():
    csv_path = pick_csv()
    if not csv_path:
        print("Keine CSV gewählt. Abbruch.")
        return

    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig", dtype=str).fillna("")
    for cls, col in FLAG_COLS.items():
        if col not in df.columns:
            raise SystemExit(f"Spalte '{col}' fehlt in der CSV.")
        df[f"y_{cls}"] = df[col].map(flag)

    Yall = df[[f"y_{c}" for c in CLASSES]].values.astype(int)
    usable = (Yall.sum(axis=1) > 0) & (df[TEXT_COL].str.strip().values != "")
    df = df[usable].reset_index(drop=True)
    X = df[TEXT_COL].astype(str).values
    Y = df[[f"y_{c}" for c in CLASSES]].values.astype(int)

    n_labels = Y.sum(axis=1)
    single_mask = n_labels == 1          # reine Einzel-Label-Bilder
    multi_mask = n_labels >= 2           # Multilabel-Bilder

    print(f"Datei: {csv_path}")
    print(f"Nutzbare Abbildungen   : {len(df)}")
    print(f"  davon Einzel-Label   : {int(single_mask.sum())}")
    print(f"  davon Multilabel     : {int(multi_mask.sum())}")
    for j, c in enumerate(CLASSES):
        print(f"  positive '{c}'        : {int(Y[:, j].sum())}")

    if len(df) < 30 or single_mask.sum() < 15:
        print("\n⚠ Noch zu wenige (Einzel-)Label für eine belastbare Auswertung.")
        return

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Out-of-Fold-Vorhersagen sammeln (gleiche Test-Indizes für beide Varianten)
    pred_A = np.zeros_like(Y)   # single-only
    pred_B = np.zeros_like(Y)   # single+multi

    for train_idx, test_idx in kf.split(X):
        # Variante B: alle Trainingszeilen
        pipeB = make_pipeline()
        pipeB.fit(X[train_idx], Y[train_idx])
        pred_B[test_idx] = pipeB.predict(X[test_idx])

        # Variante A: nur reine Einzel-Label-Zeilen aus dem Trainingsteil
        sub = train_idx[single_mask[train_idx]]
        if len(sub) >= len(CLASSES):
            pipeA = make_pipeline()
            pipeA.fit(X[sub], Y[sub])
            pred_A[test_idx] = pipeA.predict(X[test_idx])

    pred_R = regex_predict(X)   # Regex braucht kein Training

    mR = metrics(Y, pred_R, multi_mask)
    mA = metrics(Y, pred_A, multi_mask)
    mB = metrics(Y, pred_B, multi_mask)

    # --- Ergebnis-Tabelle ---
    print("\n" + "=" * 78)
    print("  Vergleich (5-fold CV, alle Bilder im Test-Set)")
    print("=" * 78)
    head = f"  {'Ansatz':28}" + "".join(f"{c:>10}" for c in CLASSES) + \
           f"{'Macro-F1':>10}{'Exact':>8}{'Exact(ML)':>11}"
    print(head)
    print("  " + "-" * 76)
    for name, m in [("Regex A (Baseline)", mR),
                    ("Classifier single-only", mA),
                    ("Classifier single+multi", mB)]:
        row = f"  {name:28}" + "".join(f"{m[c]:>10.3f}" for c in CLASSES)
        row += f"{m['macro_f1']:>10.3f}{m['exact']:>8.3f}{m['exact_multi']:>11.3f}"
        print(row)
    print("  " + "-" * 76)
    print("  Exact(ML) = Anteil korrekter Vorhersagen NUR auf Multilabel-Bildern")

    # Interpretation
    print("\nKurz-Interpretation:")
    d_macro = mB["macro_f1"] - mA["macro_f1"]
    if abs(d_macro) < 0.01:
        print("  - Multilabel-Training macht beim Macro-F1 kaum einen Unterschied.")
    elif d_macro > 0:
        print(f"  - single+multi ist beim Macro-F1 besser (+{d_macro:.3f}).")
    else:
        print(f"  - single-only ist beim Macro-F1 besser ({d_macro:.3f}).")
    if not np.isnan(mA["exact_multi"]) and not np.isnan(mB["exact_multi"]):
        print(f"  - Auf Multilabel-Bildern: single-only {mA['exact_multi']:.3f} "
              f"vs. single+multi {mB['exact_multi']:.3f} "
              f"(erwartet: single-only schwächer, weil es solche Fälle nie sah).")

    # Detailergebnisse speichern
    out = df.copy()
    for j, c in enumerate(CLASSES):
        out[f"true_{c}"] = Y[:, j]
        out[f"predA_{c}"] = pred_A[:, j]
        out[f"predB_{c}"] = pred_B[:, j]
    out["is_multilabel"] = multi_mask.astype(int)
    out_path = os.path.splitext(csv_path)[0] + "_compare_results.csv"
    out.to_csv(out_path, sep=";", index=False, encoding="utf-8-sig")
    print(f"\nDetailergebnisse gespeichert: {out_path}")


if __name__ == "__main__":
    main()
