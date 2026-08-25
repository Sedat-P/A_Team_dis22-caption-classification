# -*- coding: utf-8 -*-
"""
Keywords datengetrieben aus den Annotationen ableiten.

Statt die Schluesselwoerter der Regex-Baseline per Hand zu raten, leiten wir sie
aus den eigenen annotierten Captions ab: per TF-IDF + Logistic Regression
bestimmen wir je Klasse die trennschaerfsten Begriffe. Das liefert die
wissenschaftliche Grundlage fuer die Keyword-Liste (und zeigt, welche
hand-gewaehlten Begriffe von den Daten bestaetigt werden).

Erzeugt:  keywords_derived.csv  (Top-Begriffe je Klasse mit Gewicht)

Benoetigt:  pip install scikit-learn pandas
Aufruf:     python derive_keywords.py [annotation.csv]
            (ohne Argument: Datei-Dialog)
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CLASSES = ["plot", "diagram", "photo"]
FLAGS = {"plot": "manual_plot", "diagram": "manual_diagram", "photo": "manual_photo"}
TOP_N = 25
MIN_DF = 5    # ein Begriff muss in >= 5 Captions vorkommen (filtert Rauschen)


def flag(v):
    return 1 if str(v).strip() in ("1", "1.0", "x", "X", "true", "True") else 0


def pick():
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        return sys.argv[1]
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk(); root.withdraw()
    p = filedialog.askopenfilename(title="Annotierte CSV waehlen",
                                   filetypes=[("CSV", "*.csv"), ("Alle", "*.*")])
    root.destroy()
    return p or None


def get_labels(df, cls):
    ycol = f"y_{cls}"
    if ycol in df.columns:
        return df[ycol].map(flag).values
    return df[FLAGS[cls]].map(flag).values


def main():
    path = pick()
    if not path:
        print("Keine CSV gewaehlt. Abbruch."); return
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    if "caption" not in df.columns:
        raise SystemExit("Spalte 'caption' fehlt.")

    X = df["caption"].astype(str).values
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=MIN_DF,
                          sublinear_tf=True, stop_words="english")
    Xt = vec.fit_transform(X)
    feats = np.array(vec.get_feature_names_out())

    rows = []
    for cls in CLASSES:
        y = get_labels(df, cls)
        if y.sum() == 0:
            print(f"Klasse {cls}: keine positiven Beispiele, uebersprungen.")
            continue
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xt, y)
        coef = clf.coef_[0]
        top = np.argsort(coef)[::-1][:TOP_N]
        terms = [(feats[i], round(float(coef[i]), 3)) for i in top]
        print(f"\n=== {cls.upper()} — Top-{TOP_N} trennschaerfste Begriffe ===")
        print("   ", ", ".join(t for t, _ in terms))
        for rank, (t, w) in enumerate(terms, 1):
            rows.append({"klasse": cls, "rang": rank, "begriff": t, "gewicht": w})

    out = os.path.join(os.path.dirname(os.path.abspath(path)) or ".",
                       "keywords_derived.csv")
    pd.DataFrame(rows).to_csv(out, sep=";", index=False, encoding="utf-8-sig")
    print(f"\nGeschrieben: {out}")
    print("Das ist die datengetriebene Grundlage fuer deine Keyword-Liste.")


if __name__ == "__main__":
    main()
