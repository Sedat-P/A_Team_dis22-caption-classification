# -*- coding: utf-8 -*-
"""
Robustheitspruefung ueber mehrere Seeds.

Statt EINER zufaelligen paper-getrennten Aufteilung wiederholen wir die Bewertung
ueber mehrere Seeds und berichten Mittelwert +/- Streuung. Das zeigt, dass das
Ergebnis nicht von einer einzelnen (gluecklichen oder unguenstigen) Aufteilung
abhaengt, und liefert ein grobes Mass fuer die Schwankung (gerade fuer Photo).

Pipeline identisch zu train_and_evaluate.py (wird importiert).

Benoetigt:  pip install scikit-learn pandas
Aufruf:     python evaluate_multiseed.py [csv1] [csv2]
            csv1 = Training, csv2 = Eval (optional). Beide werden gepoolt und je
            Seed neu nach Paper getrennt. Ohne Argumente: Datei-Dialoge.

WICHTIG: train_and_evaluate.py muss im selben Ordner liegen.
"""

import os
import sys
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    import train_and_evaluate as te
except ImportError:
    raise SystemExit("train_and_evaluate.py nicht gefunden - bitte in denselben Ordner legen.")

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, accuracy_score

N_SEEDS = 10
TEST_SIZE = 0.20


def pick(title):
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk(); root.withdraw()
    p = filedialog.askopenfilename(title=title, filetypes=[("CSV", "*.csv"), ("Alle", "*.*")])
    root.destroy()
    return p or None


def main():
    csvs = [a for a in sys.argv[1:] if os.path.isfile(a)]
    if not csvs:
        c1 = pick("1. CSV waehlen (Training, z. B. annotation_Z.csv)")
        if c1:
            csvs.append(c1)
            c2 = pick("2. CSV waehlen (Eval) - oder Abbrechen, wenn nur eine Datei")
            if c2:
                csvs.append(c2)
    if not csvs:
        print("Keine CSV gewaehlt. Abbruch."); return

    parts = []
    for c in csvs:
        _, usable, _ = te.load_annotated(c)
        parts.append(usable)
    pool = (pd.concat(parts, ignore_index=True)
            .drop_duplicates(subset=["source_json", "caption"])
            .reset_index(drop=True))
    if "source_json" not in pool.columns:
        raise SystemExit("'source_json' fehlt - Split nach Paper nicht moeglich.")

    X, Y = te.get_XY(pool)
    groups = pool["source_json"].values
    print(f"Gepoolt: {len(pool)} Figuren / {pool['source_json'].nunique()} Papers")
    print(f"Wiederhole paper-getrennten Split (Eval-Anteil {TEST_SIZE:.0%}) ueber "
          f"{N_SEEDS} Seeds ...\n")

    keys = ["macro", "micro", "exact"] + te.CLASSES
    vals = {k: [] for k in keys}
    for seed in range(N_SEEDS):
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
        tr, ev = next(gss.split(X, groups=groups))
        clf = te.build_classifier().fit(X[tr], Y[tr])
        yp = clf.predict(X[ev])
        Yt = Y[ev]
        vals["macro"].append(f1_score(Yt, yp, average="macro", zero_division=0))
        vals["micro"].append(f1_score(Yt, yp, average="micro", zero_division=0))
        vals["exact"].append(accuracy_score(Yt, yp))
        for j, c in enumerate(te.CLASSES):
            vals[c].append(f1_score(Yt[:, j], yp[:, j], zero_division=0))

    def stat(v):
        a = np.array(v)
        return f"{a.mean():.3f} +/- {a.std():.3f}   (min {a.min():.3f}, max {a.max():.3f})"

    print(f"Ergebnisse ueber {N_SEEDS} Seeds (Mittel +/- Std):")
    print(f"  Macro-F1    : {stat(vals['macro'])}")
    print(f"  Micro-F1    : {stat(vals['micro'])}")
    print(f"  Exact-Match : {stat(vals['exact'])}")
    print("  F1 pro Klasse:")
    for c in te.CLASSES:
        print(f"     {c:8s} : {stat(vals[c])}")
    print("\nFuer die Praesi: 'Ueber 10 paper-getrennte Aufteilungen liegt Macro-F1 "
          "stabil bei X +/- Y' - zeigt, dass die Zahl nicht vom Zufall abhaengt.")


if __name__ == "__main__":
    main()
