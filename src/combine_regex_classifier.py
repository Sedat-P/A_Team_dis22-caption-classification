# -*- coding: utf-8 -*-
"""
combine_regex_classifier.py

Testet, ob eine KOMBINATION aus Regex und Classifier besser ist als der
Classifier allein. Trainiert den Classifier auf train_split, wendet Regex und
Classifier auf eval_split an und vergleicht vier Varianten:

  1. Classifier allein
  2. Regex allein
  3. Kombi ODER   (Label = 1, wenn Regex ODER Classifier es vorhersagt)
  4. Kombi UND    (Label = 1, nur wenn BEIDE es vorhersagen)

So bekommt ihr eine echte, belegte Zahl statt einer Vermutung.

Aufruf:  python combine_regex_classifier.py  [train_split.csv]  [eval_split.csv]
(ohne Argumente: zwei Dateidialoge)

WICHTIG: train_and_evaluate.py und regex_lee.py müssen im selben Ordner liegen.
"""

import os
import sys
import numpy as np
import pandas as pd

try:
    import train_and_evaluate as te
except Exception:
    raise SystemExit("train_and_evaluate.py nicht gefunden - bitte in denselben Ordner legen.")

from sklearn.metrics import f1_score, accuracy_score


def pick_file(title):
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk(); root.withdraw()
    path = filedialog.askopenfilename(
        title=title, filetypes=[("CSV", "*.csv"), ("Alle Dateien", "*.*")]) or None
    root.destroy()
    return path


def scores(Y, P):
    return {
        "Macro-F1": f1_score(Y, P, average="macro", zero_division=0),
        "Micro-F1": f1_score(Y, P, average="micro", zero_division=0),
        "Exact-Match": accuracy_score(Y, P),
    }


def main():
    train_csv = sys.argv[1] if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]) else None
    eval_csv  = sys.argv[2] if len(sys.argv) > 2 and os.path.isfile(sys.argv[2]) else None
    if not train_csv:
        train_csv = pick_file("TRAINING wählen (train_split.csv)")
    if not eval_csv:
        eval_csv = pick_file("EVALUATION wählen (eval_split.csv)")
    if not train_csv or not eval_csv:
        print("Abbruch: train_split und eval_split werden benötigt.")
        return

    # --- Daten laden (gleiche Logik wie überall) ---
    _, train_df, _ = te.load_annotated(train_csv)
    _, eval_df, _  = te.load_annotated(eval_csv)
    X_tr, Y_tr = te.get_XY(train_df)
    X_ev, Y_ev = te.get_XY(eval_df)
    print(f"Training:   {len(X_tr)} Abbildungen aus {os.path.basename(train_csv)}")
    print(f"Evaluation: {len(X_ev)} Abbildungen aus {os.path.basename(eval_csv)}")

    # --- Classifier trainieren + beide Verfahren anwenden ---
    clf = te.build_classifier()
    clf.fit(X_tr, Y_tr)
    P_clf   = clf.predict(X_ev)
    P_regex = te.regex_predict(X_ev)

    # --- Kombinationen ---
    P_or  = ((P_clf + P_regex) > 0).astype(int)        # ODER  (Vereinigung)
    P_and = ((P_clf * P_regex) > 0).astype(int)        # UND   (Schnittmenge)

    variants = {
        "Classifier allein": P_clf,
        "Regex allein":      P_regex,
        "Kombi ODER":        P_or,
        "Kombi UND":         P_and,
    }

    # --- Gesamtvergleich ---
    print("\n" + "=" * 66)
    print("  Vergleich auf dem Eval-Set")
    print("=" * 66)
    print(f"  {'Variante':20s}{'Macro-F1':>10s}{'Micro-F1':>10s}{'Exact':>10s}")
    rows = []
    for name, P in variants.items():
        sc = scores(Y_ev, P)
        print(f"  {name:20s}{sc['Macro-F1']:10.3f}{sc['Micro-F1']:10.3f}{sc['Exact-Match']:10.3f}")
        rows.append({"Variante": name, **{k: round(v, 3) for k, v in sc.items()}})

    # --- F1 pro Klasse ---
    print("\n  Macro-F1-Bestandteile (F1 pro Klasse):")
    print(f"  {'Variante':20s}" + "".join(f"{c:>11s}" for c in te.CLASSES))
    for name, P in variants.items():
        f1c = f1_score(Y_ev, P, average=None, zero_division=0, labels=range(len(te.CLASSES)))
        print(f"  {name:20s}" + "".join(f"{v:11.3f}" for v in f1c))

    # --- Fazit automatisch ---
    best = max(variants, key=lambda n: scores(Y_ev, variants[n])["Macro-F1"])
    print("\n" + "-" * 66)
    print(f"  Bestes Verfahren nach Macro-F1: {best}")
    if best == "Classifier allein":
        print("  -> Die Kombinationen verbessern das Ergebnis NICHT.")
        print("     Belegt: das gelernte Modell allein ist die beste Wahl.")
    print("-" * 66)

    out = os.path.join(os.path.dirname(os.path.abspath(eval_csv)),
                       "combine_results.csv")
    try:
        pd.DataFrame(rows).to_csv(out, sep=";", index=False, encoding="utf-8-sig")
        print(f"\nErgebnis gespeichert: {out}")
    except PermissionError:
        print("\n(Hinweis: combine_results.csv war gesperrt - nicht gespeichert.)")


if __name__ == "__main__":
    main()
