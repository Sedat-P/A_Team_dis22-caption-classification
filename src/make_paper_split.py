# -*- coding: utf-8 -*-
"""
Paper-reiner Train/Eval-Split (Option 2).

Nimmt deine ZWEI gelabelten CSVs (Training + Eval), fuehrt sie zusammen, gruppiert
nach Paper (source_json) und teilt die PAPERS in zwei Toepfe - garantiert ohne dass
ein Paper in beiden landet. Damit ist die Bewertung frei von Leakage auf Paper-Ebene.

Erzeugt:  train_split.csv  und  eval_split.csv   (im annotate.py-Format)

Danach:   python evaluate_on_goldset.py
          -> als Training 'train_split.csv', als Eval 'eval_split.csv' waehlen.

Benoetigt:  pip install scikit-learn pandas
Aufruf:     python make_paper_split.py [train.csv] [eval.csv]
            (ohne Argumente: zwei Datei-Dialoge)
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ======================================================================
#  Konfiguration
# ======================================================================
TEST_SIZE = 0.20        # ~20 % der Figuren kommen ins Eval-Set
RANDOM_STATE = 42

KEEP = ["annotation_id", "source_json", "caption", "Maschine_Label",
        "manual_plot", "manual_photo", "manual_diagram", "unclear",
        "notes", "manual_label"]
FLAGS = {"plot": "manual_plot", "diagram": "manual_diagram", "photo": "manual_photo"}


def flag(v):
    return 1 if str(v).strip() in ("1", "1.0", "x", "X", "true", "True") else 0


def pick(title):
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk(); root.withdraw()
    p = filedialog.askopenfilename(title=title,
                                   filetypes=[("CSV", "*.csv"), ("Alle", "*.*")])
    root.destroy()
    return p or None


def load(path):
    d = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    if "source_json" not in d.columns:
        raise SystemExit(f"'source_json' fehlt in {path}")
    for c in KEEP:
        if c not in d.columns:
            d[c] = ""
    return d[KEEP]


def labeled_mask(d):
    """Zeile ist nutzbar, wenn mindestens ein Klassen-Flag gesetzt ist."""
    return (d["manual_plot"].map(flag) + d["manual_diagram"].map(flag)
            + d["manual_photo"].map(flag)) > 0


def show_dist(d, label):
    lab = d[labeled_mask(d)]
    print(f"  {label}: {len(d)} Figuren ({len(lab)} gelabelt) / {d['source_json'].nunique()} Papers")
    for c, col in FLAGS.items():
        print(f"      {c:8s}: {int(lab[col].map(flag).sum())}")


def main():
    tcsv = sys.argv[1] if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]) else None
    ecsv = sys.argv[2] if len(sys.argv) > 2 and os.path.isfile(sys.argv[2]) else None
    if not tcsv:
        tcsv = pick("TRAININGS-CSV waehlen (z. B. annotation_Z.csv)")
    if tcsv and not ecsv:
        ecsv = pick("EVAL-CSV waehlen (annotation_task_eval.csv, selbst gelabelt)")
    if not tcsv or not ecsv:
        print("CSV fehlt. Abbruch."); return

    a = load(tcsv)
    b = load(ecsv)
    alldf = pd.concat([a, b], ignore_index=True)
    before = len(alldf)
    alldf = alldf.drop_duplicates(subset=["source_json", "caption"]).reset_index(drop=True)
    # nur gelabelte Zeilen behalten (Skip/Needs-Review fliegen raus)
    alldf = alldf[labeled_mask(alldf)].reset_index(drop=True)
    print(f"Zusammengefuehrt: {before} Zeilen -> {len(alldf)} gelabelte Figuren "
          f"aus {alldf['source_json'].nunique()} Papers (Duplikate/Leere entfernt)")

    # Split nach Paper (Gruppen), Ziel ~20 % der Figuren ins Eval
    groups = alldf["source_json"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    tr_idx, ev_idx = next(gss.split(alldf, groups=groups))
    train = alldf.iloc[tr_idx].reset_index(drop=True)
    evd = alldf.iloc[ev_idx].reset_index(drop=True)

    overlap = set(train["source_json"]) & set(evd["source_json"])
    print(f"\nGeteilte Papers zwischen Train und Eval: {len(overlap)}  (muss 0 sein)")

    print("\nVerteilung:")
    show_dist(train, "Train")
    show_dist(evd, "Eval ")

    out_dir = os.path.dirname(os.path.abspath(tcsv)) or "."
    tp = os.path.join(out_dir, "train_split.csv")
    ep = os.path.join(out_dir, "eval_split.csv")
    train.to_csv(tp, sep=";", index=False, encoding="utf-8-sig")
    evd.to_csv(ep, sep=";", index=False, encoding="utf-8-sig")
    print(f"\nGeschrieben:\n  {tp}\n  {ep}")
    print("\nNaechster Schritt: python evaluate_on_goldset.py")
    print("  -> als Training 'train_split.csv', als Eval 'eval_split.csv' waehlen.")
    print("Das ist dann deine saubere, paper-getrennte Hauptevaluation.")


if __name__ == "__main__":
    main()
