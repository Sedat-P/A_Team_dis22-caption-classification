# -*- coding: utf-8 -*-
"""
Annotations-Sets vorbereiten (stratifizierte Auswahl).

Ziel:
  - Eine VOR-Klassifikation der Abbildungen deines neuen Samples (PDFs+JSON),
    damit du fuer das Training gezielt mehr Diagramme/Fotos bekommst statt fast
    nur Plots.
  - Daraus zwei fertige Annotations-CSVs fuer annotate.py:
      (1) annotation_task_train_1000.csv  -> AUSGEWOGEN (mehr seltene Klassen)
      (2) annotation_task_eval_200.csv    -> ZUFAELLIG (repraesentativ, fuer Eval)
    Beide ueberschneiden sich nicht.

WICHTIG:
  - Die Vor-Klassifikation ist nur ein FILTER zur Auswahl. Die echten Labels
    machst du weiterhin selbst in annotate.py. Die Spalte 'prescreen_pred' ist
    nur zur Info und wird in annotate.py NICHT angezeigt (kein Bias).
  - Das Eval-Set ist bewusst ZUFAELLIG, damit es die echte Verteilung abbildet.

Benoetigt:  pip install scikit-learn pandas numpy
Aufruf:     python prepare_annotation_sets.py [alte_annotation.csv] [neuer_json_ordner] [out_ordner]
            (ohne Argumente: Datei-/Ordner-Dialoge)
"""

import os
import re
import sys
import json
import glob
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline


# ======================================================================
#  Konfiguration
# ======================================================================
N_TRAIN = 1000          # Groesse des (ausgewogenen) Trainings-Sets
N_EVAL  = 200           # Groesse des (zufaelligen) Eval-Sets
RANDOM_STATE = 42

TEXT_COL  = "caption"
CLASSES   = ["plot", "diagram", "photo"]
FLAG_COLS = {"plot": "manual_plot", "diagram": "manual_diagram", "photo": "manual_photo"}

# Spalten der Annotations-CSV (Format wie annotate.py es erwartet)
OUT_COLS = ["annotation_id", "source_json", "caption", "Maschine_Label",
            "manual_plot", "manual_photo", "manual_diagram", "unclear", "notes",
            "prescreen_pred"]


def flag(v):
    return 1 if str(v).strip() in ("1", "1.0", "x", "X", "true", "True") else 0


def norm_caption(s):
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


# ======================================================================
#  Vor-Klassifikator auf den alten Annotationen trainieren
# ======================================================================
def train_prescreen(csv_path):
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig", dtype=str).fillna("")
    for cls, col in FLAG_COLS.items():
        if col not in df.columns:
            raise SystemExit(f"Spalte '{col}' fehlt in {csv_path}")
        df[f"y_{cls}"] = df[col].map(flag)
    usable = (df[[f"y_{c}" for c in CLASSES]].sum(axis=1) > 0) & \
             (df[TEXT_COL].str.strip() != "")
    df = df[usable]
    if len(df) < 30:
        raise SystemExit(f"Zu wenige annotierte Zeilen ({len(df)}) zum Vor-Trainieren.")
    X = df[TEXT_COL].astype(str).values
    Y = df[[f"y_{c}" for c in CLASSES]].values.astype(int)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                  sublinear_tf=True, stop_words="english")),
        ("clf", OneVsRestClassifier(
            LogisticRegression(max_iter=2000, class_weight="balanced"))),
    ])
    pipe.fit(X, Y)
    known_caps = set(norm_caption(c) for c in df[TEXT_COL].tolist())
    print(f"Vor-Klassifikator trainiert auf {len(df)} alten Annotationen.")
    return pipe, known_caps


# ======================================================================
#  Figures aus dem neuen JSON-Ordner sammeln
# ======================================================================
def collect_figures(json_dir):
    rows = []
    files = glob.glob(os.path.join(json_dir, "*.json"))
    print(f"{len(files)} JSON-Dateien gefunden.")
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        meta = data.get("metadata", {}) or {}
        cat = (meta.get("preprint_category") or data.get("preprint_category") or "")
        doi = data.get("doi") or meta.get("doi") or ""
        for fig in (data.get("figures", []) or []):
            cap = fig.get("caption", "") or ""
            if not cap.strip():
                continue
            rows.append({
                "source_json": os.path.basename(fp),
                "caption": cap,
                "category": str(cat).strip().lower(),
                "doi": doi,
            })
    print(f"{len(rows)} Abbildungen mit Caption gesammelt.")
    return pd.DataFrame(rows)


# ======================================================================
#  Vorhersage + Hilfsfunktionen
# ======================================================================
def combo_label(rowvals):
    parts = [c.capitalize() for c, v in zip(CLASSES, rowvals) if v == 1]
    return "+".join(parts) if parts else "None"


def write_csv(df_rows, path):
    out = df_rows.copy()
    out.insert(0, "annotation_id", range(1, len(out) + 1))
    for c in ["Maschine_Label", "manual_plot", "manual_photo", "manual_diagram",
              "unclear", "notes"]:
        out[c] = ""
    out = out[OUT_COLS]
    out.to_csv(path, sep=";", index=False, encoding="utf-8-sig")


# ======================================================================
#  Hauptablauf
# ======================================================================
def main():
    # --- Eingaben: Argumente oder Dialoge ---
    csv_path = sys.argv[1] if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]) else None
    json_dir = sys.argv[2] if len(sys.argv) > 2 and os.path.isdir(sys.argv[2]) else None
    if not csv_path or not json_dir:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        if not csv_path:
            csv_path = filedialog.askopenfilename(
                title="Alte annotierte CSV waehlen (zum Vor-Trainieren)",
                filetypes=[("CSV", "*.csv"), ("Alle", "*.*")]) or None
        if csv_path and not json_dir:
            json_dir = filedialog.askdirectory(
                title="JSON-Ordner des neuen Samples waehlen") or None
        root.destroy()
    if not csv_path:
        print("Keine CSV gewaehlt. Abbruch."); return
    if not json_dir:
        print("Kein JSON-Ordner gewaehlt. Abbruch."); return

    out_dir = sys.argv[3] if len(sys.argv) > 3 else \
        os.path.dirname(os.path.abspath(csv_path)) or "."
    os.makedirs(out_dir, exist_ok=True)

    # --- 1. Vor-Klassifikator trainieren ---
    pipe, known_caps = train_prescreen(csv_path)

    # --- 2. Figures sammeln ---
    pool = collect_figures(json_dir)
    if pool.empty:
        raise SystemExit("Keine Abbildungen gefunden - stimmt der JSON-Ordner?")

    # --- 3. Bereits annotierte (gleiche Caption) ausschliessen ---
    before = len(pool)
    pool["_norm"] = pool["caption"].map(norm_caption)
    pool = pool[~pool["_norm"].isin(known_caps)].drop(columns="_norm").reset_index(drop=True)
    if before != len(pool):
        print(f"{before - len(pool)} bereits annotierte Abbildungen ausgeschlossen.")
    print(f"{len(pool)} Abbildungen im Auswahl-Pool.\n")

    # --- 4. Vor-Klassifikation ---
    preds = pipe.predict(pool["caption"].astype(str).values)
    for j, c in enumerate(CLASSES):
        pool[f"p_{c}"] = preds[:, j]
    pool["prescreen_pred"] = [combo_label(r) for r in preds]

    print("Vor-Klassifikation des Pools (Anteil mit Klasse):")
    for j, c in enumerate(CLASSES):
        print(f"  {c:8}: {int(preds[:, j].sum())} / {len(pool)}")

    rng = np.random.RandomState(RANDOM_STATE)

    # --- 5. Eval-Set: ZUFAELLIG (repraesentativ) ---
    n_eval = min(N_EVAL, len(pool))
    eval_idx = rng.choice(pool.index.values, size=n_eval, replace=False)
    eval_df = pool.loc[eval_idx].copy()
    remainder = pool.drop(index=eval_idx)

    # --- 6. Trainings-Set: AUSGEWOGEN (seltene Klassen hochgewichten) ---
    is_rare = (remainder["p_diagram"] == 1) | (remainder["p_photo"] == 1)
    rare = remainder[is_rare].sample(frac=1, random_state=RANDOM_STATE)
    rest = remainder[~is_rare].sample(frac=1, random_state=RANDOM_STATE)

    take_rare = rare.iloc[:N_TRAIN]
    need = max(0, N_TRAIN - len(take_rare))
    take_rest = rest.iloc[:need]
    train_df = pd.concat([take_rare, take_rest]).sample(frac=1, random_state=RANDOM_STATE)

    # --- 7. CSVs schreiben ---
    train_path = os.path.join(out_dir, "annotation_task_train_1000.csv")
    eval_path  = os.path.join(out_dir, "annotation_task_eval_200.csv")
    write_csv(train_df, train_path)
    write_csv(eval_df, eval_path)

    # --- Zusammenfassung ---
    def dist(df, name):
        print(f"\n{name} ({len(df)} Abbildungen) - Vor-Klassifikation:")
        for c in CLASSES:
            print(f"  {c:8}: {int(df[f'p_{c}'].sum())}")
    dist(train_df, "TRAINING (ausgewogen)")
    dist(eval_df, "EVAL (zufaellig)")

    print(f"\nGeschrieben:")
    print(f"  {train_path}")
    print(f"  {eval_path}")
    print("\nNaechster Schritt: beide CSVs nacheinander in annotate.py oeffnen und labeln.")
    print("Die Spalte 'prescreen_pred' ist nur Info und wird beim Annotieren nicht angezeigt.")


if __name__ == "__main__":
    main()
