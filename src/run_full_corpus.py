# -*- coding: utf-8 -*-
"""
Schritt 3 des A-Team-Plans: Voll-Korpus-Lauf.

Trainiert den Classifier EINMAL auf allen manuell annotierten Daten und wendet
ihn (sowie Regex A) auf ALLE Figures in ALLEN JSON-Dateien an.
Erzeugt:
  - <out>/figures_classified.csv  : jede Figure mit Vorhersagen + Metadaten
  - <out>/summary_overall.csv      : Verteilung der Typen gesamt
  - <out>/summary_by_category.csv  : Verteilung pro Fachrichtung/Kategorie
  - <out>/dist_overall.png         : Balkendiagramm gesamt
  - <out>/dist_by_category.png     : gruppiertes Balkendiagramm pro Kategorie

Benötigt:  pip install scikit-learn pandas matplotlib
Aufruf:    python run_full_corpus.py  <annotation.csv>  <json_ordner>  [out_ordner]
"""

import os
import re
import sys
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Windows-Konsole UTF-8-sicher machen (verhindert Crash bei Umlauten/Sonderzeichen)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline


# ======================================================================
#  Konfiguration  (identisch zu train_and_evaluate.py halten!)
# ======================================================================
TEXT_COL  = "caption"
CLASSES   = ["plot", "diagram", "photo"]
FLAG_COLS = {"plot": "manual_plot", "diagram": "manual_diagram", "photo": "manual_photo"}

PRIMARY_METHOD = "classifier"   # "classifier" oder "regex" -> Basis für die Diagramme

MIN_CAPTION_WORDS = 6   # Figures mit reiner Label-Caption ("Fig. 4") ausschliessen -
                        # ohne Text kann der Classifier nichts Sinnvolles vorhersagen

# Regex-Baseline aus zentraler Quelle (regex_lee.py, gleicher Ordner)
from regex_lee import regex_predict


# ======================================================================
#  Classifier auf allen Annotationen trainieren
# ======================================================================
def flag(v):
    return 1 if str(v).strip() in ("1", "1.0", "x", "X", "true", "True") else 0


def fit_classifier(csv_path):
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig", dtype=str).fillna("")
    for cls, col in FLAG_COLS.items():
        if col not in df.columns:
            raise SystemExit(f"Spalte '{col}' fehlt in {csv_path}")
        df[f"y_{cls}"] = df[col].map(flag)
    usable = (df[[f"y_{c}" for c in CLASSES]].sum(axis=1) > 0) & \
             (df[TEXT_COL].str.strip() != "")
    df = df[usable]
    if len(df) < 30:
        raise SystemExit(f"Zu wenige annotierte Zeilen ({len(df)}). Erst evaluieren.")
    X = df[TEXT_COL].astype(str).values
    Y = df[[f"y_{c}" for c in CLASSES]].values.astype(int)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                  sublinear_tf=True, stop_words="english")),
        ("clf", OneVsRestClassifier(
            LogisticRegression(max_iter=2000, class_weight="balanced"))),
    ])
    pipe.fit(X, Y)
    print(f"Classifier trainiert auf {len(df)} annotierten Abbildungen.")
    return pipe


# ======================================================================
#  Alle JSONs einlesen -> jede Figure als Datensatz
# ======================================================================
def collect_figures(json_dir):
    records = []
    files = glob.glob(os.path.join(json_dir, "*.json"))
    print(f"{len(files)} JSON-Dateien gefunden in {json_dir}")
    n_err = 0
    n_short = 0
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            n_err += 1
            continue
        meta = data.get("metadata", {}) or {}
        category = (meta.get("preprint_category")
                    or data.get("preprint_category") or "unknown")
        doi = data.get("doi") or meta.get("doi") or os.path.basename(fp)
        for fig in (data.get("figures", []) or []):
            cap = (fig.get("caption", "") or "").strip()
            # reine Label-Captions ohne echten Text ausschliessen
            if len(cap.split()) < MIN_CAPTION_WORDS:
                n_short += 1
                continue
            records.append({
                "json_file": os.path.basename(fp),
                "doi": doi,
                "category": str(category).strip().lower(),
                "figure_name": fig.get("name", ""),
                "page": fig.get("pos_page", ""),
                "caption": cap,
            })
    if n_err:
        print(f"  ({n_err} Dateien konnten nicht gelesen werden – übersprungen)")
    print(f"{n_short} Figures ohne echten Caption-Text (nur Label) ausgeschlossen.")
    print(f"{len(records)} Figures mit Caption werden klassifiziert.")
    return pd.DataFrame(records)


# ======================================================================
#  Vorhersagen + Aggregation
# ======================================================================
def combo_label(row, prefix):
    parts = [c.capitalize() for c in CLASSES if row[f"{prefix}_{c}"] == 1]
    return "+".join(parts) if parts else "None"


def main():
    # CSV: aus Kommandozeile oder per Dialog
    csv_path = sys.argv[1] if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]) else None
    json_dir = sys.argv[2] if len(sys.argv) > 2 and os.path.isdir(sys.argv[2]) else None

    if not csv_path or not json_dir:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        if not csv_path:
            csv_path = filedialog.askopenfilename(
                title="Annotierte CSV wählen",
                filetypes=[("CSV", "*.csv"), ("Alle Dateien", "*.*")]) or None
        if csv_path and not json_dir:
            json_dir = filedialog.askdirectory(
                title="Ordner mit ALLEN JSON-Dateien wählen") or None
        root.destroy()

    if not csv_path:
        print("Keine CSV gewählt. Abbruch."); return
    if not json_dir:
        print("Kein JSON-Ordner gewählt. Abbruch."); return

    # Ausgabeordner: Argument, sonst neben der CSV
    if len(sys.argv) > 3:
        out_dir = sys.argv[3]
    else:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(csv_path)),
                               "corpus_results")
    os.makedirs(out_dir, exist_ok=True)
    print(f"CSV       : {csv_path}")
    print(f"JSON-Ordner: {json_dir}")
    print(f"Ausgabe   : {out_dir}\n")

    pipe = fit_classifier(csv_path)
    df = collect_figures(json_dir)
    if df.empty:
        raise SystemExit("Keine Figures gefunden – stimmt der JSON-Ordner?")

    texts = df["caption"].astype(str).values

    # Vorhersagen (beide Methoden, damit ihr auch auf dem Korpus vergleichen könnt)
    clf_pred = pipe.predict(texts)
    rgx_pred = regex_predict(texts)
    for j, c in enumerate(CLASSES):
        df[f"clf_{c}"] = clf_pred[:, j]
        df[f"regex_{c}"] = rgx_pred[:, j]
    df["clf_label"]   = df.apply(lambda r: combo_label(r, "clf"), axis=1)
    df["regex_label"] = df.apply(lambda r: combo_label(r, "regex"), axis=1)

    fig_csv = os.path.join(out_dir, "figures_classified.csv")
    df.to_csv(fig_csv, sep=";", index=False, encoding="utf-8-sig")
    print(f"\nGespeichert: {fig_csv}")

    prefix = "clf" if PRIMARY_METHOD == "classifier" else "regex"

    # ---- Verteilung gesamt (pro Klasse, unabhängig = Multilabel) ----
    overall = pd.DataFrame({
        "class": CLASSES,
        "count": [int(df[f"{prefix}_{c}"].sum()) for c in CLASSES],
    })
    overall["share_%"] = (100 * overall["count"] / len(df)).round(1)
    overall.to_csv(os.path.join(out_dir, "summary_overall.csv"),
                   sep=";", index=False, encoding="utf-8-sig")
    print("\nVerteilung gesamt (Anteil der Figures mit dieser Klasse):")
    print(overall.to_string(index=False))
    print("\n  HINWEIS: Das ist die VORHERGESAGTE Verteilung, keine Evaluation -")
    print("  im Korpus gibt es keine Wahrheit zum Vergleichen. Die Anteile spiegeln")
    print("  auch die Fehlerraten des Classifiers wider (z. B. Photo eher ueberschaetzt,")
    print("  Plot verlaesslich). Bei der Darstellung Precision/Recall pro Klasse dazusagen.")

    # ---- Verteilung pro Kategorie ----
    rows = []
    for cat, g in df.groupby("category"):
        row = {"category": cat, "n_figures": len(g)}
        for c in CLASSES:
            row[f"{c}_%"] = round(100 * g[f"{prefix}_{c}"].sum() / len(g), 1)
        rows.append(row)
    by_cat = pd.DataFrame(rows).sort_values("n_figures", ascending=False)
    by_cat.to_csv(os.path.join(out_dir, "summary_by_category.csv"),
                  sep=";", index=False, encoding="utf-8-sig")
    print("\nVerteilung pro Fachrichtung (% der Figures je Klasse, rein deskriptiv):")
    print(by_cat.to_string(index=False))

    # ---- Multi-Label-Struktur im Korpus (deskriptiv) ----
    label_count = df[[f"{prefix}_{c}" for c in CLASSES]].sum(axis=1)
    print("\nMulti-Label-Struktur im Korpus:")
    lc = label_count.value_counts().sort_index()
    for k, v in lc.items():
        print(f"  {int(k)} Typ(en): {v} ({100*v/len(df):.1f}%)")
    combo_counts = df[f"{prefix}_label"].value_counts()
    print("  Haeufigste Kombinationen:")
    for name, v in combo_counts.head(8).items():
        print(f"    {name:22s}: {v} ({100*v/len(df):.1f}%)")
    combo_counts.rename("count").to_csv(
        os.path.join(out_dir, "summary_combinations.csv"),
        sep=";", encoding="utf-8-sig", header=True)

    # ---- Diagramme ----
    # 1) gesamt
    plt.figure(figsize=(6, 4))
    plt.bar(overall["class"], overall["count"])
    plt.ylabel("Anzahl Figures")
    plt.title(f"Verteilung der Abbildungstypen gesamt ({PRIMARY_METHOD})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dist_overall.png"), dpi=150)
    plt.close()

    # 2) pro Kategorie (gruppiert, nur die größten Kategorien)
    top = by_cat.head(6)
    x = np.arange(len(top))
    w = 0.25
    plt.figure(figsize=(9, 5))
    for k, c in enumerate(CLASSES):
        plt.bar(x + (k - 1) * w, top[f"{c}_%"], width=w, label=c)
    plt.xticks(x, top["category"], rotation=20, ha="right")
    plt.ylabel("% der Figures")
    plt.title(f"Abbildungstypen pro Fachrichtung ({PRIMARY_METHOD})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dist_by_category.png"), dpi=150)
    plt.close()

    print(f"\nDiagramme gespeichert in: {out_dir}/")
    print("Fertig. Detaildaten in figures_classified.csv (für Fehler-/Pattern-Analyse).")


if __name__ == "__main__":
    main()
