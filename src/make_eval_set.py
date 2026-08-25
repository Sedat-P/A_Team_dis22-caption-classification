# -*- coding: utf-8 -*-
"""
Unabhaengiges Evaluations-Set erstellen.

Zieht eine ZUFAELLIGE Stichprobe von Abbildungen aus deiner Sample-Data, die NICHT
in deiner Annotations-CSV vorkommen. Das Ergebnis ist eine CSV im annotate.py-Format,
die du selbst labelst (Gold-Standard) und spaeter gegen das Modell vergleichst.

Erzeugt:  annotation_task_eval.csv

Benoetigt:  pip install pandas
Aufruf:     python make_eval_set.py [annotation.csv] [json_ordner] [out_ordner]
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


# ======================================================================
#  Konfiguration
# ======================================================================
N_EVAL = 300            # Groesse des Eval-Sets (z. B. auf 200 aendern)
RANDOM_STATE = 42
MIN_CAPTION_WORDS = 6   # Captions mit weniger Woertern (nur Label wie "Fig. 4") rausfiltern

JSON_COL = "source_json"
CAPTION_COL = "caption"

OUT_COLS = ["annotation_id", "source_json", "caption", "Maschine_Label",
            "manual_plot", "manual_photo", "manual_diagram", "unclear", "notes"]


def norm(s):
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def fig_key(s):
    """'Fig. 3' -> '3', 'Figure S5' -> 's5' (eindeutig je Preprint)."""
    if not s:
        return None
    m = re.search(r"\bfig(?:ure)?s?\.?\s*(s?\s*\d+)", s, flags=re.I)
    return re.sub(r"\s+", "", m.group(1)).lower() if m else None


# ======================================================================
#  Bereits annotierte Abbildungen aus der CSV bestimmen
# ======================================================================
def build_excluded(csv_path):
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    if JSON_COL not in df.columns or CAPTION_COL not in df.columns:
        raise SystemExit(f"Spalten '{JSON_COL}'/'{CAPTION_COL}' fehlen in der CSV.")
    by_num = set()   # (source_json, figurennummer)  -> robust gegen Caption-Aenderungen
    by_cap = set()   # (source_json, caption-prefix)  -> Fallback
    for _, r in df.iterrows():
        sj = r[JSON_COL]
        cap = r[CAPTION_COL]
        k = fig_key(cap)
        if k:
            by_num.add((sj, k))
        by_cap.add((sj, norm(cap)[:60]))
    print(f"Annotierte CSV: {len(df)} Zeilen  ->  ausgeschlossen werden diese Abbildungen.")
    return by_num, by_cap


# ======================================================================
#  Pool aus den JSONs sammeln (ohne die ausgeschlossenen)
# ======================================================================
def collect_pool(json_dir, by_num, by_cap):
    rows = []
    files = glob.glob(os.path.join(json_dir, "*.json"))
    print(f"{len(files)} JSON-Dateien gefunden.")
    n_excl = 0
    n_short = 0
    for fp in files:
        name = os.path.basename(fp)
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        meta = data.get("metadata", {}) or {}
        cat = (meta.get("preprint_category") or data.get("preprint_category") or "")
        for fig in (data.get("figures", []) or []):
            cap = (fig.get("caption", "") or "").strip()
            if not cap:
                continue
            # nur Label ohne echten Text (z. B. "Fig. 4") ausschliessen
            if len(cap.split()) < MIN_CAPTION_WORDS:
                n_short += 1
                continue
            k = fig_key(fig.get("name", "")) or fig_key(cap)
            if (k and (name, k) in by_num) or ((name, norm(cap)[:60]) in by_cap):
                n_excl += 1
                continue
            rows.append({"source_json": name, "caption": cap,
                         "category": str(cat).strip().lower()})
    print(f"{n_excl} Abbildungen ausgeschlossen (waren schon annotiert).")
    print(f"{n_short} Abbildungen ausgeschlossen (Caption ohne echten Text, nur Label).")
    print(f"{len(rows)} Abbildungen im verfuegbaren Pool.")
    return pd.DataFrame(rows)


def write_csv(df_rows, path):
    out = df_rows.copy().reset_index(drop=True)
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
    csv_path = sys.argv[1] if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]) else None
    json_dir = sys.argv[2] if len(sys.argv) > 2 and os.path.isdir(sys.argv[2]) else None
    if not csv_path or not json_dir:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        if not csv_path:
            csv_path = filedialog.askopenfilename(
                title="Annotations-CSV waehlen (die bereits gelabelten)",
                filetypes=[("CSV", "*.csv"), ("Alle", "*.*")]) or None
        if csv_path and not json_dir:
            json_dir = filedialog.askdirectory(title="JSON-Ordner der Sample-Data waehlen") or None
        root.destroy()
    if not csv_path or not json_dir:
        print("CSV oder JSON-Ordner fehlt. Abbruch."); return

    out_dir = sys.argv[3] if len(sys.argv) > 3 else (os.path.dirname(os.path.abspath(csv_path)) or ".")
    os.makedirs(out_dir, exist_ok=True)

    by_num, by_cap = build_excluded(csv_path)
    pool = collect_pool(json_dir, by_num, by_cap)
    if pool.empty:
        raise SystemExit("Pool ist leer - stimmt der JSON-Ordner?")

    n = min(N_EVAL, len(pool))
    if n < N_EVAL:
        print(f"Hinweis: nur {n} Abbildungen verfuegbar (weniger als {N_EVAL}).")
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(pool.index.values, size=n, replace=False)
    eval_df = pool.loc[idx]

    out_path = os.path.join(out_dir, "annotation_task_eval.csv")
    write_csv(eval_df, out_path)

    print(f"\nEval-Set mit {n} Abbildungen geschrieben: {out_path}")
    print("Verteilung nach Fachrichtung (nur Info):")
    print(eval_df["category"].value_counts().head(8).to_string())
    print("\nNaechster Schritt: diese CSV in annotate.py oeffnen und SELBST labeln (Gold-Standard).")


if __name__ == "__main__":
    main()
