# -*- coding: utf-8 -*-
"""
regex_lee.py  --  Regex-Baseline ("Regex B"), abgeleitet aus den
Kategorie-Definitionen von Lee et al. (2016, "Viziometrics").

IDEE
----
Lee et al. definieren ihre Bildtypen mit Beispieltermen, z. B.:
  * Plot     (e.g., bar charts, scatter plots, line charts)
  * Diagram  (e.g., schematics, conceptual diagrams, flow charts,
              architecture diagrams, illustrations)
  * Photo    (e.g., microscopy images, diagnostic images,
              radiology images, fluorescence imaging)

Wir nehmen genau diese Beispielterme als SEEDS (Tier 1) und ergänzen sie
um offensichtliche weitere Mitglieder DERSELBEN Lee-Kategorie (Tier 2,
z. B. ist ein Histogramm klar ein "Plot" im Sinne von Lee). Aus jedem
Term baut der Code automatisch ein robustes Suchmuster (Regex).

Diese Datei ist die EINZIGE Quelle der Regex-Logik. train_and_evaluate.py
und run_full_corpus.py importieren regex_predict von hier - so muss die
Wortliste nur an einer Stelle gepflegt werden.

Direkt ausführbar, um "Regex B" auf einer annotierten CSV zu bewerten:
    python regex_lee.py  [annotierte.csv]
"""

import re
import sys
import numpy as np

CLASSES = ["plot", "diagram", "photo"]          # Reihenfolge der Labels

# ----------------------------------------------------------------------
# Tier 1 -- WÖRTLICH die Beispielterme aus Lee et al. (2016)
# ----------------------------------------------------------------------
LEE_SEEDS = {
    "plot":    ["bar chart", "scatter plot", "line chart"],
    "diagram": ["schematic", "conceptual diagram", "flow chart",
                "architecture diagram", "illustration"],
    # Bei Photo sind die unterscheidenden Wörter die Bildgebungs-Verfahren;
    # "images/imaging" selbst ist zu generisch und wird weggelassen.
    "photo":   ["microscop*", "diagnostic*", "radiolog*", "fluoresc*"],
}

# ----------------------------------------------------------------------
# Quelle 2 -- DATENGETRIEBEN: trennschärfste Begriffe je Klasse,
#   abgeleitet aus train_split (TF-IDF + Logistic-Regression-Gewichte).
#   Reproduzierbar mit derive_keywords.py. Bewusst NUR aus den
#   Trainingsdaten gezogen -> das eval_split bleibt unberührt.
# ----------------------------------------------------------------------
DATA_DRIVEN = {
    "plot":    ["mean", "values", "error", "performance", "error bars",
                "axis", "line", "curves", "measured", "time"],
    "diagram": ["workflow", "structure", "diagram", "network", "binding",
                "representation", "phylogenetic", "model", "complex"],
    "photo":   ["images", "cells", "cell", "brain", "image", "cortical",
                "marker", "expression", "nano", "temporal"],
}


# ----------------------------------------------------------------------
# Aus einem Term ein Suchmuster bauen
# ----------------------------------------------------------------------
def build_pattern(term):
    """Wandelt einen Such-Term in einen Regex um.

    - Endet der Term auf '*', wird er als WORTSTAMM behandelt:
        'microscop*'  ->  \\bmicroscop\\w*
      (trifft microscopy, microscope, microscopic, ...)
    - Sonst exakte (Mehr-)Wortphrase mit Toleranz:
        'bar chart'   ->  \\bbar[\\s-]?chart(?:s|es)?\\b
      (trifft 'bar chart', 'bar charts', 'bar-chart', 'barchart')
    """
    term = term.strip().lower()
    if term.endswith("*"):
        stem = re.escape(term[:-1])
        return r"\b" + stem + r"\w*"
    words = term.split()
    joined = r"[\s\-]?".join(re.escape(w) for w in words)
    return r"\b" + joined + r"(?:s|es)?\b"


def _build_compiled():
    """Quelle 1 (Lee) + Quelle 2 (datengetrieben) zusammenführen, dedupizieren
    und zu kompilierten Regex-Mustern machen."""
    keywords = {}
    for c in CLASSES:
        seen, terms = set(), []
        for t in LEE_SEEDS.get(c, []) + DATA_DRIVEN.get(c, []):
            key = t.strip().lower()
            if key not in seen:
                seen.add(key); terms.append(t)
        keywords[c] = [build_pattern(t) for t in terms]
    compiled = {c: [re.compile(p, re.I) for p in pats]
                for c, pats in keywords.items()}
    return keywords, compiled


# REGEX_KEYWORDS = lesbare Musterliste, _RX = kompiliert (für die Suche)
REGEX_KEYWORDS, _RX = _build_compiled()


def regex_predict(texts):
    """Multi-Label-Vorhersage: pro Caption für jede Klasse 1, wenn
    mindestens ein Muster dieser Klasse im Text vorkommt."""
    preds = np.zeros((len(texts), len(CLASSES)), dtype=int)
    for i, t in enumerate(texts):
        t = t or ""
        for j, c in enumerate(CLASSES):
            if any(rx.search(t) for rx in _RX[c]):
                preds[i, j] = 1
    return preds


# ----------------------------------------------------------------------
# Optional: direkt bewerten (Regex B auf einer annotierten CSV)
# ----------------------------------------------------------------------
def _main():
    import pandas as pd
    from sklearn.metrics import precision_recall_fscore_support, f1_score

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()
            path = filedialog.askopenfilename(
                title="Annotierte CSV für Regex-B-Bewertung wählen",
                filetypes=[("CSV", "*.csv"), ("Alle Dateien", "*.*")]) or None
            root.destroy()
        except Exception:
            pass
    if not path:
        print("Keine CSV gewählt. Aufruf: python regex_lee.py <annotierte.csv>")
        return

    flagcol = {"plot": "manual_plot", "diagram": "manual_diagram", "photo": "manual_photo"}
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str).fillna("")

    def flag(v):
        return 1 if str(v).strip() in ("1", "1.0", "x", "X", "true", "True") else 0

    for c in CLASSES:
        df[f"y_{c}"] = df[flagcol[c]].map(flag)
    use = (df[[f"y_{c}" for c in CLASSES]].sum(axis=1) > 0) & (df["caption"].str.strip() != "")
    df = df[use]
    X = df["caption"].astype(str).values
    Y = df[[f"y_{c}" for c in CLASSES]].values.astype(int)
    P = regex_predict(X)

    print(f"\nRegex B (Lee-basiert) auf {len(df)} annotierten Captions\n" + "=" * 58)
    pr, rc, f1, _ = precision_recall_fscore_support(Y, P, average=None,
                                                    zero_division=0,
                                                    labels=range(len(CLASSES)))
    print(f"  {'Klasse':10s}{'Precision':>11s}{'Recall':>9s}{'F1':>8s}")
    for j, c in enumerate(CLASSES):
        print(f"  {c:10s}{pr[j]:11.3f}{rc[j]:9.3f}{f1[j]:8.3f}")
    print("-" * 58)
    print(f"  Macro-F1: {f1_score(Y, P, average='macro', zero_division=0):.3f}"
          f"   Micro-F1: {f1_score(Y, P, average='micro', zero_division=0):.3f}")
    print("\nGenutzte Muster pro Klasse:")
    for c in CLASSES:
        print(f"  {c}: {len(REGEX_KEYWORDS[c])} Muster")


if __name__ == "__main__":
    _main()
