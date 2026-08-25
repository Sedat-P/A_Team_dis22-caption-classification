# -*- coding: utf-8 -*-
"""
Captions reparieren (Version 2).

Manche JSONs speichern in 'figures[].caption' nur die erste Zeile. Der VOLLE
Caption-Text steckt aber im 'text'-Feld der JSON (im 'Figure legends'-Abschnitt).
Dieses Skript holt die vollstaendige Caption von dort zurueck und ersetzt die
gekuerzte in deiner CSV. Alle anderen Spalten - vor allem deine Labels - bleiben
unveraendert.

Es wird eine NEUE Datei geschrieben:  <deine_csv>_fullcaptions.csv

Benoetigt:  pip install pandas
Aufruf:     python repair_captions_v2.py [deine_annotation.csv] [json_ordner]
            (ohne Argumente: Datei-/Ordner-Dialoge)
"""

import os
import re
import sys
import csv
import json
import difflib
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CAPTION_COL = "caption"
JSON_COL = "source_json"

ANY_FIG = re.compile(r"Fig(?:ure)?\.?\s*S?\s*\d+\s*\.", re.I)


def clean(s):
    return re.sub(r"\s+", " ", (s or "")).strip()


def fig_token(s):
    """'Fig. 3' -> '3', 'Figure S5' -> 's5'."""
    if not s:
        return None
    m = re.search(r"\bfig(?:ure)?s?\.?\s*(s?\s*\d+)", s, flags=re.I)
    return re.sub(r"\s+", "", m.group(1)).lower() if m else None


# ---------------- Figuren-Matching (fuer das figures[].caption-Feld) ----------
def find_figure(csv_caption, figures):
    if not figures:
        return None
    csv_norm = clean(csv_caption).lower()
    csv_key = fig_token(csv_caption)
    best, best_score = None, -1.0
    for fg in figures:
        jcap = clean(fg.get("caption", "")).lower()
        j_key = fig_token(fg.get("name", "")) or fig_token(jcap)
        score = 0.0
        if csv_key and j_key and csv_key == j_key:
            score += 0.6
        if csv_norm and jcap:
            if jcap.startswith(csv_norm) or csv_norm.startswith(jcap):
                score += 0.4
            score += 0.4 * difflib.SequenceMatcher(None, csv_norm, jcap).ratio()
        if score > best_score:
            best, best_score = fg, score
    return best


# ---------------- Volle Caption aus dem 'text'-Feld holen ----------------
def recover_from_text(txt, fignum_token):
    """Sucht die vollstaendige Caption im Legenden-Abschnitt des Volltextes."""
    if not txt or not fignum_token:
        return ""
    num = fignum_token  # z.B. '3' oder 's5'
    # Marker fuer genau diese Figur
    pat = re.compile(rf"Fig(?:ure)?\.?\s*{re.escape(num[:1])}?\s*{re.escape(num.lstrip('s'))}\s*\.",
                     re.I) if num.startswith("s") else \
        re.compile(rf"Fig(?:ure)?\.?\s*{re.escape(num)}\s*\.", re.I)

    starts = [m.start() for m in pat.finditer(txt)]
    if not starts:
        return ""
    # Legenden stehen hinten -> letztes Vorkommen nehmen
    start = starts[-1]
    # Segment bis zum naechsten Fig-Marker (oder +2500 Zeichen)
    nxt = ANY_FIG.search(txt, start + 5)
    end = nxt.start() if nxt else min(len(txt), start + 2500)
    seg = txt[start:end]

    # Zeilennummern entfernen, zusammenfuegen
    lines = [ln.strip() for ln in seg.split("\n")]
    keep = [ln for ln in lines if ln and not re.fullmatch(r"\d+", ln)]
    text = clean(" ".join(keep))

    # am letzten Panel-Label "(X)." abschneiden (entfernt Figur-Label-Reste)
    panels = list(re.finditer(r"\([A-Za-z]\)\s*\.", text))
    if panels and panels[-1].end() > len(text) * 0.4:
        text = text[:panels[-1].end()]
    # Sicherheitskappe
    if len(text) > 2000:
        text = text[:2000].rsplit(". ", 1)[0] + "."
    return text if text.lower().startswith("fig") else ""


# ---------------- JSON-Cache ----------------
_cache = {}


def load_json(json_dir, name):
    if name not in _cache:
        try:
            with open(os.path.join(json_dir, name), "r", encoding="utf-8") as f:
                d = json.load(f)
            _cache[name] = (d.get("figures", []) or [], d.get("text", "") or "")
        except Exception:
            _cache[name] = ([], "")
    return _cache[name]


def best_caption(csv_cap, figures, txt):
    """Beste (laengste, gueltige) Caption aus allen Quellen."""
    candidates = [clean(csv_cap)]
    fg = find_figure(csv_cap, figures)
    if fg:
        candidates.append(clean(fg.get("caption", "")))
    rec = recover_from_text(txt, fig_token(csv_cap) or (fig_token(fg.get("name", "")) if fg else None))
    if rec:
        candidates.append(rec)
    candidates = [c for c in candidates if c]
    if not candidates:
        return clean(csv_cap)
    return max(candidates, key=len)


# ---------------- Hauptablauf ----------------
def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]) else None
    json_dir = sys.argv[2] if len(sys.argv) > 2 and os.path.isdir(sys.argv[2]) else None
    if not csv_path or not json_dir:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        if not csv_path:
            csv_path = filedialog.askopenfilename(
                title="Annotations-CSV waehlen",
                filetypes=[("CSV", "*.csv"), ("Alle", "*.*")]) or None
        if csv_path and not json_dir:
            json_dir = filedialog.askdirectory(title="JSON-Ordner waehlen") or None
        root.destroy()
    if not csv_path or not json_dir:
        print("CSV oder JSON-Ordner fehlt. Abbruch."); return

    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    if CAPTION_COL not in df.columns or JSON_COL not in df.columns:
        raise SystemExit(f"Spalten '{CAPTION_COL}'/'{JSON_COL}' fehlen.")

    n_longer = n_same = 0
    examples = []
    new_caps = []
    for _, row in df.iterrows():
        old = row[CAPTION_COL]
        figs, txt = load_json(json_dir, row[JSON_COL])
        new = best_caption(old, figs, txt)
        new_caps.append(new)
        if len(new) > len(clean(old)) + 5:
            n_longer += 1
            if len(examples) < 4:
                examples.append((clean(old)[:70], new[:150]))
        else:
            n_same += 1
    df[CAPTION_COL] = new_caps

    out_path = os.path.splitext(csv_path)[0] + "_fullcaptions.csv"
    df.to_csv(out_path, sep=";", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

    print(f"Zeilen gesamt       : {len(df)}")
    print(f"Caption verlaengert : {n_longer}")
    print(f"Unveraendert        : {n_same}")
    print(f"\nGeschrieben: {out_path}")
    if examples:
        print("\nBeispiele (vorher -> nachher):")
        for old, new in examples:
            print(f"  - '{old}...'\n    -> '{new}...'")
    print("\nDeine Labels sind erhalten. Mit dieser Datei kannst du weiterannotieren/trainieren.")


if __name__ == "__main__":
    main()
