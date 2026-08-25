# -*- coding: utf-8 -*-
"""
Annotations-Sets vorbereiten -- VISUELL mit CLIP (Computer Vision).

NUR VORBEREITUNG: CLIP schaut sich die Figur-Bilder an und sortiert vor,
damit deine 1000 selbst zu annotierenden Abbildungen AUSGEWOGENER ueber die
Klassen verteilt sind (mehr Diagramme/Fotos statt fast nur Plots).
Die echten Labels vergibst DU spaeter in annotate.py. Die CLIP-Vorhersage
landet nur in der Info-Spalte 'prescreen_pred' und wird beim Annotieren
NICHT angezeigt (kein Bias).

Erzeugt:
  (1) annotation_task_train_1000.csv  -> AUSGEWOGEN (fuer dein Training)
  (2) annotation_task_eval_200.csv    -> ZUFAELLIG (repraesentativ, fuer die Evaluation)
Beide ueberschneiden sich nicht.

Benoetigt:  pip install torch open_clip_torch pymupdf pillow pandas numpy
Aufruf:     python prepare_annotation_sets_clip.py [json_ordner] [pdf_ordner] [out_ordner]
            (ohne Argumente: Ordner-Dialoge)
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
N_TRAIN = 1000
N_EVAL  = 200
RANDOM_STATE = 42

CLASSES = ["plot", "diagram", "photo"]

# CLIP-Modell (klein & schnell, laeuft auf CPU)
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

# Text-Prompts pro Klasse (werden gemittelt = robustes zero-shot)
PROMPTS = {
    "plot": [
        "a scientific plot", "a bar chart", "a line graph", "a scatter plot",
        "a data chart with x and y axes", "a statistical graph", "a histogram",
    ],
    "diagram": [
        "a schematic diagram", "a flowchart", "an illustration of a mechanism",
        "a conceptual diagram", "a workflow diagram", "a model overview figure",
    ],
    "photo": [
        "a microscopy image", "a fluorescence microscopy photograph",
        "a medical or biological photograph", "a photograph of cells or tissue",
        "a representative image", "a histology image",
    ],
}

# Bild-Rendering aus der PDF
PAGE_ONE_INDEXED = True
BBOX_PADDING = 10
RENDER_ZOOM = 2.0
BATCH_SIZE = 32

OUT_COLS = ["annotation_id", "source_json", "caption", "Maschine_Label",
            "manual_plot", "manual_photo", "manual_diagram", "unclear", "notes",
            "prescreen_pred"]


def norm_caption(s):
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


# ======================================================================
#  Figures aus den JSONs sammeln (inkl. Bounding-Box + PDF-Pfad)
# ======================================================================
def collect_figures(json_dir, pdf_dir):
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

        # PDF-Pfad bestimmen (wie in annotate.py)
        pdf_name = None
        for key in ("local_pdf_path", "pdf_path"):
            val = data.get(key) or meta.get(key)
            if val:
                pdf_name = os.path.basename(val)
                break
        if not pdf_name:
            pdf_name = os.path.splitext(os.path.basename(fp))[0] + ".pdf"
        pdf_path = os.path.join(pdf_dir, pdf_name)
        if not os.path.isfile(pdf_path):
            alt = os.path.join(pdf_dir, os.path.splitext(os.path.basename(fp))[0] + ".pdf")
            pdf_path = alt if os.path.isfile(alt) else None

        for fig in (data.get("figures", []) or []):
            cap = fig.get("caption", "") or ""
            if not cap.strip():
                continue
            rows.append({
                "source_json": os.path.basename(fp),
                "caption": cap,
                "category": str(cat).strip().lower(),
                "doi": doi,
                "pdf_path": pdf_path,
                "pos_page": fig.get("pos_page", 1),
                "pos_left": fig.get("pos_left"),
                "pos_top": fig.get("pos_top"),
                "pos_right": fig.get("pos_right"),
                "pos_bottom": fig.get("pos_bottom"),
            })
    print(f"{len(rows)} Abbildungen mit Caption gesammelt.")
    return pd.DataFrame(rows)


# ======================================================================
#  Bild aus der PDF rendern
# ======================================================================
_pdf_cache = {}


def render_figure(row):
    """Rendert die Figur-Region als PIL.Image, oder None bei Fehler."""
    import fitz
    from PIL import Image
    pdf_path = row["pdf_path"]
    if not pdf_path:
        return None
    try:
        if pdf_path not in _pdf_cache:
            _pdf_cache[pdf_path] = fitz.open(pdf_path)
        doc = _pdf_cache[pdf_path]
        idx = int(row["pos_page"]) - 1 if PAGE_ONE_INDEXED else int(row["pos_page"])
        idx = max(0, min(idx, doc.page_count - 1))
        page = doc[idx]
        clip = None
        if all(row[k] is not None for k in ("pos_left", "pos_top", "pos_right", "pos_bottom")):
            clip = fitz.Rect(row["pos_left"] - BBOX_PADDING, row["pos_top"] - BBOX_PADDING,
                             row["pos_right"] + BBOX_PADDING, row["pos_bottom"] + BBOX_PADDING) & page.rect
        pix = page.get_pixmap(matrix=fitz.Matrix(RENDER_ZOOM, RENDER_ZOOM), clip=clip)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    except Exception:
        return None


# ======================================================================
#  CLIP laden + klassifizieren
# ======================================================================
def load_clip():
    import torch
    import open_clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Lade CLIP ({CLIP_MODEL}) auf {device} ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED)
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    model = model.to(device).eval()

    # gemittelte Text-Features pro Klasse
    with torch.no_grad():
        class_feats = []
        for c in CLASSES:
            toks = tokenizer(PROMPTS[c]).to(device)
            tf = model.encode_text(toks)
            tf = tf / tf.norm(dim=-1, keepdim=True)
            mean = tf.mean(dim=0)
            class_feats.append(mean / mean.norm())
        class_feats = torch.stack(class_feats)  # [3, dim]
    return model, preprocess, device, class_feats


def classify_pool(pool):
    """Gibt fuer jede Zeile einen Klassen-Index (0/1/2) oder -1 (Bild fehlt)."""
    import torch
    model, preprocess, device, class_feats = load_clip()

    preds = np.full(len(pool), -1, dtype=int)
    batch_imgs, batch_rows = [], []
    n_ok, n_fail = 0, 0

    def flush():
        nonlocal batch_imgs, batch_rows
        if not batch_imgs:
            return
        with torch.no_grad():
            x = torch.stack(batch_imgs).to(device)
            feats = model.encode_image(x)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            sims = feats @ class_feats.T          # [B, 3]
            idx = sims.argmax(dim=1).cpu().numpy()
        for r, p in zip(batch_rows, idx):
            preds[r] = int(p)
        batch_imgs, batch_rows = [], []

    for i, (_, row) in enumerate(pool.iterrows()):
        img = render_figure(row)
        if img is None:
            n_fail += 1
            continue
        batch_imgs.append(preprocess(img))
        batch_rows.append(i)
        n_ok += 1
        if len(batch_imgs) >= BATCH_SIZE:
            flush()
        if (i + 1) % 200 == 0:
            print(f"  ... {i + 1}/{len(pool)} Bilder verarbeitet")
    flush()
    print(f"CLIP fertig: {n_ok} Bilder klassifiziert, {n_fail} ohne Bild (uebersprungen).")
    return preds


# ======================================================================
#  Auswahl-Logik (ohne CLIP testbar)
# ======================================================================
def build_sets(pool, preds, n_train=N_TRAIN, n_eval=N_EVAL, seed=RANDOM_STATE):
    """pool: DataFrame; preds: array von Klassen-Indizes (-1 = unbekannt)."""
    pool = pool.reset_index(drop=True).copy()
    pool["pred_idx"] = preds
    name = {0: "Plot", 1: "Diagram", 2: "Photo", -1: "Unknown"}
    pool["prescreen_pred"] = pool["pred_idx"].map(name)

    rng = np.random.RandomState(seed)

    # Eval: ZUFAELLIG
    n_eval = min(n_eval, len(pool))
    eval_idx = rng.choice(pool.index.values, size=n_eval, replace=False)
    eval_df = pool.loc[eval_idx].copy()
    remainder = pool.drop(index=eval_idx)

    # Training: AUSGEWOGEN -> seltene Klassen (Diagram/Photo) zuerst
    is_rare = remainder["pred_idx"].isin([1, 2])
    rare = remainder[is_rare].sample(frac=1, random_state=seed)
    rest = remainder[~is_rare].sample(frac=1, random_state=seed)
    take_rare = rare.iloc[:n_train]
    need = max(0, n_train - len(take_rare))
    train_df = pd.concat([take_rare, rest.iloc[:need]]).sample(frac=1, random_state=seed)
    return train_df, eval_df


def write_csv(df_rows, path):
    out = df_rows.copy().reset_index(drop=True)
    out.insert(0, "annotation_id", range(1, len(out) + 1))
    for c in ["Maschine_Label", "manual_plot", "manual_photo", "manual_diagram",
              "unclear", "notes"]:
        out[c] = ""
    out = out[OUT_COLS]
    out.to_csv(path, sep=";", index=False, encoding="utf-8-sig")


def summarize(df, name):
    counts = df["prescreen_pred"].value_counts().to_dict()
    parts = ", ".join(f"{k}: {counts.get(k, 0)}" for k in ["Plot", "Diagram", "Photo", "Unknown"])
    print(f"  {name} ({len(df)}): {parts}")


# ======================================================================
#  Hauptablauf
# ======================================================================
def main():
    json_dir = sys.argv[1] if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]) else None
    pdf_dir = sys.argv[2] if len(sys.argv) > 2 and os.path.isdir(sys.argv[2]) else None
    if not json_dir or not pdf_dir:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        if not json_dir:
            json_dir = filedialog.askdirectory(title="JSON-Ordner des neuen Samples waehlen") or None
        if json_dir and not pdf_dir:
            pdf_dir = filedialog.askdirectory(title="PDF-Ordner des neuen Samples waehlen") or None
        root.destroy()
    if not json_dir or not pdf_dir:
        print("JSON- oder PDF-Ordner fehlt. Abbruch."); return

    out_dir = sys.argv[3] if len(sys.argv) > 3 else (os.path.dirname(os.path.abspath(json_dir)) or ".")
    os.makedirs(out_dir, exist_ok=True)

    pool = collect_figures(json_dir, pdf_dir)
    if pool.empty:
        raise SystemExit("Keine Abbildungen gefunden - stimmt der JSON-Ordner?")
    missing = pool["pdf_path"].isna().sum()
    if missing:
        print(f"Hinweis: {missing} Abbildungen ohne passende PDF (werden als 'Unknown' behandelt).")

    preds = classify_pool(pool)

    print("\nVor-Klassifikation des Pools:")
    for j, c in enumerate(["Plot", "Diagram", "Photo"]):
        print(f"  {c:8}: {(preds == j).sum()}")
    print(f"  Unknown : {(preds == -1).sum()}")

    train_df, eval_df = build_sets(pool, preds)

    train_path = os.path.join(out_dir, "annotation_task_train_1000.csv")
    eval_path = os.path.join(out_dir, "annotation_task_eval_200.csv")
    write_csv(train_df, train_path)
    write_csv(eval_df, eval_path)

    print("\nErgebnis (CLIP-Vorsortierung):")
    summarize(train_df, "TRAINING (ausgewogen)")
    summarize(eval_df, "EVAL (zufaellig)")
    print(f"\nGeschrieben:\n  {train_path}\n  {eval_path}")
    print("\nNaechster Schritt: beide CSVs in annotate.py oeffnen und SELBST labeln.")
    print("'prescreen_pred' ist nur Info und wird beim Annotieren nicht angezeigt.")


if __name__ == "__main__":
    main()
