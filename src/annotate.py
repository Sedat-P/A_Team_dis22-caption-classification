# -*- coding: utf-8 -*-
"""
Interaktives Annotations-Tool für wissenschaftliche Abbildungen (A-Team).

Workflow:
  - Liest die CSV (Semikolon-getrennt) mit den 1000 ausgewählten Abbildungen.
  - Findet zu jeder Zeile die passende Figure in der JSON (über Figurennummer + Caption).
  - Schneidet die Figure-Region aus der zugehörigen PDF aus und zeigt sie an.
  - Du vergibst per Tastendruck ein Label. Die CSV wird nach jeder Eingabe gespeichert.

Tasten (Hauptmenü):
  1 = Plot
  2 = Diagram
  3 = Photo
  4 = Multilabel  -> Untermenü
  5 = Skip (überspringen)
  6 = needs Review
  9 = Abbrechen & speichern
  f = ganze Seite anzeigen (umschalten) / r = neu zeichnen
  b oder <- = eine Abbildung zurück

Multilabel-Untermenü:
  1 = Plot + Diagram
  2 = Plot + Photo
  3 = Diagram + Photo
  4 = Plot + Diagram + Photo
  Esc = zurück zum Hauptmenü

Benötigt:  pip install pymupdf pillow
(tkinter ist bei Python normalerweise dabei; auf Linux ggf. "sudo apt install python3-tk")
"""

import os
import re
import sys
import csv
import glob
import difflib
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Fehler: PyMuPDF fehlt. Bitte installieren:  pip install pymupdf")
    sys.exit(1)

try:
    from PIL import Image, ImageTk
except ImportError:
    print("Fehler: Pillow fehlt. Bitte installieren:  pip install pillow")
    sys.exit(1)


# ======================================================================
#  KONFIGURATION  -- bei Bedarf hier anpassen
# ======================================================================
CSV_PATH = "annotation_task_1000.csv"        # wird per Popup gewählt, falls nicht gefunden
JSON_DIR_CANDIDATES = ["jsons", "json", "."]  # Standard-Ordner für JSON-Dateien
PDF_DIR_CANDIDATES  = ["pdfs", "pdf", "."]    # Standard-Ordner für PDF-Dateien

PAGE_ONE_INDEXED = True   # pos_page ist 1-basiert (Seite 1 = erste Seite)
BBOX_PADDING     = 14     # Rand (PDF-Punkte) um die Figure herum
RENDER_ZOOM      = 2.5    # Render-Auflösung (höher = schärfer, langsamer)
MAX_IMG_W        = 1000   # max. Anzeigebreite in Pixeln
MAX_IMG_H        = 680    # max. Anzeigehöhe in Pixeln

# CSV-Spalten
COL_ID      = "annotation_id"
COL_JSON    = "source_json"
COL_CAPTION = "caption"
COL_MACHINE = "Maschine_Label"
COL_PLOT    = "manual_plot"
COL_PHOTO   = "manual_photo"
COL_DIAGRAM = "manual_diagram"
COL_UNCLEAR = "unclear"
COL_NOTES   = "notes"
COL_LABEL   = "manual_label"   # wird hinzugefügt, falls nicht vorhanden (lesbares Ergebnis)


# ======================================================================
#  CSV laden / speichern
# ======================================================================
def load_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        fieldnames = list(reader.fieldnames)
        rows = [dict(r) for r in reader]
    if COL_LABEL not in fieldnames:
        fieldnames.append(COL_LABEL)
        for r in rows:
            r[COL_LABEL] = r.get(COL_LABEL, "")
    return rows, fieldnames


def save_csv(path, rows, fieldnames):
    # erst in temporäre Datei schreiben, dann ersetzen (sicher gegen Abbrüche)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";",
                                quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    os.replace(tmp, path)


# ======================================================================
#  Ordner / Dateien finden
# ======================================================================
def detect_dir(candidates, base, filenames):
    """
    Liefert den ersten Kandidaten-Ordner, der mindestens eine der
    konkret benötigten Dateien (filenames) enthält. Sonst None.
    """
    for c in candidates:
        p = c if os.path.isabs(c) else os.path.join(base, c)
        if os.path.isdir(p):
            for fn in filenames:
                if os.path.isfile(os.path.join(p, fn)):
                    return os.path.abspath(p)
    return None


def count_present(folder, filenames):
    """Wie viele der filenames liegen wirklich im folder?"""
    if not folder or not os.path.isdir(folder):
        return 0
    return sum(1 for fn in filenames if os.path.isfile(os.path.join(folder, fn)))


def ask_dir(title, root, initialdir=None):
    d = filedialog.askdirectory(title=title, parent=root, initialdir=initialdir)
    return d or None


# ======================================================================
#  Figure-Matching: passende Figure in der JSON zur CSV-Zeile finden
# ======================================================================
def norm_text(s):
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def fig_key(s):
    """Extrahiert eine Figuren-Kennung wie 's5', '7', '4' aus einem Text."""
    if not s:
        return None
    m = re.search(r"\b(?:fig(?:ure)?s?)\.?\s*(s?\s*\d+[a-z]?)", s, flags=re.I)
    if not m:
        return None
    return re.sub(r"\s+", "", m.group(1)).lower()


def find_figure(csv_caption, figures):
    """
    Gibt (figure_dict, confidence_float) zurück.
    confidence ~1.0 = sicher, niedrig = unsicher.
    """
    if not figures:
        return None, 0.0

    csv_norm = norm_text(csv_caption)
    csv_key = fig_key(csv_caption)

    best, best_score = None, -1.0
    for fg in figures:
        jcap = fg.get("caption", "") or ""
        jname = fg.get("name", "") or ""
        j_norm = norm_text(jcap)
        j_key = fig_key(jname) or fig_key(jcap)

        score = 0.0
        # 1) Figurennummer stimmt überein -> starkes Signal
        if csv_key and j_key and csv_key == j_key:
            score += 0.6
        # 2) Prefix/Substring (CSV-Caption ist oft gekürzt)
        if csv_norm and j_norm:
            if j_norm.startswith(csv_norm) or csv_norm.startswith(j_norm):
                score += 0.3
            score += 0.4 * difflib.SequenceMatcher(None, csv_norm, j_norm).ratio()

        if score > best_score:
            best, best_score = fg, score

    # confidence grob normalisieren auf 0..1
    conf = min(1.0, best_score / 1.0) if best_score > 0 else 0.0
    return best, conf


# ======================================================================
#  PDF-Rendering
# ======================================================================
_pdf_cache = {}   # pdf_path -> fitz.Document


def get_doc(pdf_path):
    if pdf_path not in _pdf_cache:
        _pdf_cache[pdf_path] = fitz.open(pdf_path)
    return _pdf_cache[pdf_path]


def render_figure(pdf_path, figure, full_page=False):
    """Rendert die Figure-Region (oder die ganze Seite) als PIL.Image."""
    doc = get_doc(pdf_path)
    page_no = int(figure.get("pos_page", 1))
    idx = page_no - 1 if PAGE_ONE_INDEXED else page_no
    idx = max(0, min(idx, doc.page_count - 1))
    page = doc[idx]

    mat = fitz.Matrix(RENDER_ZOOM, RENDER_ZOOM)
    if full_page:
        pix = page.get_pixmap(matrix=mat)
    else:
        try:
            clip = fitz.Rect(
                figure["pos_left"] - BBOX_PADDING,
                figure["pos_top"] - BBOX_PADDING,
                figure["pos_right"] + BBOX_PADDING,
                figure["pos_bottom"] + BBOX_PADDING,
            ) & page.rect
        except Exception:
            clip = None
        pix = page.get_pixmap(matrix=mat, clip=clip)

    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    img.thumbnail((MAX_IMG_W, MAX_IMG_H), Image.LANCZOS)
    return img


# ======================================================================
#  Label-Logik
# ======================================================================
LABELS = {
    "plot":          dict(text="Plot",                  plot=1, diagram=0, photo=0, unclear=0),
    "diagram":       dict(text="Diagram",               plot=0, diagram=1, photo=0, unclear=0),
    "photo":         dict(text="Photo",                 plot=0, diagram=0, photo=1, unclear=0),
    "plot+diagram":  dict(text="Plot+Diagram",          plot=1, diagram=1, photo=0, unclear=0),
    "plot+photo":    dict(text="Plot+Photo",            plot=1, diagram=0, photo=1, unclear=0),
    "diagram+photo": dict(text="Diagram+Photo",         plot=0, diagram=1, photo=1, unclear=0),
    "all":           dict(text="Plot+Diagram+Photo",    plot=1, diagram=1, photo=1, unclear=0),
    "skip":          dict(text="Skip",                  plot=0, diagram=0, photo=0, unclear=0),
    "review":        dict(text="Needs Review",          plot=0, diagram=0, photo=0, unclear=1),
}


def apply_label(row, key):
    d = LABELS[key]
    row[COL_LABEL]   = d["text"]
    row[COL_PLOT]    = "1" if d["plot"] else ""
    row[COL_DIAGRAM] = "1" if d["diagram"] else ""
    row[COL_PHOTO]   = "1" if d["photo"] else ""
    row[COL_UNCLEAR] = "1" if d["unclear"] else ""


def is_done(row):
    return bool((row.get(COL_LABEL) or "").strip())


# ======================================================================
#  GUI-Anwendung
# ======================================================================
class Annotator:
    def __init__(self, root, rows, fieldnames, csv_path, json_dir, pdf_dir,
                 start_idx, end_idx):
        self.root = root
        self.rows = rows
        self.fieldnames = fieldnames
        self.csv_path = csv_path
        self.json_dir = json_dir
        self.pdf_dir = pdf_dir
        self.start_idx = start_idx
        self.end_idx = end_idx          # exklusiv
        self.i = start_idx
        self.multilabel_mode = False
        self.full_page = False
        self.cur_img = None             # Referenz halten (sonst GC)
        self.cur_figure = None
        self.cur_pdf = None

        self._build_ui()
        self._bind_keys()
        self.show_current()

    # ---------- UI-Aufbau ----------
    def _build_ui(self):
        self.root.title("A-Team Annotation")
        self.root.configure(bg="#1e1e1e")
        self.root.geometry("1100x900")

        self.header = tk.Label(self.root, text="", font=("Helvetica", 14, "bold"),
                               fg="#ffffff", bg="#1e1e1e", anchor="w", justify="left")
        self.header.pack(fill="x", padx=12, pady=(10, 2))

        self.machine = tk.Label(self.root, text="", font=("Helvetica", 11),
                                fg="#999999", bg="#1e1e1e", anchor="w")
        self.machine.pack(fill="x", padx=12, pady=(0, 6))

        self.img_label = tk.Label(self.root, bg="#2b2b2b")
        self.img_label.pack(expand=True, padx=12, pady=6)

        self.caption = tk.Label(self.root, text="", font=("Helvetica", 10),
                                fg="#d0d0d0", bg="#1e1e1e", wraplength=1050,
                                anchor="w", justify="left")
        self.caption.pack(fill="x", padx=12, pady=(4, 4))

        self.legend = tk.Label(self.root, text="", font=("Helvetica", 12, "bold"),
                               fg="#7ec8ff", bg="#111111", anchor="w", justify="left")
        self.legend.pack(fill="x", padx=0, pady=0, ipady=8)

        self._set_main_legend()

    def _set_main_legend(self):
        self.legend.config(
            text="  [1] Plot   [2] Diagram   [3] Photo   [4] Multilabel   "
                 "[5] Skip   [6] Review   [9] Speichern&Ende      "
                 "([f] ganze Seite  [b/←] zurück)",
            fg="#7ec8ff")

    def _set_multi_legend(self):
        self.legend.config(
            text="  MULTILABEL:  [1] Plot+Diagram   [2] Plot+Photo   "
                 "[3] Diagram+Photo   [4] Plot+Diagram+Photo      [Esc] zurück",
            fg="#ffd479")

    # ---------- Tasten ----------
    def _bind_keys(self):
        self.root.bind("<Key>", self.on_key)
        self.root.bind("<Left>", lambda e: self.go_back())

    def on_key(self, event):
        k = event.keysym.lower()
        ch = event.char

        if self.multilabel_mode:
            if k == "escape":
                self.multilabel_mode = False
                self._set_main_legend()
                return
            mapping = {"1": "plot+diagram", "2": "plot+photo",
                       "3": "diagram+photo", "4": "all"}
            if ch in mapping:
                self.commit(mapping[ch])
            return

        # Hauptmenü
        if ch == "1":
            self.commit("plot")
        elif ch == "2":
            self.commit("diagram")
        elif ch == "3":
            self.commit("photo")
        elif ch == "4":
            self.multilabel_mode = True
            self._set_multi_legend()
        elif ch == "5":
            self.commit("skip")
        elif ch == "6":
            self.commit("review")
        elif ch == "9":
            self.quit_save()
        elif k in ("f",):
            self.full_page = not self.full_page
            self.show_current()
        elif k in ("r",):
            self.show_current()
        elif k in ("b",):
            self.go_back()

    # ---------- Navigation ----------
    def commit(self, label_key):
        row = self.rows[self.i]
        apply_label(row, label_key)
        save_csv(self.csv_path, self.rows, self.fieldnames)  # Autosave
        self.multilabel_mode = False
        self._set_main_legend()
        self.advance()

    def advance(self):
        self.i += 1
        self.full_page = False
        if self.i >= self.end_idx:
            self.finish()
        else:
            self.show_current()

    def go_back(self):
        if self.i > self.start_idx:
            self.i -= 1
            self.full_page = False
            self.multilabel_mode = False
            self._set_main_legend()
            self.show_current()

    def finish(self):
        save_csv(self.csv_path, self.rows, self.fieldnames)
        done = sum(1 for r in self.rows if is_done(r))
        messagebox.showinfo("Fertig",
                            f"Bereich abgeschlossen.\n\nGespeichert in:\n{self.csv_path}\n\n"
                            f"Insgesamt annotiert: {done} / {len(self.rows)}")
        self.root.destroy()

    def quit_save(self):
        save_csv(self.csv_path, self.rows, self.fieldnames)
        done = sum(1 for r in self.rows if is_done(r))
        messagebox.showinfo("Gespeichert",
                            f"Aktueller Stand gespeichert.\n\n{self.csv_path}\n\n"
                            f"Annotiert: {done} / {len(self.rows)}\n"
                            f"Zuletzt bei Index {self.i + 1} (annotation_id "
                            f"{self.rows[self.i].get(COL_ID,'?')}).")
        self.root.destroy()

    # ---------- Anzeige ----------
    def show_current(self):
        row = self.rows[self.i]
        ann_id = row.get(COL_ID, "?")
        json_name = row.get(COL_JSON, "")
        csv_cap = row.get(COL_CAPTION, "")
        machine = row.get(COL_MACHINE, "")
        prev = (row.get(COL_LABEL) or "").strip()

        total = self.end_idx - self.start_idx
        pos_in_run = self.i - self.start_idx + 1
        prev_txt = f"   ✓ bereits: {prev}" if prev else ""
        self.header.config(
            text=f"[{pos_in_run}/{total}]   annotation_id {ann_id}   "
                 f"({json_name}){prev_txt}")
        self.machine.config(text=f"Maschinen-Label: {machine or '—'}")

        # JSON laden + Figure finden + rendern
        img, info = self._load_image(json_name, csv_cap)
        if img is not None:
            self.cur_img = ImageTk.PhotoImage(img)
            self.img_label.config(image=self.cur_img, text="")
        else:
            self.cur_img = None
            self.img_label.config(image="", text=info,
                                  fg="#ff8080", font=("Helvetica", 12))

        view = "GANZE SEITE" if self.full_page else "Figure-Ausschnitt"
        self.caption.config(text=f"[{view}]  {info_prefix(info, img)}\n\n{csv_cap}")

    def _load_image(self, json_name, csv_cap):
        json_path = os.path.join(self.json_dir, json_name)
        if not os.path.isfile(json_path):
            return None, f"JSON nicht gefunden:\n{json_path}"
        try:
            import json as _json
            with open(json_path, "r", encoding="utf-8") as f:
                data = _json.load(f)
        except Exception as e:
            return None, f"JSON-Fehler: {e}"

        figures = data.get("figures", []) or []
        figure, conf = find_figure(csv_cap, figures)
        if figure is None:
            return None, "Keine Figure in der JSON gefunden."

        # PDF-Pfad bestimmen: bevorzugt aus JSON, sonst aus json-Namen ableiten
        pdf_name = None
        for key in ("local_pdf_path", "pdf_path"):
            val = data.get(key) or (data.get("metadata", {}) or {}).get(key)
            if val:
                pdf_name = os.path.basename(val)
                break
        if not pdf_name:
            pdf_name = os.path.splitext(json_name)[0] + ".pdf"

        pdf_path = os.path.join(self.pdf_dir, pdf_name)
        if not os.path.isfile(pdf_path):
            # zweiter Versuch: aus json-Namen abgeleitet
            alt = os.path.join(self.pdf_dir, os.path.splitext(json_name)[0] + ".pdf")
            if os.path.isfile(alt):
                pdf_path = alt
            else:
                return None, f"PDF nicht gefunden:\n{pdf_path}"

        try:
            img = render_figure(pdf_path, figure, full_page=self.full_page)
        except Exception as e:
            return None, f"Render-Fehler: {e}"

        warn = ""
        if conf < 0.5:
            warn = "  ⚠ Figure-Zuordnung unsicher – bei Zweifel [f] ganze Seite ansehen."
        return img, f"Figure: {figure.get('name','?')}  (Seite {figure.get('pos_page','?')}){warn}"


def info_prefix(info, img):
    return info if img is None else info


# ======================================================================
#  Start: Pfade & Bereich abfragen, dann GUI starten
# ======================================================================
def main():
    root = tk.Tk()
    root.withdraw()  # vorerst verstecken

    base = os.getcwd()

    # --- CSV finden ---
    csv_path = CSV_PATH if os.path.isfile(CSV_PATH) else None
    if not csv_path:
        csv_path = filedialog.askopenfilename(
            title="Annotations-CSV wählen",
            filetypes=[("CSV", "*.csv"), ("Alle", "*.*")])
        if not csv_path:
            print("Keine CSV gewählt. Abbruch.")
            return
    csv_dir = os.path.dirname(os.path.abspath(csv_path)) or base

    rows, fieldnames = load_csv(csv_path)
    n = len(rows)
    done = sum(1 for r in rows if is_done(r))

    # benötigte Dateinamen aus den ersten Zeilen der CSV (für zuverlässige Erkennung)
    json_names = [r.get(COL_JSON, "") for r in rows if r.get(COL_JSON)]
    sample_json = json_names[:40]
    sample_pdf = [os.path.splitext(j)[0] + ".pdf" for j in sample_json]

    # --- JSON-Ordner finden (Popup, falls die echten Dateien nicht da sind) ---
    json_dir = detect_dir(JSON_DIR_CANDIDATES, csv_dir, sample_json)
    if not json_dir:
        messagebox.showinfo(
            "JSON-Ordner",
            "Die JSON-Dateien wurden nicht automatisch gefunden.\n\n"
            "Bitte im nächsten Fenster den Ordner auswählen, in dem die "
            "JSON-Dateien liegen.")
        json_dir = ask_dir("Ordner mit den JSON-Dateien wählen", root,
                           initialdir=csv_dir)
        if not json_dir:
            print("Kein JSON-Ordner gewählt. Abbruch.")
            return
        # Kontrolle: liegen dort wirklich die gesuchten JSONs?
        present = count_present(json_dir, sample_json)
        if present == 0:
            if not messagebox.askyesno(
                    "Keine JSONs gefunden",
                    f"Im gewählten Ordner wurde keine der erwarteten JSON-Dateien "
                    f"gefunden:\n{json_dir}\n\n"
                    f"Beispiel gesucht: {sample_json[0] if sample_json else '?'}\n\n"
                    f"Trotzdem fortfahren?"):
                return

    # --- PDF-Ordner finden ---
    pdf_dir = detect_dir(PDF_DIR_CANDIDATES, csv_dir, sample_pdf)
    if not pdf_dir:
        messagebox.showinfo(
            "PDF-Ordner",
            "Die PDF-Dateien wurden nicht automatisch gefunden.\n\n"
            "Bitte im nächsten Fenster den Ordner auswählen, in dem die "
            "PDF-Dateien liegen.")
        pdf_dir = ask_dir("Ordner mit den PDF-Dateien wählen", root,
                          initialdir=csv_dir)
        if not pdf_dir:
            print("Kein PDF-Ordner gewählt. Abbruch.")
            return
        present = count_present(pdf_dir, sample_pdf)
        if present == 0:
            if not messagebox.askyesno(
                    "Keine PDFs gefunden",
                    f"Im gewählten Ordner wurde keine der erwarteten PDF-Dateien "
                    f"gefunden:\n{pdf_dir}\n\n"
                    f"Beispiel gesucht: {sample_pdf[0] if sample_pdf else '?'}\n\n"
                    f"Trotzdem fortfahren?"):
                return

    # --- Start- / End-Index abfragen ---
    first_unlabeled = next((idx for idx, r in enumerate(rows) if not is_done(r)), 0)
    start = simpledialog.askinteger(
        "Start",
        f"Bei welcher annotation_id starten? (1–{n})\n\n"
        f"Bereits annotiert: {done} / {n}\n"
        f"Erste offene id: {rows[first_unlabeled].get(COL_ID, first_unlabeled+1)}",
        parent=root, minvalue=1, maxvalue=n,
        initialvalue=int(rows[first_unlabeled].get(COL_ID, first_unlabeled + 1)))
    if start is None:
        return

    end = simpledialog.askinteger(
        "Ende",
        f"Bis zu welcher annotation_id (einschließlich)?\n"
        f"Leer/Abbrechen = bis zum Ende ({n}).",
        parent=root, minvalue=start, maxvalue=n, initialvalue=n)
    if end is None:
        end = n

    # annotation_id -> Listenindex (ids beginnen bei 1, fortlaufend)
    start_idx = start - 1
    end_idx = end          # exklusiv-Grenze in der Schleife -> end ist 1-basiert inkl.

    root.deiconify()
    Annotator(root, rows, fieldnames, csv_path, json_dir, pdf_dir,
              start_idx, end_idx)
    root.mainloop()


if __name__ == "__main__":
    main()
