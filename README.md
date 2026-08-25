# Die visuelle Sprache wissenschaftlicher Preprints

**Caption-basierte Klassifikation von Abbildungen auf bioRxiv**

Projektmodul DIS22 · Technische Hochschule Köln · Sommersemester 2026
Betreuung: Prof. Dr. Philipp Schaer, Fabian Haak

---

## Worum geht es?

Wissenschaftliche Abbildungen sind zentrale Träger der Argumentation. Dieses Projekt
untersucht, ob sich der **Typ einer Abbildung** (Plot, Diagram, Photo) **allein aus dem
Text ihrer Bildunterschrift** (Caption) bestimmen lässt – ohne das Bild selbst zu
analysieren.

Der methodische Bezugspunkt ist Lee et al. (2016), *Viziometrics*. Von dort stammt das
**Kategoriensystem**; das Verfahren dort ist jedoch **bildbasiert** (gelernt aus Pixeln).
Wir untersuchen den komplementären, **textbasierten** Weg: Er benötigt keine Bilddateien,
sondern nur die JSON-Metadaten, und skaliert dadurch mühelos auf ganze Korpora.

**Forschungsfrage:** Reicht der Text einer Caption aus, um den visuellen Typ einer
Abbildung zuverlässig zu erkennen?

---

## Kernergebnisse

Evaluation auf einem unabhängigen, paper-rein getrennten Gold-Set (311 Abbildungen):

| Verfahren                        | Macro-F1 | Micro-F1 | Exact-Match |
|----------------------------------|---------:|---------:|------------:|
| Majority (triviale Baseline)     |     0,28 |        – |           – |
| Regex (Lee + datengetrieben)     |    0,578 |    0,588 |       0,273 |
| Kombination Regex ODER Classifier|    0,696 |    0,735 |       0,350 |
| **Classifier (TF-IDF + LogReg)** | **0,700**| **0,755**|   **0,457** |

Ergebnisse pro Klasse (Classifier):

| Klasse  | Support | Precision | Recall |   F1 |
|---------|--------:|----------:|-------:|-----:|
| Plot    |     231 |      0,85 |   0,89 | 0,87 |
| Diagram |     132 |      0,62 |   0,71 | 0,66 |
| Photo   |      86 |      0,68 |   0,49 | 0,57 |

Weitere Befunde:

- **Multi-Label-Training lohnt sich:** Nimmt man zusammengesetzte Abbildungen ins
  Training auf, steigt der Macro-F1 von 0,61 auf 0,74 (gleiche Testdaten, gleicher
  Classifier, nur die Trainingsdaten unterscheiden sich).
- **Keine Kombination schlägt das Einzelmodell:** Sowohl eine ODER- als auch eine
  UND-Verknüpfung von Regex und Classifier wurde gemessen; beide bleiben hinter dem
  Classifier zurück (siehe Tabelle oben und `src/combine_regex_classifier.py`).
- **Jeder Typ hat eine sprachliche Signatur:** Die höchstgewichteten Begriffe pro Klasse
  sind u. a. *mean, error bars, axis* (Plot), *schematic, workflow, network* (Diagram),
  *scale bar, cells, images* (Photo).

Anwendung des trainierten Modells auf einen größeren Fremdkorpus (43.897 Abbildungen,
reine Demonstration der Skalierbarkeit, **ohne Evaluation**, da dort keine
Ground-Truth-Annotation vorliegt): Plot 83 %, Diagram 39 %, Photo 15 %; rund ein Drittel
der Abbildungen ist zusammengesetzt (häufigste Kombination Plot + Diagram). Die Anteile
summieren sich wegen Multi-Label nicht auf 100 %.

---

## Methodik in Kürze

**Klassifikator.** Jede Caption wird per **TF-IDF** (Unigramme und Bigramme,
`min_df=2`, `sublinear_tf`, englische Stoppwörter) in einen gewichteten Merkmalsvektor
überführt. Darauf arbeiten drei unabhängige **Logistic-Regression**-Modelle im
**One-vs-Rest**-Schema – je eines für Plot, Diagram und Photo. Dadurch sind
Mehrfachzuordnungen (Multi-Label) möglich. `class_weight="balanced"` gleicht die
Unterrepräsentation der Photo-Klasse aus.

**Regex-Baseline.** Eine Stichwortsuche mit belegter Wortliste: Die Seed-Begriffe stammen
aus den Kategorie-Definitionen von Lee et al., ergänzt um datengetrieben aus dem
Trainingssplit abgeleitete Terme (`src/derive_keywords.py`). Die Liste wird ausschließlich
in `src/regex_lee.py` gepflegt und von allen anderen Skripten importiert.

**Evaluationsdesign.** Die Aufteilung erfolgt **paper-rein** (`GroupShuffleSplit` über
`source_json`): Kein Preprint liegt gleichzeitig im Trainings- und im Evaluationsset. Das
verhindert Data Leakage durch geteiltes Vokabular innerhalb eines Papers. Alle Verfahren
werden auf demselben Gold-Set bewertet. Berichtet wird **Macro-F1**, da er alle drei
Klassen gleich gewichtet und so verhindert, dass die häufige Plot-Klasse das Ergebnis
dominiert.

**Annotation.** Die Labels wurden manuell vergeben, dabei **ausschließlich anhand des
Bildes** (aus dem PDF gerendert), nicht anhand der Caption. Das verhindert einen
Zirkelschluss und erhält die Caption als unabhängige Informationsquelle.

---

## Installation

Voraussetzung: Python 3.10 oder neuer.

```bash
git clone https://github.com/Sedat-P/A_Team_dis22-caption-classification
cd A_Team_dis22-caption-classification
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Für einen bitgenau identischen Paketstand liegt zusätzlich `requirements-lock.txt` bei
(`pip install -r requirements-lock.txt`). `requirements.txt` mit offenen Versionsgrenzen ist
der Normalfall; die Lock-Datei dient der exakten Nachstellung.

### Schnelltest

Der folgende Befehl prüft in unter einer Minute, ob die Installation vollständig ist. Er
trainiert den Classifier und wertet ihn auf dem Gold-Set aus:

```bash
cd src
python evaluate_on_goldset.py ../data/train_split.csv ../data/eval_split.csv
```

Erwartete Ausgabe: *Training: 943 nutzbare Zeilen*, *Eval: 311 nutzbare Zeilen* und für den
Classifier Macro-F1 0,700, Micro-F1 0,755, Exact-Match 0,457. Erzeugt wird
`../data/eval_split_goldeval.xlsx`.

Hinweis: Die Skripte nutzen `tkinter` für Dateidialoge. Unter Linux ggf. nachinstallieren
(`sudo apt install python3-tk`); unter Windows und macOS ist es in der Regel enthalten.
Alle **Auswertungsskripte** akzeptieren die Eingabepfade alternativ als
Kommandozeilenargumente und sind damit auch ohne grafische Oberfläche reproduzierbar; ohne
Argumente öffnen sie stattdessen einen Dateidialog. Lediglich das Annotationstool
`annotate.py` erfordert eine grafische Oberfläche.

---

## Reproduktion der Ergebnisse

Alle Befehle werden im Verzeichnis `src/` ausgeführt.

Die beiden annotierten Splits liegen dem Repository bei; die Schritte 3 und 4 sind damit
**ohne weitere Daten sofort ausführbar**. Die Schritte 1, 2 und 5 benötigen zusätzlich die
Roh-Preprints (JSON und PDF), die aus Lizenz- und Größengründen nicht enthalten sind – siehe
[`data/README.md`](data/README.md). Die Ergebnisse des Korpuslaufs aus Schritt 5 liegen als
fertige Dateien in `results/`.

| Schritt | Ohne Rohdaten ausführbar? |
|---|---|
| 1 Annotation | nein (JSON + PDF + grafische Oberfläche) |
| 2 Paper-reine Aufteilung | nein (Roh-Annotationen) |
| **3 Hauptevaluation** | **ja** |
| **4 Ergänzende Experimente** | **ja** |
| 5 Gesamtkorpus | nein (JSON-Verzeichnis) |

### 1. Annotation (optional, erzeugt die Ground Truth)

```bash
python annotate.py
```
Grafisches Annotationstool: rendert jede Abbildung über die JSON-Koordinaten passgenau aus
dem Original-PDF und erlaubt reine Tastaturbedienung
(`1` = Plot, `2` = Diagram, `3` = Photo, `4` = Multi-Label).
Ergänzend erzeugt `make_eval_set.py` die zusätzliche Stichprobe für das Evaluationsset.

### 2. Paper-reine Aufteilung

```bash
python make_paper_split.py <annotation_train.csv> <annotation_eval.csv>
```
Führt die annotierten Dateien zusammen und teilt sie nach Preprint getrennt in
`train_split.csv` (943 Abbildungen) und `eval_split.csv` (311 Abbildungen).
Die Konsolenausgabe bestätigt: *Geteilte Papers: 0*.

### 3. Hauptevaluation

```bash
python evaluate_on_goldset.py ../data/train_split.csv ../data/eval_split.csv
```
Trainiert den Classifier auf dem Trainingssplit und bewertet ihn auf dem Gold-Set.
Erzeugt `eval_split_goldeval.xlsx` mit vier Blättern: **Scores** (Vergleich der
Verfahren), **Details** (Vorhersage je Abbildung inkl. Fehlermarkierung),
**Fehleranalyse** (TP/FP/FN, Verwechslungsmatrix, Beispielfehler) und **Signatur**
(Top-30-Begriffe je Klasse aus dem trainierten Modell).

### 4. Ergänzende Experimente

```bash
python evaluate_multiseed.py ../data/train_split.csv        # Robustheit über mehrere Aufteilungen
python compare_classifiers.py ../data/train_split.csv       # Einzel- vs. Multi-Label-Training
python combine_regex_classifier.py ../data/train_split.csv ../data/eval_split.csv
python derive_keywords.py ../data/train_split.csv           # Herleitung der Regex-Begriffe
python regex_lee.py ../data/eval_split.csv                  # Regex-Baseline isoliert bewerten
python train_and_evaluate.py ../data/train_split.csv        # k-fold-Auswertung auf dem Trainingssplit
```

`evaluate_multiseed.py` ist bewusst nur mit `train_split.csv` aufzurufen – die Begründung
steht in [`data/README.md`](data/README.md).

### 5. Anwendung auf einen Gesamtkorpus

```bash
python run_full_corpus.py ../data/train_split.csv <json-verzeichnis> ../results/corpus
```
Trainiert das finale Modell und wendet es auf alle JSON-Dateien im angegebenen Verzeichnis
an. Captions ohne inhaltlichen Text (reine Label-Captions wie „Fig. 3“) werden über
`MIN_CAPTION_WORDS` ausgeschlossen und separat ausgewiesen. Ergebnis sind
`figures_classified.csv`, die Zusammenfassungen `summary_*.csv` sowie die
Verteilungsgrafiken. **Dieser Schritt ist keine Evaluation**, da hier keine Ground Truth
existiert; die Ausgabe weist explizit darauf hin.

---

## Projektstruktur

```
.
├── src/                            Produktiver Code
│   ├── regex_lee.py                Regex-Baseline (einzige Quelle der Wortlisten)
│   ├── train_and_evaluate.py       Pipeline-Bausteine, Metriken, Reporting (wird importiert)
│   ├── annotate.py                 Annotationstool (Tkinter + PyMuPDF)
│   ├── make_eval_set.py            Stichprobe für das Evaluationsset
│   ├── make_paper_split.py         Paper-reine Aufteilung in Train/Eval
│   ├── derive_keywords.py          Datengetriebene Herleitung der Regex-Begriffe
│   ├── evaluate_on_goldset.py      Hauptevaluation (Scores, Fehleranalyse, Signatur)
│   ├── evaluate_multiseed.py       Robustheit über mehrere Aufteilungen
│   ├── compare_classifiers.py      Experiment Einzel- vs. Multi-Label-Training
│   ├── combine_regex_classifier.py Kombination Regex/Classifier (ODER, UND)
│   └── run_full_corpus.py          Anwendung auf einen Gesamtkorpus
├── experimental/                   Erprobte, nicht weiterverfolgte Ansätze
│   ├── prepare_annotation_sets.py      Stratifizierte Vorauswahl per TF-IDF-Vorklassifikation
│   ├── prepare_annotation_sets_clip.py Dasselbe visuell per CLIP (benötigt requirements-clip.txt)
│   └── repair_captions_v2.py           Nachträgliche Reparatur gekürzter Captions aus dem JSON-Text
├── data/                           train_split.csv, eval_split.csv (Rohdaten separat)
├── results/                        Ergebnisdateien und Abbildungen des Korpuslaufs
├── docs/                           Projektpräsentation
├── requirements.txt
├── requirements-lock.txt           Exakt getesteter Paketstand
├── requirements-clip.txt           Nur für den CLIP-Ansatz in experimental/
├── LICENSE
└── README.md
```

Die Skripte in `experimental/` dienten der Vorbereitung der Annotationsstichprobe und sind
für die Reproduktion der berichteten Ergebnisse **nicht erforderlich**.

`regex_lee.py` und `train_and_evaluate.py` werden von den Auswertungsskripten **importiert**
und müssen deshalb im selben Verzeichnis liegen. Beide sind zusätzlich direkt ausführbar:
`regex_lee.py` bewertet die Regex-Baseline isoliert, `train_and_evaluate.py` führt eine
k-fold-Auswertung auf einer einzelnen annotierten Datei durch.

---

## Grenzen

- **Caption als Stellvertreter.** Bewertet wird, was die Caption über den Abbildungstyp
  verrät, nicht das Bild selbst. Das ist die bewusste Forschungsfrage, kein Ersatz für
  bildbasierte Verfahren.
- **Annotation.** Die Labels wurden im Team vergeben; ein systematisch gemessenes
  Inter-Annotator-Agreement (z. B. Cohen's Kappa) steht aus.
- **Korpusverteilung.** Die Verteilung auf dem Gesamtkorpus ist eine Modellschätzung und
  übernimmt die Fehlertendenzen des Classifiers; Photo ist wegen des niedrigen Recalls
  vermutlich unterschätzt.
- **Reichweite.** Grundlage ist ein Stichproben-Datensatz aus bioRxiv-Preprints; die
  Aussagen gelten für diesen Ausschnitt.

---

## Literatur

Lee, P., West, J. D., & Howe, B. (2016). *Viziometrics: Analyzing Visual Information in
the Scientific Literature.* IEEE Transactions on Big Data.
https://faculty.washington.edu/billhowe/publications/pdfs/lee2016viziometrics.pdf

---

## Autoren

- Ismail Salkovic
- Talha Arslan
- Necip Sedat Palaoglu

Technische Hochschule Köln, Projektmodul DIS22 (Sommersemester 2026).

## Lizenz

Der Quellcode steht unter der MIT-Lizenz, siehe [LICENSE](LICENSE). Für die verarbeiteten
Preprint-Daten gelten die Lizenzbedingungen der jeweiligen Originalveröffentlichungen.
