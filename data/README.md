# Datenverzeichnis

Dieses Verzeichnis enthält die beiden annotierten Splits, mit denen sich die im
Haupt-README berichteten Ergebnisse unmittelbar reproduzieren lassen. Die **Rohdaten**
(bioRxiv-Preprints als JSON und PDF) sind dagegen **nicht Teil dieses Repositories**:
Sie unterliegen den Lizenzbedingungen der jeweiligen Originalveröffentlichungen und sind
für eine Versionsverwaltung zu groß.

## Enthalten

| Datei | Beschreibung |
|---|---|
| `train_split.csv` | Trainingssplit, 943 annotierte Abbildungen aus 291 Preprints |
| `eval_split.csv` | Unabhängiges Gold-Set, 311 annotierte Abbildungen |

## Nicht enthalten (nur für einzelne Zusatzskripte nötig)

| Verzeichnis | Wofür |
|---|---|
| `jsons/` | Preprint-JSON-Dateien, für `run_full_corpus.py` und `make_eval_set.py` |
| `pdfs/` | Zugehörige PDFs, ausschließlich für das Annotationstool `annotate.py` |

Ohne diese Verzeichnisse sind alle Auswertungsskripte lauffähig; lediglich Annotation und
Korpusanwendung lassen sich nicht wiederholen. Die Ergebnisse des Korpuslaufs liegen als
fertige Dateien in `../results/`.

## Hinweis zur Herkunft von `eval_split.csv`

Die hier abgelegte Fassung von `eval_split.csv` wurde aus der Ergebnisdatei
`results/eval_split_goldeval.csv` **rekonstruiert**. Sie enthält die vollständigen
Gold-Labels aller 311 Abbildungen und reproduziert die berichteten Kennzahlen exakt
(Macro-F1 0,700 / Micro-F1 0,755 / Exact-Match 0,457).

Die Spalte `source_json` konnte dabei **nicht** wiederhergestellt werden, da sie in der
Ergebnisdatei nicht mitgeführt wird. Daraus folgen zwei Einschränkungen:

- `make_paper_split.py` lässt sich mit dieser Datei nicht nachvollziehen. Die
  paper-reine Trennung wurde bei der ursprünglichen Erzeugung durchgeführt und mit
  *Geteilte Papers: 0* bestätigt, ist hier aber nicht erneut überprüfbar.
- `evaluate_multiseed.py` darf **nicht** mit beiden Splits zugleich aufgerufen werden.
  Ohne `source_json` fielen alle 311 Eval-Zeilen in eine einzige Gruppe, und die
  Aufteilung wäre nicht mehr paper-rein. Für diesen Robustheitstest ist ausschließlich
  `train_split.csv` zu verwenden:

  ```bash
  python evaluate_multiseed.py ../data/train_split.csv
  ```

Alle übrigen Skripte sind von der Einschränkung nicht betroffen.

## Format der Annotationsdateien

CSV, Trennzeichen `;`, Kodierung `utf-8-sig`. Relevante Spalten:

| Spalte | Bedeutung |
|---|---|
| `annotation_id` | Laufende Nummer |
| `source_json` | Quell-Preprint; Gruppierungsschlüssel für die paper-reine Aufteilung |
| `caption` | Bildunterschrift, die einzige Eingabe des Klassifikators |
| `manual_plot`, `manual_diagram`, `manual_photo` | Manuelle Labels, `1` = trifft zu |
| `unclear` | Markierung unklarer Fälle |

Die Labels wurden ausschließlich anhand des gerenderten Bildes vergeben, nicht anhand der
Caption. Eine Abbildung kann mehrere Labels gleichzeitig tragen (Multi-Label).

Sind die Roh-Annotationen vorhanden, werden die Splits so erzeugt:

```bash
cd ../src
python make_paper_split.py <annotation_train.csv> <annotation_eval.csv>
```

Die Aufteilung erfolgt paper-rein, das heißt kein Preprint erscheint gleichzeitig in beiden
Dateien. Die Konsolenausgabe bestätigt dies mit *Geteilte Papers: 0*.

## Datenquelle

Preprints der Plattform bioRxiv. Der zugrunde liegende Stichproben-Datensatz wurde im
Rahmen des Projektmoduls DIS22 bereitgestellt. Der für die Korpusanwendung genutzte
größere Datensatz stammt aus einem anderen Projektteam und diente ausschließlich der
Demonstration der Skalierbarkeit, nicht der Evaluation.
