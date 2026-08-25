# Ergebnisse

Ausgewählte Ergebnisdateien der finalen Auswertung. Umfangreiche Zwischenausgaben
(insbesondere `figures_classified.csv` mit rund 44.000 Zeilen) sind aus Platzgründen nicht
enthalten und werden durch `src/run_full_corpus.py` neu erzeugt.

| Datei | Inhalt |
|---|---|
| `eval_split_goldeval.xlsx` | Hauptevaluation auf dem Gold-Set: Blätter *Scores*, *Details*, *Fehleranalyse*, *Signatur* |
| `summary_overall.csv` | Verteilung der Abbildungstypen über den Gesamtkorpus |
| `summary_by_category.csv` | Verteilung je Fachrichtung (rein deskriptiv) |
| `summary_combinations.csv` | Häufigkeit der Label-Kombinationen (Multi-Label-Struktur) |
| `dist_overall.png` | Balkendiagramm der Gesamtverteilung |
| `dist_by_category.png` | Balkendiagramm je Fachrichtung |

## Hinweis zur Interpretation

Die Korpuszahlen (`summary_*.csv`, `dist_*.png`) sind **Modellvorhersagen, keine
Evaluation**: Für den Gesamtkorpus liegt keine manuelle Ground Truth vor. Die Verteilung
übernimmt daher die Fehlertendenzen des Klassifikators. Insbesondere ist Photo aufgrund des
niedrigen Recalls (0,49) vermutlich unterschätzt, Diagram tendenziell überschätzt.

Belastbare Leistungskennzahlen finden sich ausschließlich in `eval_split_goldeval.xlsx`,
gemessen auf dem unabhängigen, paper-rein getrennten Gold-Set.
