# Erprobte, nicht weiterverfolgte Ansätze

Dieses Verzeichnis dokumentiert Arbeitsstände, die im Projektverlauf entwickelt, geprüft
und bewusst verworfen wurden. Sie sind **nicht Teil der finalen Pipeline** und werden hier
zur Nachvollziehbarkeit des Entscheidungswegs aufbewahrt.

## `repair_captions_v2.py`

Rekonstruiert vollständige Bildunterschriften aus dem Fließtextfeld der JSON-Dateien
(Abschnitt „Figure legends“), falls das Feld `figures[].caption` nur eine gekürzte
Beschriftung enthält.

**Warum verworfen:** Eine Prüfung der Trainingsdaten ergab, dass die Captions dort bereits
vollständig vorlagen (Median rund 82 Wörter). Eine nachträgliche Reparatur hätte die
Datengrundlage uneinheitlich gemacht. Stattdessen werden inhaltsleere Label-Captions in
`run_full_corpus.py` über den Schwellwert `MIN_CAPTION_WORDS` ausgeschlossen und separat
ausgewiesen.

## `prepare_annotation_sets.py`, `prepare_annotation_sets_clip.py`

Vorklassifikation der Abbildungen zur Beschleunigung der manuellen Annotation, in der
CLIP-Variante bildbasiert über ein Vision-Language-Modell.

**Warum verworfen:** Eine maschinelle Vorbelegung hätte die manuelle Annotation
beeinflusst und damit die Unabhängigkeit der Ground Truth gefährdet. Da die Labels
ausschließlich anhand des Bildeindrucks vergeben werden sollten, wurde ohne Vorschlagswerte
annotiert. Die CLIP-Variante bringt zudem eine schwergewichtige Abhängigkeit
(`open_clip`) mit, die für den textbasierten Ansatz nicht erforderlich ist.
