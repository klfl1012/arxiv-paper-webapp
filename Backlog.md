# Backlog

Automatische Klassifikation der Paper:
    - Implementiere ein Modell zur Themenklassifikation der Paper. Du könntest ein vortrainiertes Modell wie BERT oder eine spezifische thematische Klassifikationsarchitektur verwenden, um die Paper automatisch in verschiedene Kategorien (z.B. „Machine Learning“, „Computer Vision“, „Natural Language Processing“ etc.) einzuordnen.

Zusammenfassungen generieren:
    - Verwende ein Textzusammenfassungsmodell (z.B. BERTSUM oder T5), um kurze, prägnante Zusammenfassungen der Paper zu erstellen und vergleiche sie mit dem originalen Abstract des echten Papers. (integriere ein Feedback System um die Daten zu Labeln -> Higher Lower Game)

pretrained model ziehen und auf ai paper fine tunen, dann auf den beiden task oben trainieren / testen

Zitierempfehlungen / Recommender System:
    - Ähnlichkeitsanalysen zwischen Papern durchführen. Diese Embeddings könnten dann visuell im Graph dargestellt werden, um Cluster oder Trends zu erkennen.

Erkennung von Einflussreichen Arbeiten:
    - Zitiercounter des Autors als Funktion die man im Vec Space model aktivieren kann: kennzeichnet alle Paper bei der der Autor zitiert oder mitgearbeitet hat im vec space model