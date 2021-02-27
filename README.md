# Projet RAPACE
> ***Regroupement d'Articles de Presse Avec des Clusters grâce aux Etiquettes***

Projet réalisé dans le cadre d'un cours de Master 2 par **HERRENO Paola** et **HIEBEL Nicolas**

## Contenu de chaque fichier

### Notebooks (.ipynb)

- **Clustering** : Expériences de clustering avec les étiquettes morphosyntaxiques
- **Clustering_EN** : Expériences de clustering avec les entités nommées
- **Elbow** : Calculs de la méthode du coude pour choisir le nombre de clusters 
- **Elbow_Experimentation** : Tests de retraits du plus gros cluster dans les calculs de la méthode du coude
- **Fusion_Experimentation** : Tests de la combinaison des vectorisation des étiquettes et des entités nommées
- **Stats_jeu_donnees** : Calculs de statistiques sur le corpus
- **Stanford_tagger** : Etiquetage morphosyntaxique avec le tagger français de Stanford : génère `articlesTags_Stanford.json`

### Scripts (.py)

- **extraction_EN** : Script d'extraction des entités nommées : génère `entites_nommees.json`
- **extraction_articles** : Script d'extraction des articles : génère `articles.json` (non présent sur le dépot)
- **fonctions** : contient les fonctions utilisés dans les différents scripts et notebooks
- **pos_tagging_nltk** : Script d'étiquetage morphosyntaxique avec le tagger de nltk : génère `articlesTags_nltk.json`

### Donnees (.json)

- **articlesTags_Stanford** : Liste des titres des articles avec les tags par le tagger de Stanford
- **articlesTags_nltk** : Liste des titres des articles avec les tags par le tagger de nltk
- **entites_nommees** : Liste des titres des articles avec les entités nommées

### Utilisation du Stanford Tagger

1. [**Télécharger le tagger**](https://nlp.stanford.edu/software/tagger.shtml#Download)
2. **Modifier dans `Stanford_tagger.ipynb`** : les chemins du tagger, du modèle choisi et de `java.exe` sur la machine
