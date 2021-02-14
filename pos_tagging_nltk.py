import nltk, json
from nltk import word_tokenize
from fonctions import *

# si besoin, télécharger le tagger pour nltk
#nltk.download('averaged_perceptron_tagger')

# récupération des titres dans le dictionnaire de la forme {chemin : titre}
dicTitres = lire_json('articles.json')

#titres = list(dicTitres.values())

dicTag = {}

for path, titre in dicTitres.items():
    tokens = word_tokenize(titre, language='french') # tokenisation du titre
    tags = nltk.pos_tag(tokens) # tags des tokens
    dicTag.setdefault(path, [titre, tags])

ecrire_json('articlesTags.json', dicTag)