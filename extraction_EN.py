import json
import spacy
from fonctions import *
nlp = spacy.load("fr_core_news_sm")

titres = lire_json('articles.json')

dic_entites = {}

for k, v in titres.items():
    doc = nlp(v)
    list_ent = []
    list_etiq = []
    for ent in doc.ents: 
        list_ent.append(ent.text)
        list_etiq.append(ent.label_)
    dic_entites[k] = []
    dic_entites[k].append(list_ent)
    dic_entites[k].append(list_etiq)
    dic_entites[k].append(v)
    
ecrire_json('entites_nommees.json', dic_entites)