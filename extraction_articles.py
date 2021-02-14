import glob
from bs4 import BeautifulSoup
from fonctions import *

# lecture binaire des lignes d'un fichier texte
def lire_lignes(chemin):
    with open(chemin, "rb") as fin:
        lignes = fin.readlines()
    return lignes

dic = {}

# ajout des titres des fichiers du dossier 2009 dans la variable dic
for fic in glob.glob("french-docs/2009/*/*/*"):
    contenu = lire_lignes(fic)
    titre = contenu[2].decode("latin-1") # le titre correspond à la troisième ligne du fichier
    titre = titre[:-1]
    dic[fic] = titre
	
# ajout des titres des fichiers du dossier 2008 dans la variable dic
for path in glob.glob('french-docs/2008/*/*/*/*'):
    f = lire_fichier(path)
    soup  = BeautifulSoup(f)
    titre = soup.p.text 	# le titre correspond à la première balise <p>
    dic[path] = titre
    
ecrire_json("articles.json", dic)