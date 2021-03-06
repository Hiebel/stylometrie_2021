import json
import csv
import re
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from random import randint
        
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

from sklearn import metrics 
from scipy.spatial.distance import cdist 

# lecture d'un fichier au format json et récupération dans un dictionnaire
def lire_json(chemin):
    with open(chemin, "r", encoding="utf-8") as fin:
        dic = json.load(fin)
    return dic

# lecture des lignes d'un fichier 
def lire_fichier(chemin):
    with open(chemin, "r", encoding="utf-8") as fin:
        lignes = fin.read()
    return lignes

# ecriture d'un dictionnaire ou d'une liste au format json
def ecrire_json(chemin, dic):
    with open(chemin, "w", encoding="utf-8") as fout:
        fout.write(json.dumps(dic, ensure_ascii=False, indent=2))

# compter les effectifs de chaque clusters du model d'entrée
def effectifs_clusters(model):
    count = {}
    for l in model.labels_:
        count.setdefault(l, 0)
        count[l] += 1
    return count

# récupérer les noms des journaux des articles de presse contenus dans leurs chemins
def get_liste_journaux(liste_chemins):
    liste_journaux = []

    for c in liste_chemins:
        j = re.findall("_(.+)_", c)
        liste_journaux.append(j[0])
    return liste_journaux

# création d'un nouveau dictionnaire pour faciliter les traitements
# le dictionnaire d'entrée contient des tuples (mot, tag)
# on souhaite séparer d'un côté les mots, de l'autre les tags
def reorganiser_POS(dic):
    new_dic = {}
    for k, v in dic.items():
        mots = []
        tags = []
        for mot, tag in v[1]:
            mots.append(mot)
            tags.append(tag)
        new_dic[k] = []
        new_dic[k].append(mots)
        new_dic[k].append(tags)
        new_dic[k].append(v[0])
    return new_dic

# Vectorisation des documents
# liste_tags : tags des documents
# ngram_min : taille minimale des ngrams
# ngram_max : taille maximale des ngrams
def creer_X(liste_tags, ngram_min, ngram_max):
    #V = CountVectorizer()
    V = TfidfVectorizer(ngram_range=(ngram_min,ngram_max))
    tokenized_tags = [" ".join(x) for x in liste_tags]
    X = V.fit_transform(tokenized_tags).toarray()
    return X, V

# créer un model de clustering avec la méthode des k plus proches voisins
# nb_clusters : nombre de clusters souhaités
# X : liste des vecteurs représentant les documents
def creer_model_KM(nb_clusters, X):
    model = KMeans(n_clusters=nb_clusters)
    model.fit(X)
    return model

# création d'un dictionnaire contenant pour chaque cluster la liste d'articles avec leurs tags et vecteurs correspondants
# X : liste des vecteurs
# predictions : liste des prédictions pour chaque vecteur
# liste_tags : liste des tags des articles
# liste_titres : liste titres des articles
# les indices entre toutes les listes correspondent
def dic_complet(X, predictions, liste_tags, liste_titres, liste_journaux):
    dic_res = {}
    for i in range(len(X)):
        num_cluster = predictions[i]
        titre = liste_titres[i]
        tags = liste_tags[i]
        vecteur = X[i]
        journal = liste_journaux[i]
        dic_res.setdefault(num_cluster,  [])
        dic_res[num_cluster].append((titre, tags, vecteur, journal))
    return dic_res


# affichier des titres d'articles au hasard parmis chaque cluster
# taille_echantillons : nombre de titres par cluster
# dictionnaire : contient pour chaque cluster la liste d'articles avec leurs tags et vecteurs correspondants
def afficher_titres_hasard(taille_echantillons, dictionnaire):
    echantillons = []
    tags = []
    journaux = []
    for i in range(len(dictionnaire)):
        echantillons.append([])
        tags.append([])
        for j in range(taille_echantillons):
            indice = randint(0, len(dictionnaire[i])-1)
            echantillons[i].append(dictionnaire[i][indice][0])
            tags[i].append(dictionnaire[i][indice][1])
            journaux.append(dictionnaire[i][indice][3])

    for i in range(len(echantillons)):
        print("Cluster %s : %s articles" % (i, len(dictionnaire[i])))
        for j in range(len(echantillons[i])):
            print(echantillons[i][j])
            print(" ".join(tags[i][j]))
            print(journaux[i])
        print("-"*10)

# afficher les dimensions les plus caractéristiques de chaque cluster
def dimensions_clusters(model, vectorizer, nb_dim):
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()
    with open("dimensions_clusters.csv", "w", encoding="utf-8") as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(model.n_clusters):
            print("Cluster %d:" % i)
            row = []
            for ind in order_centroids[i, :nb_dim]:
                print('(%s)' % terms[ind].upper(), end=' | ')
                row.append(terms[ind].upper())
            spamwriter.writerow(row)
            print()
            print("-"*10)

# création du graphique de l'analyse en composantes principales
# X : liste des vecteurs
# predictions : liste des prédictions
def tracer_ACP(X, predictions):
    model_tsne = TSNE(learning_rate=100)
    transformed = model_tsne.fit_transform(X)
    x_axis = transformed[:, 0]
    y_axis = transformed[:, 1]
    plt.scatter(x_axis, y_axis, c=predictions)
    plt.title('Analyse en composantes principales')
    plt.savefig("ACP.png")
    plt.show()

# construction du dendrogram
# fonction récupérée sur 
# https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    
# créer le modèle avec un nombre de clusters choisi et tracer le dendrogramme correspondant
def tracer_dendrogram(model):
    plt.figure(figsize=(10, 10))
    plt.title('Clustering hiérarchique')
    plot_dendrogram(model, truncate_mode='level', p=2)
    plt.xlabel("Nombre de documents dans le noeud")
    plt.show()
    
# calcul pour chaque cluster l'effectif des différents journaux
def nb_journaux(predictions, liste_journaux):
    effectifs_journaux = {}

    for i in range(len(predictions)):
        effectifs_journaux.setdefault(predictions[i], {})
        effectifs_journaux[predictions[i]].setdefault(liste_journaux[i], 0)
        effectifs_journaux[predictions[i]][liste_journaux[i]] += 1
    return effectifs_journaux
    
# calculer et renvoyer un dictionnaire qui contient pour chaque cluster du model 
# le nombre de documents et
# la moyenne des distances au centroid du cluster correspondant 
def calcul_cluster_heterogeneity(model, X, predictions):
    X_dist = model.transform(X)**2
    rangement = {}
    for i in range(0, len(predictions)):
        rangement.setdefault(predictions[i], [])
        rangement[predictions[i]].append(np.min(X_dist[i]))
    distances = {}
    
    for k, v in rangement.items():
        distances.setdefault(k, {})
        distances[k]["nb_docs"] = len(v)
        distances[k]["distance"] = np.mean(v)
    return distances

# calcul des valeurs permettant de construire la courbe en coude
def elbow(X, plage_nb_clusters):
    distortions = [] 
    inertias = []
    for k in plage_nb_clusters: 
        kmeanModel = creer_model_KM(k, X)
        #dic_eff = effectifs_clusters(kmeanModel)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 
                        'euclidean'),axis=1)) / X.shape[0]) 
        inertias.append(kmeanModel.inertia_)
    return distortions, inertias

# affichage de la méthode du coude
def afficher_elbow(distortions, inertias, plage_nb_clusters):
    plt.plot(plage_nb_clusters, distortions, 'bx-') 
    plt.xlabel('Values of K') 
    plt.ylabel('Distortion') 
    plt.title('The Elbow Method using Distortion') 
    plt.show()

    plt.plot(plage_nb_clusters, inertias, 'bx-') 
    plt.xlabel('Values of K') 
    plt.ylabel('Inertia') 
    plt.title('The Elbow Method using Inertia') 
    plt.show()