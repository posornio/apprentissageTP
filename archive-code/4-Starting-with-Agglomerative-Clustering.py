import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics

from sklearn.metrics.pairwise import euclidean_distances


###################################################################
# Exemple : Agglomerative Clustering

def distance(a, b) :
    return np.sqrt(np.sum((a-b)**2))

"""
def separation(model) :
    centroids = model.cluster_centers_
    dists = euclidean_distances(centroids)
    min_sep = float('inf')
    max_sep = 0
    total = 0 
    n = 0
    for i in range(len(dists)) :
        for j in range(len(dists[i])) :
            if (dists[i][j] > max_sep ) :
                max_sep  = dists[i][j]
            if (0 < dists[i][j] < min_sep ) :
                min_sep  = dists[i][j]
            total += dists[i][j]
            n+=1

    moyenne_sep = total/n

    return moyenne_sep, min_sep, max_sep

def regroupement(k, model, datanp) :
    labels = model.labels_
    centroids = model.cluster_centers_
    max_reg = np.full(k, 0.0)
    min_reg = np.full(k,float('inf'))
    j=0
    total = np.full(k,0.0)
    n= np.full(k,0.0)
    m=0
    for l in centroids :
        for i in datanp :
            if (labels[j]==m) :
                dist = distance(i,l)
                if ( dist > max_reg[m]) :
                    max_reg[m] = dist
                if (dist < min_reg[m]) :
                    min_reg[m] = dist
                total[m] += dist
                n[m] +=1
            j+=1
        j = 0
        m+=1

    moyenne_reg = np.full(k,0.0)

    for i in range(len(moyenne_reg)) :
        moyenne_reg[i] = total[i]/n[i]

    return moyenne_reg, min_reg, max_reg
"""
def print_cluster(model, datanp) :
    f0 = datanp[:,0] # tous les élements de la première colonne
    f1 = datanp[:,1] # tous les éléments de la deuxième colonne
    labels = model.labels_
    plt.scatter(f0, f1, c=labels, s=8)
    plt.show()


def silhouette_par_k(datanp, methode) :
    maxi = float('-inf')
    k_max=0
    for k in range(2,12) :
        model = cluster.AgglomerativeClustering(linkage=methode, n_clusters=k)
        model.fit(datanp)
        labels = model.labels_
        silhouette = metrics.silhouette_score(datanp, labels)
        if (silhouette > maxi) :
            maxi = silhouette
            k_max = k
    return k_max


def evaluate(name) :
    print("--------------------------------------------------")
    print("EVALUATION : ", name)
    path = './artificial/'
    databrut = arff.loadarff(open(path+str(name), 'r'))
    datanp = np.array([[x[0],x[1]] for x in databrut[0]])
    methods = ['single', 'complete', 'average', 'ward']
    for methode in methods :
        now = time.time()
        print("Méthode : ", methode)
        k = silhouette_par_k(datanp, methode)
        print("Meilleur K = ", k)
        model = cluster.AgglomerativeClustering(linkage=methode, n_clusters=k)
        model.fit(datanp)
        #print_cluster(model, datanp)
        """
        moyenne_reg,min_reg,max_reg = regroupement(k, model, datanp)
        print("REGROUPEMENT : moy = ", moyenne_reg, "min = ", min_reg, "max = ",max_reg)
        moyenne_sep, min_sep, max_sep = separation(model)
        print("SEPARATION : moy = ", moyenne_sep, "min = ", min_sep, "max = ",max_sep)
        """
        print("Temps pour ", methode, " : ", time.time()-now)
        print("--------------------------------------------------")

"""
path = './artificial/'
name="xclara.arff"

#path_out = './fig/'
databrut = arff.loadarff(open(path+str(name), 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])
"""
"""
# PLOT datanp (en 2D) - / scatter plot
# Extraire chaque valeur de features pour en faire une liste
# EX : 
# - pour t1=t[:,0] --> [1, 3, 5, 7]
# - pour t2=t[:,1] --> [2, 4, 6, 8]
print("---------------------------------------")
print("Affichage données initiales            "+ str(name))
f0 = datanp[:,0] # tous les élements de la première colonne
f1 = datanp[:,1] # tous les éléments de la deuxième colonne

#plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales : "+ str(name))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-init.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()

"""
"""
### FIXER la distance
# 
tps1 = time.time()
seuil_dist=10
model = cluster.AgglomerativeClustering(distance_threshold=seuil_dist, linkage='average', n_clusters=None)
model = model.fit(datanp)
tps2 = time.time()
labels = model.labels_
# Nb iteration of this method
#iteration = model.n_iter_
k = model.n_clusters_
leaves=model.n_leaves_
plt.scatter(f0, f1, c=labels, s=8)
plt.title("Clustering agglomératif (average, distance_treshold= "+str(seuil_dist)+") "+str(name))
plt.show()
print("nb clusters =",k,", nb feuilles = ", leaves, " runtime = ", round((tps2 - tps1)*1000,2),"ms")

###
# FIXER le nombre de clusters
###
k=4
tps1 = time.time()
model = cluster.AgglomerativeClustering(linkage='average', n_clusters=k)
model = model.fit(datanp)
tps2 = time.time()
labels = model.labels_
# Nb iteration of this method
#iteration = model.n_iter_
kres = model.n_clusters_
leaves=model.n_leaves_
#print(labels)
#print(kres)

plt.scatter(f0, f1, c=labels, s=8)
plt.title("Clustering agglomératif (average, n_cluster= "+str(k)+") "+str(name))
plt.show()
print("nb clusters =",kres,", nb feuilles = ", leaves, " runtime = ", round((tps2 - tps1)*1000,2),"ms")

"""

###################################################
"""
#3.2
name = "xclara.arff"
evaluate(name)
"""

#3.3
#Single, average

name = "wingnut.arff"
evaluate(name)
name = "complex8.arff"
evaluate(name)
#name = "elly-2d10c13s.arff"
#evaluate(name)

