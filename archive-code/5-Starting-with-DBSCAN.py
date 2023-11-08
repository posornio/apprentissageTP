import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
import hdbscan
import pandas as pd

##################################################################
# Exemple : DBSCAN Clustering

"""
path = './artificial/'
name="3-spiral.arff"

#path_out = './fig/'
databrut = arff.loadarff(open(path+str(name), 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])

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


# Run DBSCAN clustering method 
# for a given number of parameters eps and min_samples
# 
print("------------------------------------------------------")
print("Appel DBSCAN (1) ... ")
tps1 = time.time()
epsilon=2 #2  # 4
min_pts= 5 #10   # 10
model = cluster.DBSCAN(eps=epsilon, min_samples=min_pts)
model.fit(datanp)
tps2 = time.time()
labels = model.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print('Number of clusters: %d' % n_clusters)
print('Number of noise points: %d' % n_noise)

plt.scatter(f0, f1, c=labels, s=8)
plt.title("Données après clustering DBSCAN (1) - Epislon= "+str(epsilon)+" MinPts= "+str(min_pts))
plt.show()


####################################################
# Standardisation des donnees

scaler = preprocessing.StandardScaler().fit(datanp)
data_scaled = scaler.transform(datanp)
print("Affichage données standardisées            ")
f0_scaled = data_scaled[:,0] # tous les élements de la première colonne
f1_scaled = data_scaled[:,1] # tous les éléments de la deuxième colonne

#plt.figure(figsize=(10, 10))
plt.scatter(f0_scaled, f1_scaled, s=8)
plt.title("Donnees standardisées")
plt.show()


print("------------------------------------------------------")
print("Appel DBSCAN (2) sur données standardisees ... ")
tps1 = time.time()
epsilon=0.05 #0.05
min_pts=5 # 10
model = cluster.DBSCAN(eps=epsilon, min_samples=min_pts)
model.fit(data_scaled)

tps2 = time.time()
labels = model.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print('Number of clusters: %d' % n_clusters)
print('Number of noise points: %d' % n_noise)

plt.scatter(f0_scaled, f1_scaled, c=labels, s=8)
plt.title("Données après clustering DBSCAN (2) - Epislon= "+str(epsilon)+" MinPts= "+str(min_pts))
plt.show()


print("------------------------------------------------------")
print("Appel DBSCAN (3) sur données standardisees ... ")
tps1 = time.time()
epsilon=0.15 #0.05
min_pts=5 # 10
model = cluster.DBSCAN(eps=epsilon, min_samples=min_pts)
model.fit(data_scaled)

tps2 = time.time()
labels = model.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print('Number of clusters: %d' % n_clusters)
print('Number of noise points: %d' % n_noise)

plt.scatter(f0_scaled, f1_scaled, c=labels, s=8)
plt.title("Données après clustering DBSCAN (3) - Epislon= "+str(epsilon)+" MinPts= "+str(min_pts))
plt.show()
"""

def Neigh(name):
    
    path = './artificial/'

    #path_out = './fig/'
    databrut = arff.loadarff(open(path+str(name), 'r'))
    datanp = np.array([[x[0],x[1]] for x in databrut[0]])     #Distances aux k plus proches voisins
# Donnees dans X
    k = 2
    neigh = NearestNeighbors ( n_neighbors = k )
    neigh.fit( datanp )
    distances , indices = neigh.kneighbors(datanp)
    # distance moyenne sur les k plus proches voisins
    # en retirant le point " origine "
    newDistances = np.asarray( [ np.mean( distances[ i ][ 1 : ] ) for i in range (0 , distances.shape[0]) ] )
    # trier par ordre croissant
    distancetrie = np.sort( newDistances )
    print("distance trie  : ", distancetrie)
    plt.title( " Plus proches voisins " + str( k ) )
    plt.plot( distancetrie ) 
    plt.show()

def dbscan(name, eps, nb_point) :
    path = './artificial/'
    databrut = arff.loadarff(open(path+str(name), 'r'))
    datanp = np.array([[x[0],x[1]] for x in databrut[0]])
    scaler = preprocessing.StandardScaler().fit(datanp)
    data_scaled = scaler.transform(datanp)
    f0_scaled = data_scaled[:,0] # tous les élements de la première colonne
    f1_scaled = data_scaled[:,1] # tous les éléments de la deuxième colonne
    tps1 = time.time()
    model = cluster.DBSCAN(eps=eps, min_samples=nb_point)
    model.fit(data_scaled)

    tps2 = time.time()
    print("Execution time DBSCAN : ", tps2-tps1)
    labels = model.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print('Number of clusters: %d' % n_clusters)
    print('Number of noise points: %d' % n_noise)

    plt.scatter(f0_scaled, f1_scaled, c=labels, s=8)
    plt.title("Données après clustering DBSCAN (3) - Epislon= "+str(eps)+" MinPts= "+str(nb_point))
    plt.show()



def HDBscan(name, min_cluster_size):
    path = './artificial/'
    databrut = arff.loadarff(open(path + str(name), 'r'))
    datanp = np.array([list(x)[:2] for x in databrut[0]])
    scaler = preprocessing.StandardScaler().fit(datanp)
    data_scaled = scaler.transform(datanp)

    tps1 = time.time()
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(data_scaled)
    tps2 = time.time()

    print("Execution time HDBSCAN : ", tps2 - tps1)
    
    # Afficher le nombre de clusters (clusters différents de -1, qui représente le bruit)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print('Number of clusters: %d' % n_clusters)
    print('Number of noise points: %d' % n_noise)

    plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=labels, s=8)
    plt.title("Données après clustering HDBSCAN - Min Cluster Size = " + str(min_cluster_size))
    plt.show()
    return n_noise

"""
#4.2 
name = "3-spiral.arff"
name ="xclara.arff" 
Neigh(name)
path = './artificial/'
databrut = arff.loadarff(open(path + str(name), 'r'))
datanp = np.array([list(x)[:2] for x in databrut[0]])
X = pd.DataFrame(datanp, columns = ['abscisse','ordonnée'])
y_pred = cluster.DBSCAN(eps = 1.9, min_samples=2).fit_predict(datanp)
plt.title("DBSCAN sans preprocessing")
plt.scatter(X['abscisse'],X['ordonnée'],c = y_pred)
plt.show()
"""
"""
#4.4
#fonctionne : 

name = "dense-disk-3000.arff"
name = "donutcurves.arff"
name = "diamond9.arff"

#Fonctionne pas : 

name = "dense-disk-3000.arff"
#Changement de densité pose problème
name = "jain.arff"
#name = "elly-2d10c13s.arff"


dbscan(name, 0.1, 11)
"""



#4.4
def find_min_sample() : 
    k_min = 60000000
    r = 0
    for i in range(2,20) :
        k = HDBscan(name, i)
        if (k<k_min) :
            r = i
    print("Le meilleur k est : ", r)
name ="xclara.arff"

#Fonctionne toujours pas trop 
name = "jain.arff"

name = "dense-disk-3000.arff"
name = "cure-t2-4k.arff"
name = "birch-rg1.arff"
dbscan(name, 0.08, 300)
#HDBscan(name, 50)




#Augmenter le nombre depoint augmente le nombre de cluster, et diminuer l'epsilon augmente aussi le nombre de cluster

#epsilon = 0.15 et min_pts = 5 donne le meilleur résultat pour xclara.arff

#DBSCAN probleme avec grand nombre de points + temps de calcul important st900.arff, zelnik4.arff car bcp de outliers 