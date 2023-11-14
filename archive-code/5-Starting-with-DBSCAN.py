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

#La fonction Neigh permet de calculer la distance moyenne entre un point et ses k plus proches voisins
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


#La fonction dbscan permet de faire du clustering avec l'algorithme DBSCAN et d´afficher le résultat
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


#La fonction HDBscan permet de faire du clustering avec l'algorithme HDBSCAN et d´afficher le résultat
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