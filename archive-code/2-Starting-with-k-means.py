import os
os.environ["OMP_NUM_THREADS"] = '4'

import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
import time


def distance(a, b) :
    return np.sqrt(np.sum((a-b)**2))


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


def inertie_intra_cluster (points,centre):
    dist_total = 0
    for i in range(len(points)):
        dist = distance(points[i],centre)
        dist = dist**2
        dist_total += dist
    return dist_total
        
def couper_cluster (points,labels,k):
    ret = []
    for i in range (max(labels)+1):
        ret.append([])
    for index,point in enumerate(points):
        lab_point = labels[index]
        ret[lab_point].append(point)
    return ret



def inertie_cluster (datanp,model, k):
    ret = np.full(k, 0.0)
    clusters = couper_cluster(datanp,model.labels_,k)
    for index,cluster in enumerate(clusters):
        ret[index] = inertie_intra_cluster(cluster,model.cluster_centers_[index])
    return ret
    
def inertie_par_k(datanp) :
    now = time.time()
    inerties = np.full(11,0.0)
    for k in range(1,12) :
        model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
        model.fit(datanp)
        # informations sur le clustering obtenu
        inertie = model.inertia_
        inerties[k-1] = inertie
    
    plt.plot(range(1,12),inerties)
    plt.title("Inertie en fonction du nombre de clusters")
    plt.show()
    
    print("runtime inertie = ", (time.time() - now),"s")

def silhouette_par_k(datanp) :
    now = time.time()
    silhouettes = []
    max = 0 
    total_time = 0
    for k in range(2,42) :
        t1 = time.time()
        model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
        model.fit(datanp)
        total_time +=  time.time()-t1
        
        # informations sur le clustering obtenu
        labels = model.labels_
        silhouette = metrics.silhouette_score(datanp, labels)
        silhouettes.append(silhouette)
        if (silhouette > max) :
            max = silhouette
            k_max = k
        
    print("TEMPS POUR K-MEANS : ", total_time)
    
    plt.plot(range(2,42),silhouettes)
    plt.title("Silhouette en fonction du nombre de clusters")
    plt.show()
    
    #print("runtime silhouette = ", (time.time() - now),"s")
    return k_max

def silhouette_par_k_minibatch(datanp) :
    silhouettes = []
    max = 0 
    total_time = 0 
    for k in range(2,12) :
        t1 = time.time()
        model = cluster.MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=1)
        model.fit(datanp)
        total_time +=  time.time()-t1
        
        # informations sur le clustering obtenu
        labels = model.labels_
        silhouette = metrics.silhouette_score(datanp, labels)
        silhouettes.append(silhouette)
        if (silhouette > max) :
            max = silhouette
            k_max = k
        
    print("TEMPS POUR MINIBATCH : ", total_time)
    """"
    plt.plot(range(2,12),silhouettes)
    plt.title("Silhouette en fonction du nombre de clusters")
    plt.show()
    """
    return k_max

def init_data(name) :
    path = './artificial/'
    databrut = arff.loadarff(open(path+str(name), 'r'))
    datanp = np.array([[x[0],x[1]] for x in databrut[0]])
    return datanp


def trainings_kmeans(k, datanp) :
    model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
    model.fit(datanp)
    return model
    """
    labels = model.labels_
    # informations sur le clustering obtenu
    iteration = model.n_iter_
    inertie = model.inertia_
    centroids = model.cluster_centers_
    """
def trainings_minibatch(k, datanp) :
    model= cluster.MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=1)
    model.fit(datanp)
    return model


def showcluster(datanp, model) :
    f0 = datanp[:,0] # tous les élements de la première colonne
    f1 = datanp[:,1] # tous les éléments de la deuxième colonne
    centroids = model.cluster_centers_
    labels = model.labels_
    plt.figure(figsize=(6, 6))
    plt.scatter(f0, f1, c=labels, s=8)
    plt.scatter(centroids[:, 0],centroids[:, 1], marker="x", s=50, linewidths=3, color="red")
    #plt.title("Données après clustering : "+ str(name) + " - Nb clusters ="+ str(k))
    #plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-cluster.jpg",bbox_inches='tight', pad_inches=0.1)
    plt.show()

#La fonction evaluate nous permet d'évaluer k-means sur les différents datasets
def evaluate(nom) :
    datanp = init_data(nom)
    k = silhouette_par_k(datanp)
    #print ("k = ", k)
    model= trainings_kmeans(k, datanp)
    moyenne_reg,min_reg,max_reg = regroupement(k, model, datanp)
    #print("REGROUPEMENT : moy = ", moyenne_reg, "min = ", min_reg, "max = ",max_reg)

    moyenne_sep, min_sep, max_sep = separation(model)
    #print("SEPARATION : moy = ", moyenne_sep, "min = ", min_sep, "max = ",max_sep)
    showcluster(datanp, model)


#La fonction evaluate nous permet d'évaluer minibatch sur les différents datasets
def evaluate_minibatch(nom) :
    datanp = init_data(nom)
    k = silhouette_par_k_minibatch(datanp)
    #print ("k = ", k)
    model= trainings_minibatch(k, datanp)
    moyenne_reg,min_reg,max_reg = regroupement(k, model, datanp)
    #print("REGROUPEMENT : moy = ", moyenne_reg, "min = ", min_reg, "max = ",max_reg)

    moyenne_sep, min_sep, max_sep = separation(model)
    #print("SEPARATION : moy = ", moyenne_sep, "min = ", min_sep, "max = ",max_sep)
    showcluster(datanp, model)
    


"""
plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, c=labels, s=8)
plt.scatter(centroids[:, 0],centroids[:, 1], marker="x", s=50, linewidths=3, color="red")
plt.title("Données après clustering : "+ str(name) + " - Nb clusters ="+ str(k))
plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-cluster.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()
"""
#print("nb clusters =",k,", nb iter =",iteration, ", inertie = ",inertie, ", runtime = ", round((tps2 - tps1)*1000,2),"ms")
#print("labels", labels)

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_samples

#dists = euclidean_distances(centroids)
#print("Euclidiean distance : \n", dists)



"""
print("REGROUPEMENT :")
print("Min regroupement = ",min_reg )
print("Max regroupement =", max_reg )
print("Moyenne regroupement =", moyenne_reg)

print("SEPARATION :")
print("Min separation = ",min_sep )
print("Max separation =", max_sep )
print("Moyenne separation =", moyenne_sep )
"""

"""
print("Inertie moyenne : ", inertie)

print("Inertie moyenne par cluster ",(inertie_cluster(datanp,labels,k)))
print(np.sum(inertie_cluster(datanp,labels,k)))
"""
"""
#2.3
# Analyse les resultats:  
datanp = init_data("2d-4c-no9.arff")
inertie_par_k(datanp)
k = silhouette_par_k(datanp)
print("Meilleur k pour silhouette = ", k)
model= trainings_kmeans(k, datanp)
moyenne_reg,min_reg,max_reg = regroupement(k, model, datanp)
print("Score de Regroupement : moy = ", moyenne_reg, "min = ", min_reg, "max = ",max_reg)

moyenne_sep, min_sep, max_sep = separation(model)
print("SEPARATION : moy = ", moyenne_sep, "min = ", min_sep, "max = ",max_sep)

#On remarque que silhouette est beaucoup plus long, mais permet de sortir automatiquement le résulat, car il suffit de trouver le max.

"""



#2.4

#Un pour lequel ça marche : (Le deuxième est celui du 2.3)
nom = "2d-4c.arff"
evaluate(nom)

nom = "twodiamonds.arff"
evaluate(nom)

nom = "fourty.arff"
evaluate(nom)

#1 pour lequel ça marche pas :
nom = "3-spiral.arff"
evaluate(nom)

nom = "disk-4600n.arff"
evaluate(nom)


#Un intermediaire :
nom = "2sp2glob.arff"
evaluate(nom)


#2.5
"""
nom = "2d-4c.arff"
evaluate_minibatch(nom)
evaluate(nom)
#c'était pareil
nom = "3-spiral.arff"
evaluate_minibatch(nom)
evaluate(nom)
#résultat différent mais toujours mauvais
nom = "2sp2glob.arff"
evaluate_minibatch(nom)
evaluate(nom)
#Même résultat
nom = "birch-rg1.arff"
evaluate_minibatch(nom)
evaluate(nom)
"""