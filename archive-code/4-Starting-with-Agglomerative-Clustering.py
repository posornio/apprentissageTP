import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from scipy.cluster.hierarchy import dendrogram


from sklearn.metrics.pairwise import euclidean_distances


###################################################################
# Exemple : Agglomerative Clustering


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
    centroids = model.
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
def print_cluster(model, datanp,method,name) :
    f0 = datanp[:,0] # tous les élements de la première colonne
    f1 = datanp[:,1] # tous les éléments de la deuxième colonne
    labels = model.labels_
    plt.scatter(f0, f1, c=labels, s=8)
    plt.savefig("cluster--"+method+"--"+str(name)+".jpg",bbox_inches='tight', pad_inches=0.1)


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

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix) #, **kwargs)


def saveDendo(name,method):
    path = './artificial/'
    path_out = './fig/'
    databrut = arff.loadarff(open(path+str(name), 'r'))
    datanp = np.array([[x[0],x[1]] for x in databrut[0]])
    model = cluster.AgglomerativeClustering(distance_threshold=0, linkage= method, n_clusters=None)

    model = model.fit(datanp)
    f0 = datanp[:,0] # tous les élements de la première colonne
    f1 = datanp[:,1] # tous les éléments de la deuxième colonne
    labels = model.labels_

    fig = plt.figure(figsize=(12, 12))
    plt.scatter(f0, f1, s=8)
    plt.title("Hierarchical Clustering Dendrogram for "+str(name) + " with "+method+" linkage")
    # plot the top p levels of the dendrogram
    plot_dendrogram(model) #, truncate_mode="level", p=5)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    
    fig.savefig("figsDendrogramme-"+method+"-dendrogram-"+str(name)+".jpg",bbox_inches='tight', pad_inches=0.1)
    print("fig saved : "+method+"-dendrogram-"+str(name)+".jpg")

linkages = ["single","complete","average","ward"]



def evaluate(name) :
    print("--------------------------------------------------")
    print("EVALUATION : ", name)
    path = './artificial/'
    databrut = arff.loadarff(open(path+str(name), 'r'))
    datanp = np.array([[x[0],x[1]] for x in databrut[0]])
    methods = [ 'single']
    for methode in methods :
        now = time.time()
        print("Méthode : ", methode)
        k = silhouette_par_k(datanp, methode)
        print("Meilleur K = ", k)
        model = cluster.AgglomerativeClustering(linkage=methode, n_clusters=k)
        model.fit(datanp)
        #saveDendo(name,methode)
        #print_cluster(model, datanp, name, methode)
        moyenne_reg,min_reg,max_reg = regroupement(k, model, datanp)
        print("Score de regroupement : moy = ", moyenne_reg, "min = ", min_reg, "max = ",max_reg)

        moyenne_sep, min_sep, max_sep = separation(model)
        print("Score de séparation  : moy = ", moyenne_sep, "min = ", min_sep, "max = ",max_sep)
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
name = "xclara.arff"
#name = "cluto-t5-8k.arff"
evaluate(name)
#name = "complex8.arff"
#evaluate(name)
#name = "elly-2d10c13s.arff"
#evaluate(name)

