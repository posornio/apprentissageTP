import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from scipy.cluster.hierarchy import dendrogram


from sklearn.metrics.pairwise import euclidean_distances


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



def distance(a, b) :
    return np.sqrt(np.sum((a-b)**2))

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
    dendrogram(linkage_matrix)


def saveDendo(name,method):
    path = './artificial/'
    path_out = './fig/'
    databrut = arff.loadarff(open(path+str(name), 'r'))
    datanp = np.array([[x[0],x[1]] for x in databrut[0]])
    model = cluster.AgglomerativeClustering(distance_threshold=0, linkage= method, n_clusters=None)

    model = model.fit(datanp)
    f0 = datanp[:,0]
    f1 = datanp[:,1] 
    labels = model.labels_

    fig = plt.figure(figsize=(12, 12))
    plt.scatter(f0, f1, s=8)
    plt.title("Hierarchical Clustering Dendrogram for "+str(name) + " with "+method+" linkage")
    plot_dendrogram(model) 
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    
    fig.savefig("figsDendrogramme-"+method+"-dendrogram-"+str(name)+".jpg",bbox_inches='tight', pad_inches=0.1)
    print("fig saved : "+method+"-dendrogram-"+str(name)+".jpg")

linkages = ["single","complete","average","ward"]

#La fonction evaluate nous permet d'évaluer les méthodes de clustering sur les différents datasets

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

