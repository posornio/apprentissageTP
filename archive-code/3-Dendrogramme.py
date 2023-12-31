import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from scipy.cluster.hierarchy import dendrogram

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
    dendrogram(linkage_matrix) #, **kwargs)

def saveFigs(method):
    path = './artificial/'
    names = ["3-spiral.arff"]
    for name in names:
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
        plot_dendrogram(model) 
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        fig.savefig("figsDendrogramme-"+method+"-dendrogram-"+str(name)+".jpg",bbox_inches='tight', pad_inches=0.1)
        print("fig saved : "+method+"-dendrogram-"+str(name)+".jpg")

linkages = ["single","complete","average","ward"]

for method in linkages:
    saveFigs(method)




#Brouillon :

"""
tps1 = time.time()
seuil_dist = 10
model = cluster.AgglomerativeClustering(distance_threshold=seuil_dist, linkage='single', n_clusters=None)
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


k=2
tps1 = time.time()
model = cluster.AgglomerativeClustering(linkage='single', n_clusters=k)
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
