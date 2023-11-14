
import os
os.environ["OMP_NUM_THREADS"] = '4'

import numpy as np
import matplotlib.pyplot as plt

from scipy.io import arff


path = './artificial/'
name="square1.arff"
#path_out = './fig/'

databrut = arff.loadarff(open(path+str(name), 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])

print("---------------------------------------")
print("Récupérer les données initiales            "+ str(name))
f0 = datanp[:,0] # tous les élements de la première colonne
f1 = datanp[:,1] # tous les éléments de la deuxième colonne
#print(f0)
#print(f1)

plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, s=10) #s pour régler l'affichage des points - s=0.01
plt.title("Donnees initiales : "+ str(name))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-init.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()