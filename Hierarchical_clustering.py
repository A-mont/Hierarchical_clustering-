# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 15:27:02 2021

@author: monte
"""
import numpy as np 
import pandas as pd
from scipy import ndimage 
from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 
from matplotlib import pyplot as plt 
from sklearn import manifold, datasets 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.datasets.samples_generator import make_blobs 


#CARGAMOS EL DATA SET
filename = 'cars_clus.csv'

#Read csv
pdf = pd.read_csv(filename)
print ("Forma del set de datos: ", pdf.shape)

pdf.head(5)

###LIMPIEZA DE DATOS
#limpiemos el set de datos eliminando filas que tienen valores nulos:
print ("Shape of dataset before cleaning: ", pdf.size)
pdf[[ 'sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']] = pdf[['sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')
pdf = pdf.dropna()
pdf = pdf.reset_index(drop=True)
print ("Forma del dataset luego de la limpieza: ", pdf.size)
pdf.head(5)

#Elijamos nuestro set en cuestión:
featureset = pdf[['engine_s',  'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]


#Ahora podemos normalizar el set. MinMaxScaler transforma poniendo en escala a un rango.
from sklearn.preprocessing import MinMaxScaler
x = featureset.values #returns a numpy array
min_max_scaler = MinMaxScaler()
feature_mtx = min_max_scaler.fit_transform(x)
feature_mtx [0:5]

#En esta parte, usaremos el paquete Scipy para agrupar el set de datos:
import scipy
leng = feature_mtx.shape[0]
D = scipy.zeros([leng,leng])
for i in range(leng):
    for j in range(leng):
        D[i,j] = scipy.spatial.distance.euclidean(feature_mtx[i], feature_mtx[j])
        
# Los métodos siguientes los soporta Scipy para calcular la distancia entre los recientementes formados clusters y para cada: - simple - completo - promedio - ponderado - centroide
#Usaremos completo para nuestro caso, pero si deseas puedes cambiar para ver los diferentes resultados.
import pylab
import scipy.cluster.hierarchy
Z = hierarchy.linkage(D, 'complete')

#Escencialmente, el clustering jerárquico no necesita de un número específico de clusters. Sin embargo, algunas aplicaciones que queremos una partición de clusters disjuntos como si fueran clustering tradicional. Por lo tanto podrás usar una linea en el medio:
from scipy.cluster.hierarchy import fcluster
max_d = 3
clusters = fcluster(Z, max_d, criterion='distance')
clusters

#También, podrás determinar la cantidad de clusters directamente:
from scipy.cluster.hierarchy import fcluster
k = 5
clusters = fcluster(Z, k, criterion='maxclust')
clusters

#Ahora, se trazará el dendograma:
fig = pylab.figure(figsize=(18,50))
def llf(id):
    return '[%s %s %s]' % (pdf['manufact'][id], pdf['model'][id], int(float(pdf['type'][id])) )
    
dendro = hierarchy.dendrogram(Z,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =12, orientation = 'right')

#Volvamos a construir el cluster, pero esta vez usando el paquete scikit-learn:

dist_matrix = distance_matrix(feature_mtx,feature_mtx) 
print(dist_matrix)

#Ahora, podemos usar la función 'AgglomerativeClustering' de la librería scikit-learn para agrupar el set de datos. Esta función hace un clustering jerárquico por medio de un enfoque de abajo hacia arriba (bottom up). El criterio de enlace determina la métrica utilizada para la estrategia de unificación:

agglom = AgglomerativeClustering(n_clusters = 6, linkage = 'complete')
agglom.fit(feature_mtx)
agglom.labels_

#Podemos agregar un nuevo campo a nuestro marco de datos para mostrar el cluster de cada fila:
pdf['cluster_'] = agglom.labels_
pdf.head()


import matplotlib.cm as cm
n_clusters = max(agglom.labels_)+1
colors = cm.rainbow(np.linspace(0, 1, n_clusters))
cluster_labels = list(range(0, n_clusters))

# Crear una figura de 6 pulgadas (aprox 15cm) por 4 pulgadas (aprox 10cm).
plt.figure(figsize=(16,14))

for color, label in zip(colors, cluster_labels):
    subset = pdf[pdf.cluster_ == label]
    for i in subset.index:
            plt.text(subset.horsepow[i], subset.mpg[i],str(subset['model'][i]), rotation=25) 
    plt.scatter(subset.horsepow, subset.mpg, s= subset.price*10, c=color, label='cluster'+str(label),alpha=0.5)
#    plt.scatter(subset.horsepow, subset.mpg)
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')

#Primero contamos la cantidad de casos de cada grupo:
pdf.groupby(['cluster_','type'])['cluster_'].count()

#Ahora, podemos examinar a las características de cada cluster:
agg_cars = pdf.groupby(['cluster_','type'])['horsepow','engine_s','mpg','price'].mean()
agg_cars

plt.figure(figsize=(16,10))
for color, label in zip(colors, cluster_labels):
    subset = agg_cars.loc[(label,),]
    for i in subset.index:
        plt.text(subset.loc[i][0]+5, subset.loc[i][2], 'type='+str(int(i)) + ', price='+str(int(subset.loc[i][3]))+'k')
    plt.scatter(subset.horsepow, subset.mpg, s=subset.price*20, c=color, label='cluster'+str(label))
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')