from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
x=np.random.randn(100,2)
print(x)
kmeans=KMeans(n_clusters=3)
kmeans.fit(x)
centroids=kmeans.cluster_centers_
labels=kmeans.labels_
plt.scatter(x[:0],x[:1],c=labels,cmap='viridis',alpha=0.5,edgecolor='k')
plt.scatter(centroids[:0],centroids[:1],marker='x',s=300,c='red',label='centroids')
plt.show()