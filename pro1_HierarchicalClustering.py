from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram,linkage
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import normalize
data=pd.read_csv('D:\\Data_science\\newdtsets\\adult.csv')
data.dropna(inplace=True)
#Label encoding from object to numerical.
label_encoder=LabelEncoder()
objlist=data.select_dtypes(include='object').columns
for feat in objlist:
    data[feat]=label_encoder.fit_transform(data[feat].astype(str))
print(data.info())
#Normalizing data
data_scaled=normalize(data)
data_scaled=pd.DataFrame(data_scaled,columns=data.columns)
print(data_scaled.head())
#plotting dendrogram to decide the no.of.clusters
plt.figure(figsize=(10,7))
plt.title('Dendrograms')
dendrogram(linkage(data_scaled,method='ward'))
plt.show()
#plot a threshold
plt.figure(figsize=(10,7))
plt.title('Dendrogram with threshold')
dendrogram(linkage(data_scaled,method='ward'))
plt.axhline(y=6,color='r',linestyle='--')
plt.show()
cluster=AgglomerativeClustering(n_clusters=2)
cluster.fit_predict(data_scaled)
#Scatter plot to show clustering
plt.figure(figsize=(10,7))
plt.scatter(data_scaled[])
plt.show()
print('hello')
