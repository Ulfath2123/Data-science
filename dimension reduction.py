from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
iris=load_iris()
x=iris.data
y=iris.target
target_names=iris.target_names
x=StandardScaler().fit_transform(x)
pca=PCA(n_components=3)
x_pca=pca.fit_transform(x)
plt.figure()
colors=['navy','turquoise','darkorange']
lw=2
for color,i,target_name in zip(colors,[0,1,2],target_names):
    plt.scatter(x_pca[y==i,0],x_pca[y==i,1],color=color,alpha=0.8,lw=lw,label=target_names)
plt.legend(loc='best',shadow=False,scatterpoints=1)
plt.show()