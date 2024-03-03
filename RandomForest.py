import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
iris=load_iris()
print(iris)
x=iris.data[:,:2]
y=iris.target
clf=RandomForestClassifier(n_estimators=100,random_state=42)
clf.fit(x,y)
x_min,x_max=x[:,0].min()-1,x[:,0].max()+1
y_min,y_max=x[:,1].min()-1,x[:,1].max()+1
xx,yy=np.meshgrid(np.arange(x_min,x_max,0.02),np.arange(y_min,y_max,0.02))
z=clf.predict(np.c_[xx.ravel(),yy.ravel()])
z=z.reshape(xx.shape)
plt.contourf(xx,yy,z,cmap=plt.cm.RdYlBu,alpha=0.8)
plt.scatter(x[:,0],x[:,1],c=y,cmap=plt.cm.RdYlBu)
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.title('Random forest Decision Boundaries')
plt.show()
