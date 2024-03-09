
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
#Giving input
data=pd.read_csv('D:\\Data_science\\newdtsets\\adult.csv')
data.dropna(inplace=True)
x=data.drop(columns=['workclass','education','marital-status','occupation','relationship','race','sex','nativeCountry','income'])
#LAbel encoding.
label_encoder=LabelEncoder()
label_encoder.fit(data['income'])
data['income']=label_encoder.transform(data['income'])
y=data['income']
#Splitting train and test data.
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#Performing model
rf=RandomForestClassifier()
rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
#Accuracy score
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)
#plotting
plt.contourf(x_test,y_test,y_pred,cmap=plt.cm.RdYlBu,alpha=0.8)
plt.scatter(x[:,0],x[:,1],c=y,cmap=plt.cm.RdYlBu)
plt.xlabel('features')
plt.ylabel('target')
plt.title('Random forest Decision Boundaries')
plt.show()


