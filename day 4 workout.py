import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df=pd.read_csv('D:\\Data_science\\newdtsets\\irisdataset.csv')
x=df.drop(columns=['Class']).values
y=df['Class'].values
label_encoding=[LabelEncoder() for _ in range(x.shape[1])]
for i in range(x.shape[1]):
    x[:,i]=label_encoding[i].fit_transform(x[:,i])
gnb=GaussianNB()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
gnb.fit(x_train,y_train)
y_pred=gnb.predict(x_test)
print(y_pred)
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)