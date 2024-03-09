import sys
sys.path.append('C:\\Users\\ACER\\Appdata\\Locals\\Programs\\Python\\Python38\\Lib\\site-packages')
import pandas as pd
import pymysql as mysql
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error
conn=mysql.connect(host='localhost', port=3306, user='root', passwd='ulfath123', db='sales')
query='select * from salesreport'
data=pd.read_sql(query,conn)
conn.close()
#plotting data
plt.figure(figsize=(12,6))
plt.plot(data)
plt.title('Sales report')
plt.xlabel('date')
plt.ylabel('sales')
plt.grid(True)
plt.show()
#Autocorrelation
autocorrelation_plot(data)
plt.show()
#Splitting into train and test datas
train_size=int(len(data)*0.8)
train_data,test_data=data[:train_size],data[train_size:]
#Fitting model
order=(2,1,2)
model=ARIMA(train_data,order=order)
model_fit=model.fit(disp=0)
print(model_fit.summary())
#Forecasting
forecast=model_fit.forecast(steps=len(test_data))
#Mean squared error.
mse=mean_squared_error(test_data,forecast)
print(mse)
#plotting original data and forecasting data.
plt.figure(figsize=(12,6))
plt.plot(test_data.index,test_data,label='Test data')
plt.plot(test_data.index,forecast,color='red',label='Forecast')
plt.legend()
plt.title('Actual vs Forecasted')
plt.xlabel('date')
plt.ylabel('sales')
plt.grid(True)
plt.show()
