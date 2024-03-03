import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
np.random.seed(42)
time_index=pd.date_range(start='2022-01-01',end='2022-12-31',freq='D')
values=np.random.randn(len(time_index))
ts_data=pd.Series(values,index=time_index)
print(ts_data)
plt.figure(figsize=(10,6))
plt.plot(ts_data)
plt.grid(True)
plt.show()
train_size=int(len(ts_data)*0.8)
train_data,test_data=ts_data[:train_size],ts_data[train_size]
order=(2,1,1)
model=ARIMA(train_data,order=order)
model_fit=model.fit()
forecast_steps=len(test_data)
forecast=model_fit.forecast(steps=forecast_steps)
plt.figure(figsize=(10,6))
plt.plot(test_data.index,test_data,label='Actual')
plt.plot(test_data.index,forecast,color='red',label='Forecast')
plt.title("Arima Forecast")
plt.xlabel('Data')
plt.ylabel('Value')
plt.legend()
plt.grid()
plt.show()