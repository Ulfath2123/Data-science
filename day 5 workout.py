#Normal deviation.
import numpy as np
data=np.array([10,12,15,18,20,22,25,28,30])
mean=np.mean(data)
std_dev=np.std(data)
value=22
z_score=(value-mean)/std_dev
print("z-score:",z_score)

#z-score
import pandas as pd
from scipy.stats import zscore
data=pd.read_csv('D:\\Data_science\\newdtsets\\adult.csv')
data["zscore"]=zscore(data["age"])
print(data)

#confidence interval
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
np.random.seed(42)
data=np.random.normal(loc=10,scale=2,size=100)
confidence_level=0.95
mean=np.mean(data)
std_err=stats.sem(data)
confidence_interval=stats.norm.interval(confidence_level,loc=mean,scale=std_err)
plt.hist(data,bins=20,alpha=0.5,color='skyblue',edgecolor='black')
plt.axvline(mean,color='red',linestyle='dashed',linewidth=1,label='mean')
plt.axvline(confidence_interval[0],color='green',linestyle='dashed',linewidth=1,label='confidence interval')
plt.axvline(confidence_interval[1],color='green',linestyle='dashed',linewidth=1)
plt.legend()
plt.xlabel('value')
plt.ylabel('frequency')
plt.title('histogram of sample data with 95% confidence interval')
