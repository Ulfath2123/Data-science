# Normal distribution using dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
data=pd.read_csv('D:\\Data_science\\newdtsets\\Salary_Data.csv')
values=data.YearsExperience
mu,sigma=norm.fit(values)
print(mu,sigma)
plt.hist(values,bins=30,density=True,alpha=0.6,color='g')
xmin,xmax=plt.xlim()
print(xmin,xmax)
x=np.linspace(xmin,xmax,100)
p=norm.pdf(x,mu,sigma)
plt.plot(x,p,'g',linewidth=2)
plt.show()