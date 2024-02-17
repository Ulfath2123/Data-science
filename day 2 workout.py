import pandas as pd
import numpy as np
data=pd.read_csv('D:\\Data_science\\newdtsets\\Salary_Data.csv')
#print(data.info())
expunique=data.YearsExperience.unique()
print(expunique)
expcount=data.YearsExperience.value_counts()
print(expcount)
#data visualization
import matplotlib.pyplot as plt
data.YearsExperience.plot.hist()
pl.show()
