import pandas as pd
import numpy as np
adult_df=pd.read_csv('D:\\Data_science\\newdtsets\\adult.csv')
#print(adult_df)
print(adult_df.head())
print(adult_df.tail())
print(adult_df.age)
print(type(adult_df))
print(adult_df.loc[0].index)
print(adult_df.loc[1])
print(adult_df.age.index)
print(adult_df.set_index(np.arange(10000,42561),inplace=True))
print
