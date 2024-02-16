import pandas as pd
import numpy as np
data=pd.read_csv('D:\\Data_science\\newdtsets\\Salary_Data.csv')

print(data)
print(data.head())
print(data.tail())
print(data.Salary)
print(type(data))
print(data.loc[0].index)
print(data.loc[1])
print(data.Salary.index)

#To get a particular data
print(data.iloc[0].loc['Salary'])
print(data.Salary.loc[5])
print(data['Salary'].iloc[2])
print(data.at[2,'Salary'])

#To get a particular data row wise.
row_series=data.loc[2]
print(row_series.loc['Salary'])

print(row_series.iloc[1])
print(row_series['Salary'])
print(row_series.Salary)
#To get a particular data column wise
column_series=data.Salary
print(column_series.loc[2])
print(column_series.iloc[2])
print(column_series[2])

#Selecting column using location
selected_columns=data.loc[0:10,'YearsExperience':'Salary']
print('Selected column using location: ')
print(selected_columns)

#sorting and selecting rows
sorted_df=data.sort_values('YearsExperience').reset_index()
selected_rows=sorted_df.iloc[1::2]
print('Selected rows after sorting: ')
print(selected_rows)

#Mean
mean=np.mean(data['YearsExperience'])
print(mean)
