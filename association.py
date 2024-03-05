import mlxtend.frequent_patterns import apriori,association_rules
import pandas as pd
import matplotlib.pyplot as plt
dataset={
    'TID':[1,2,3,4,5],
'Items': [['bread','milk'],['bread','diaper','beer','egg'],
          ['milk','diaper','beer','cola'],['bread','milk','diaper','beer']
          ,['bread','milk','diaper','cola']]
}
df=pd.Dataframe(dataset)
basket_sets=pd.get_dummies(df['Items'].apply(pd.Series).stack()).groupby(level=0).sum()
frequent_itemsets=apriori(basket_sets,min_support=0.5,use_colnames=True)
rules=association_rules(frequent_itemsets,metric='lift',min_threshold=1)
print(frequent_itemsets)
print(rules)
plt.figure(figsize=(10,6))
plt.scatter(rules['support'],rules['confidence'],alpha=0.5)
plt.title('Support vs Confidence')
plt.show()