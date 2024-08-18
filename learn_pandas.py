import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

s = pd.Series(np.random.randn(4), name='daily returns')
s.index = ['l1', 'l2', 'l3', 'l4']
print (s)
print(s.describe())

df = pd.read_csv('https://github.com/QuantEcon/QuantEcon.lectures.code/raw/master/pandas/data/test_pwt.csv')
print(df)
print(df.iloc[2:5, 0:4])
print(df.loc[df.index[2:5], ['country', 'tcgdp']])
df = df.set_index('country')
print(df)

df = df.sort_values(by='POP', ascending=False)
df['POP'].plot(kind='bar')
plt.show()