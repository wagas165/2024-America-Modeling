import pandas as pd

df=pd.read_csv('2023-wimbledon-points_副本.csv')
df.dropna(subset=['leverage', 'momentum'],inplace=True)
df.to_csv('2023-wimbledon-points_副本.csv')

