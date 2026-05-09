import pandas as pd

df = pd.read_csv("dataset/spam.csv", encoding='latin-1')

print(df.head())