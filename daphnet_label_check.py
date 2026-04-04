import pandas as pd

df = pd.read_csv("파일이름.txt", sep=" ", header=None)

print(df.iloc[:, -1].value_counts())