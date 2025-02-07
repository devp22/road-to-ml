import pandas as pd # type: ignore

df = pd.read_csv('datasets/titanic.csv')

# isNa method to check for missing values
print(df.isna())