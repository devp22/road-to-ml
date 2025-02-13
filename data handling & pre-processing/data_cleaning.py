import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('datasets/titanic.csv')

#-----------------------------------------------------------
# 1. isNa method to check for missing values
# print(df.isna())

#-----------------------------------------------------------
# 2. isNull method to check for missing values
# print(df.isna())

#-----------------------------------------------------------
# 3. dropNa() method removes rows or columns containing missing values, axis=0 removes rows, axis=1 removes columns
# print("size before dropping rows: ",df.size)
# df = df.dropna()
# print("size after dropping rows: ",df.size)

#-----------------------------------------------------------
# 4. interpolate() Estimates missing values based on the existing data.
# print("df before interpolation: ")
# print(df)
# df.interpolate(inplace=True)
# print("df after after interpolation: ")
# print(df)

#-----------------------------------------------------------
# 5. fillna() replaces missing values with specific values
#inplace=True modifies the dataframe directly without saving it afain in itself
# print("df before filling na: ")
# print(df)
# df.fillna(df.mean(),inplace=True)
# print("df after filling na: ")
# print(df)

#-----------------------------------------------------------
# 6. replace() method is used to replace values with other
# df.replace(np.nan,-1,inplace=True)

#-----------------------------------------------------------
# 7. duplicate() is used to detect duplicated rows in df and can be dropped by using drop_duplicates()
# print("duplicates")
# print(df.duplicated())
# print("dropped duplicates")
# print(df.drop_duplicates())

#-----------------------------------------------------------
# 8. Outliers are data points that significantly deviate from the rest of the data. They can be detected and handled using various methods in Pandas.
# to remove it we can use techniques like z-score, and it can be implemented with available libraries instead of relying on creating a logic from scratch
# z-score = This value/score helps to understand that how far is the data point from the mean.
#  Zscore = (data_point -mean) / std. deviation 
# NOTE: Values with a Z-score greater than 3 or less than -3 are often considered outliers. 

# df = df.dropna()
# df_non_nan = pd.concat( [df.iloc[:,:3] , df.iloc[:,5:8],df.iloc[:,9]],axis=1)
# df_zscore = np.abs(stats.zscore(df_non_nan))
# print(df_zscore)
# # the max wil show many columns having values with zscore greater than 3, those are our outliers that need to be removed
# print(df_zscore.describe())
# #logic to drop outliers with more than 2.99
# outliers =(df_zscore>2.99).any(axis=1)
# dropped_outliers = df_zscore.loc[~outliers]
# print(dropped_outliers)
# print(dropped_outliers.describe())

#-----------------------------------------------------------
# 9. In a Pandas DataFrame, data types can be broadly categorized into numerical, categorical, and datetime.
# numerical: int, float
# categorical: string/object, category
# dtetime: datetime64, datetime, Timestamp

# we can identify them by using dtypes() method
print(df.dtypes)
# to select appropriate ones, we can use methods like select_dtypes
print(df.select_dtypes(include=np.number))
print(df.select_dtypes(include='object'))
