import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv('datasets/titanic.csv')


#-----------------------------------------------------------
# In a Pandas DataFrame, data types can be broadly categorized into numerical, categorical, and datetime.
# numerical: int, float
# categorical: string/object, category
# dtetime: datetime64, datetime, Timestamp

# we can identify them by using dtypes() method
# print(df.dtypes)
# to select appropriate ones, we can use methods like select_dtypes
# print(df.select_dtypes(include=np.number))
# print(df.select_dtypes(include='object'))


# Feature engineering is the process of transforming raw data into features that are suitable for machine learning models. There are various techniques that can be used in feature engineering to create new features by combining or transforming the existing ones. The following are some of the commonly used feature engineering techniques:

#-----------------------------------------------------------

# 1. One Hot Encoding is a method for converting categorical variables into a binary format. It creates new columns for each category where 1 means the category is present and 0 means it is not. The primary purpose of One Hot Encoding is to ensure that categorical data can be effectively used in machine learning models.

#   a.  using pd
# pd.get_dummies is used to one-hot encode the categorical columns

# df_encoded = pd.get_dummies(df, columns=['Name','Sex','Ticket','Cabin','Embarked'])
# print(df_encoded)

#   b.  using sklearn
# another method is using OneHotEncoder from sklearn and is more recommended as it has a lot of different tools

# categorical_columns = df.select_dtypes(include="object").columns.tolist()
# print(categorical_columns)
# encoder = OneHotEncoder(sparse_output=False)
# encoded_data = encoder.fit_transform(df[categorical_columns])
# encoded_df = pd.DataFrame(encoded_data,columns=encoder.get_feature_names_out(categorical_columns))
# df = pd.concat([df,encoded_df],axis=1)
# df.drop(categorical_columns,axis=1,inplace=True)
# print(df)

#-----------------------------------------------------------
# 2. Binning, also known as bucketing or discretization, is a data preprocessing technique used to transform continuous numerical data into discrete categories or intervals. It involves dividing the range of a continuous variable into a set of bins and assigning each data point to its corresponding bin.

# this method transforms data in bins of different ranges so as to study the data better
# e.g considering ages of customers in a shop, using continuous data might be difficult to understand trend, however grouping them based on ages gives us better insights on what age group does visit a shop most

# categorical_columns = df.select_dtypes(include="number").columns.tolist()
# print(categorical_columns)
# df_age = df['Age'].fillna(df['Age'].mean())
# hist, bins = np.histogram(df_age,bins=10)

#Bin Edges (bins), are the boundaries that define the intervals (bins) into which the data is divided. Each bin includes values up to, but not including, the next bin edge. Histogram Counts (hist) are the frequencies or counts of data points that fall within each bin

# print(hist)
# print(bins)
# plt.hist(df_age,bins=10)
# plt.show()

# the data shows that highest group of people who boarded titatnic were from range 22.919 to 30.502 and their number was 183

#-----------------------------------------------------------
# 3. The most common scaling techniques are standardization and normalization. Standardization scales the variable so that it has zero mean and unit variance. Normalization scales the variable so that it has a range of values between 0 and 1.
#Scaling techniques like standardization and normalization transform data to make it easier to analyze and interpret. Scaling doesn’t distort the underlying relationships between features and the target variable—it standardizes their representation, making patterns more visible, easier to interpret, and usable for ML models. It changes the numbers, not the essence of the data.

# STANDARDIZATION
# df['Age'].fillna(df['Age'].mean(),inplace=True)
# df_age_standardized = stats.zscore(df['Age'])
# df_age_standardized_compare = pd.concat([df['Age'],df_age_standardized],axis=1)
# print(df_age_standardized_compare)

# NORMALIZATION
df['Age'].fillna(df['Age'].mean(),inplace=True)
scaler = MinMaxScaler
scaler = MinMaxScaler()
df['Age_Normalized'] = scaler.fit_transform(df[['Age']])  # Ensure it's 2D with double brackets

print(df)