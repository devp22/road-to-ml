import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,normalize
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import os


os.environ["OMP_NUM_THREADS"] = "1"

df = pd.read_csv('datasets/Mall_Customers.csv')
cols = df.select_dtypes(include='object')
labelencoder = LabelEncoder()
labels = labelencoder.fit_transform(df['Gender'])
df['Gender'] = labels
df.dropna(inplace=True)

# -------------------KMEANS-----------------------
X = df.drop(['CustomerID','Spending Score (1-100)'],axis=1)
y = df['Spending Score (1-100)']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=90)
X_train_normalized = normalize(X_train)
X_test_normalized = normalize(X_test)
kmeans = KMeans(n_init=5,n_clusters=3,max_iter=5,random_state=90)
kmeans.fit(X_train_normalized)
sns.set_style('whitegrid')
sns.scatterplot(data=X_train,x='Age',y='Annual Income (k$)',hue=kmeans.labels_)
plt.show()