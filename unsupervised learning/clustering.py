import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,normalize
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import os
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

os.environ["OMP_NUM_THREADS"] = "1"

df = pd.read_csv('datasets/Mall_Customers.csv')
cols = df.select_dtypes(include='object')
labelencoder = LabelEncoder()
labels = labelencoder.fit_transform(df['Gender'])
df['Gender'] = labels
df.dropna(inplace=True)

# -------------------KMEANS-----------------------
# X = df.drop(['CustomerID','Spending Score (1-100)'],axis=1)
# y = df['Spending Score (1-100)']
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=90)
# X_train_normalized = normalize(X_train)
# X_test_normalized = normalize(X_test)
# kmeans = KMeans(n_init=5,n_clusters=3,max_iter=5,random_state=90)
# kmeans.fit(X_train_normalized)
# sns.set_style('whitegrid')
# sns.scatterplot(data=X_train,x='Age',y='Annual Income (k$)',hue=kmeans.labels_)
# plt.show()

# -------------------AGGLOMERATIVE-----------------------
# X = df.drop(['CustomerID','Spending Score (1-100)'],axis=1)
# y = df['Spending Score (1-100)']
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=90)
# X_train_normalized = normalize(X_train)
# X_test_normalized = normalize(X_test)
# agg = AgglomerativeClustering(n_clusters=3,metric='euclidean',compute_distances=True)
# agg.fit(X_train_normalized)
# sns.set_style('whitegrid')
# sns.scatterplot(data=X_train,x='Age',y='Annual Income (k$)',hue=agg.labels_)
# plt.show()
# # MAKING DENDOGRAM
# Z = linkage(X_train_normalized,'ward')
# dendrogram(Z)
# plt.title('Hierarchical Clustering')
# plt.xlabel('Data Points of Cluster (X)')
# plt.ylabel('Distance')
# plt.show()

# -------------------DBSCAN-----------------------
X = df.drop(['CustomerID','Spending Score (1-100)'],axis=1)
y = df['Spending Score (1-100)']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=90)
X_train_normalized = normalize(X_train)
X_test_normalized = normalize(X_test)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
tsne = TSNE(n_components=2, random_state=42)
X_train_tsne = tsne.fit_transform(X_train_scaled)

dbscan = DBSCAN(eps=0.7,min_samples=6,metric='euclidean')
clusters = dbscan.fit_predict(X_train_scaled)
sns.set_style('whitegrid')
sns.scatterplot(x=X_train_tsne[:, 0], y=X_train_tsne[:, 1], hue=clusters, palette='tab10', s=60)
plt.show()
score = silhouette_score(X_train_scaled, clusters)
print(f"Silhouette Score: {score}")
