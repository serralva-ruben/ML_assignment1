from sklearn.cluster import DBSCAN
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

#2
ulu_data = pd.read_csv("./ulu.csv")
X = ulu_data.values

dbscan_clustering = DBSCAN().fit(X)

labels = dbscan_clustering.labels_

n_of_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print(f"Number of clusters using DBSCAN is: {n_of_clusters}")

#3

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', s=5)
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('x vs y')

axes[1].scatter(X[:, 0], X[:, 2], c=labels, cmap='rainbow', s=5)
axes[1].set_xlabel('x')
axes[1].set_ylabel('z')
axes[1].set_title('x vs z')

axes[2].scatter(X[:, 1], X[:, 2], c=labels, cmap='rainbow', s=5)
axes[2].set_xlabel('y')
axes[2].set_ylabel('z')
axes[2].set_title('y vs z')

plt.tight_layout()
plt.show()

#4
kmeans_clustering = KMeans(n_clusters=8)
kmeans_labels = kmeans_clustering.fit_predict(X)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='rainbow', s=5)
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('x vs y')

axes[1].scatter(X[:, 0], X[:, 2], c=kmeans_labels, cmap='rainbow', s=5)
axes[1].set_xlabel('x')
axes[1].set_ylabel('z')
axes[1].set_title('x vs z')

axes[2].scatter(X[:, 1], X[:, 2], c=kmeans_labels, cmap='rainbow', s=5)
axes[2].set_xlabel('y')
axes[2].set_ylabel('z')
axes[2].set_title('y vs z')

plt.tight_layout()
plt.show()

#5
dbscan_silhouette = silhouette_score(X, labels)
kmeans_silhouette = silhouette_score(X, kmeans_labels)

print(f"Silhouette score for DBScan: {dbscan_silhouette}")
print(f"Silhouette score for K-means: {kmeans_silhouette}")

#6
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='rainbow', s=5)
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('PCA colored by clusters determiend by DBSCAN')
plt.colorbar()
plt.show()

#7
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='rainbow', s=5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Dataset Plot in 3D')

fig.colorbar(scatter, ax=ax)
plt.show()



