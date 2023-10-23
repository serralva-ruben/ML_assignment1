import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, adjusted_rand_score

# Ensure reproducibility with a fixed random seed
np.random.seed(42)

# Load iris dataset
iris = datasets.load_iris()
data = iris.data
target = iris.target

# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(data)

# Create a DataFrame with the iris data and the cluster assignments
df = pd.DataFrame(data, columns=iris.feature_names)
df['cluster'] = clusters  # Add the clusters as a new column

# Create the pairplot
sns.pairplot(data=df, hue='cluster', plot_kws={"s": 50}, palette="Dark2")
plt.suptitle("Pairwise scatter plots colored by K-means clusters", y=1.02)
plt.show()

# Create a mapping for K-means labels to true labels
mapping = {}
for cluster in [0, 1, 2]:
    true_labels = target[clusters == cluster]
    mapping[cluster] = np.bincount(true_labels).argmax()

# Map the clusters to actual labels
predicted_labels = [mapping[cluster] for cluster in clusters]

# Calculate accuracy
accuracy = accuracy_score(target, predicted_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Calculate confusion matrix
conf_matrix = confusion_matrix(target, predicted_labels)
print("Confusion Matrix:")
print(conf_matrix)

# Adjusted Rand Score
ars = adjusted_rand_score(target, clusters)
print(f"Adjusted Rand Score: {ars:.2f}")

# Evaluation
if ars > 0.7:
    print("The trained model is suitable for this problem.")
else:
    print("The trained model may not be highly suitable for this problem.")

from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np

# Ensure reproducibility with a fixed random seed
np.random.seed(42)

# Load iris dataset
iris = datasets.load_iris()
data = iris.data

# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(data)

# Calculate silhouette scores for each sample
silhouette_vals = silhouette_samples(data, clusters)

# Calculate average silhouette score for each cluster
for cluster in [0, 1, 2]:
    cluster_silhouette_vals = silhouette_vals[clusters == cluster]
    avg_cluster_silhouette = np.mean(cluster_silhouette_vals)
    print(f"Average silhouette score for cluster {cluster}: {avg_cluster_silhouette:.2f}")

# Overall silhouette score
overall_avg_silhouette = silhouette_score(data, clusters)
print(f"Overall average silhouette score: {overall_avg_silhouette:.2f}")

import pandas as pd
from sklearn.metrics import silhouette_score

# Load the unknown species data
unknown_data = pd.read_csv('unknown_species.csv')

# Predict the clusters for the unknown species using the trained k-means model
predicted_clusters = kmeans.predict(unknown_data)

# Print the predicted clusters for each flower
for idx, cluster in enumerate(predicted_clusters, 1):
    print(f"Flower ID {idx} is predicted to belong to cluster {cluster}")

# Compute silhouette score for each individual flower
for idx, (data_point, cluster) in enumerate(zip(unknown_data.values, predicted_clusters), 1):
    silhouette = silhouette_score(unknown_data, [cluster]*len(unknown_data))
    print(f"Silhouette score for Flower ID {idx}: {silhouette:.2f}")
