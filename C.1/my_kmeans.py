import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from scipy.stats import mode

def initialize_centres(X, k):
    #randomly initialize the centres
    random_idx = np.random.choice(X.shape[0], k, replace=False)
    centres = X[random_idx, :]
    return centres

def assign_clusters(X, centres):
    #assign each data point to the nearest centroid
    distances = np.sqrt(((X - centres[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def update_centres(X, labels, k):
    #update the centres based on the mean of the points assigned to them
    new_centres = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    return new_centres

def kmeans(X, k, max_iters=100):
    #K-means clustering
    centres = initialize_centres(X, k)
    for _ in range(max_iters):
        labels = assign_clusters(X, centres)
        new_centres = update_centres(X, labels, k)
        if np.all(centres == new_centres):
            break
        centres = new_centres
    return labels, centres

def assign_clusters_using_centroids(X, centroids):
    distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)


iris_data = pd.read_csv("./iris.csv")
X_iris = iris_data.iloc[:, :-1].values

_, centroids = kmeans(X_iris, 3)

unknown_data = pd.read_csv("./unknown_species.csv")
X_unknown = unknown_data.iloc[:, 1:5].values
labels_custom = assign_clusters_using_centroids(X_unknown, centroids)

predicted_clusters_sklearn = pd.read_csv('skicit-learn_KMeans_predictions.csv')
predicted_clusters_sklearn['CustomPredictionCluster'] = labels_custom

labels_sklearn = predicted_clusters_sklearn['PredictionCluster'].values

#the value of ARI is between -1 and 1
#if close to 1, results are very similar
#if close to 0, results are random
#if negative, results are worse than random
ari = adjusted_rand_score(labels_custom, labels_sklearn)
print(f"Adjusted Rand Index comparing custom k-means with scikit-learn's KMeans: {ari:.2f}")

mapping = {}
for cluster in range(3):
    mask = (predicted_clusters_sklearn['CustomPredictionCluster'] == cluster)
    mode_result = mode(predicted_clusters_sklearn[mask]['PredictionCluster'])
    
    #checking if mode_result[0] is scalar
    if np.isscalar(mode_result[0]):
        most_common = mode_result[0]
    else:
        most_common = mode_result[0][0]
    
    mapping[cluster] = most_common

#applying the mapping to the CustomPredictionCluster column
predicted_clusters_sklearn['AlignedCustomCluster'] = predicted_clusters_sklearn['CustomPredictionCluster'].map(mapping)

#saving the updated DataFrame to the CSV file
predicted_clusters_sklearn.to_csv('skicit-learn_KMeans_predictions.csv', index=False)


