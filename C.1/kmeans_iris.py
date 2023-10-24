from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import silhouette_samples, silhouette_score
import pandas as pd
import matplotlib.pyplot as plt

#2.
data = pd.read_csv("./iris.csv")

#selecting rows, excluding the last column, because kMeans doesnt use labels
X = data.iloc[:, :-1].values

#kMeans clustering applied, using 3 clusters because of 3 known distinct species in iris
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
data['Cluster'] = kmeans.labels_

#excluding the last two columns, label and cluster
properties = data.columns[:-2]

plt.figure(figsize=(15, 10))
plot_number = 1

for i in range(len(properties)):
    for j in range(i+1, len(properties)):
        plt.subplot(2, 3, plot_number)
        plt.scatter(data[properties[i]], data[properties[j]], c=data['Cluster'], cmap='rainbow', edgecolor='k', s=60)
        plt.xlabel(properties[i])
        plt.ylabel(properties[j])
        plt.title(f"{properties[i]} vs {properties[j]}")
        plot_number += 1

plt.tight_layout()
plt.show()

#3.
labels = []

for cluster in range(3):  
    true_labels = data[data['Cluster'] == cluster]['species']
    most_common_label = true_labels.mode().iloc[0]
    labels.append(most_common_label)

data['MappedCluster'] = data['Cluster'].apply(lambda x: labels[x])

accuracy = accuracy_score(data['species'], data['MappedCluster'])
print(f"Accuracy: {accuracy:.2f}")

conf_matrix = confusion_matrix(data['species'], data['MappedCluster'])
print("Confusion Matrix:")
print(conf_matrix)

#4.
silhouette_vals = silhouette_samples(X, data['Cluster'])
data['Silhouette'] = silhouette_vals

avg_silhouette_scores = data.groupby('Cluster')['Silhouette'].mean()

print("Average Silhouette Scores for every cluster:")
print(avg_silhouette_scores)

overall_avg_silhouette = silhouette_score(X, data['Cluster'])
print(f"\nOverall Average Silhouette Score: {overall_avg_silhouette:.2f}")

#5
unknown_data = pd.read_csv("./unknown_species.csv")

#lenght features are in columns 1 to 4
X_unknown = unknown_data.iloc[:, 1:5].values

unknown_data['PredictionCluster'] = kmeans.predict(X_unknown)

silhouette_scores = silhouette_samples(X_unknown, unknown_data['PredictionCluster'])

unknown_data['Silhouette'] = silhouette_scores

#saving prediction clusters to a new file, so I can load it and compare it to prediction clusters of custom K-Means
unknown_data[['id', 'PredictionCluster']].to_csv('skicit-learn_KMeans_predictions.csv', index=False)

#printing the outcome of prediction clusters
print("Here are the prediction clusters for uknown species:")
print(unknown_data[['id', 'PredictionCluster']].to_string(index=False))

#printing the silhouette scores of different flowers
#printing the outcome
print("Here are the silhouette scores for uknown species:")
print(unknown_data[['id', 'Silhouette']].to_string(index=False))

#interpretation of the silhouette scores
for index, row in unknown_data.iterrows():
    print(f"\nFlower ID {row['id']}:")
    if row['Silhouette'] > 0.6:
        print("This flower is well clustered.")
    elif row['Silhouette'] > 0.3:
        print("This flower is moderately clustered, may have some overlap or closeness with another cluster.")
    else:
        print("This flower might be close to the boundary of another cluster and maybe better suited to a neighboring cluster.")







