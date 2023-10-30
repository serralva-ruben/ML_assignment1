import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
pd.set_option('display.max_rows', None)

# 1. Dataset 1 creation:

rng = np.random.default_rng(42)

mean_class1 = [2, 2]
cov_class1 = [[0.5, 0], [0, 0.5]]
data1_class1 = rng.multivariate_normal(mean_class1, cov_class1, 40)
mean_class2 = [4, 4]
cov_class2 = [[0.5, 0], [0, 0.5]]
data1_class2 = rng.multivariate_normal(mean_class2, cov_class2, 40)

df_class1 = pd.DataFrame(data1_class1, columns=['X', 'Y'])
df_class2 = pd.DataFrame(data1_class2, columns=['X', 'Y'])

print("Dataset for Class 1:")
print(df_class1)
print("\nDataset for Class 2:")
print(df_class2)

# 2
# generating outliers:

outliers_class1 = rng.uniform(0, 6, (4, 2))
outliers_class2 = rng.uniform(0, 6, (4, 2))

df_outliers_class1 = pd.DataFrame(outliers_class1, columns=['X', 'Y'])
df_outliers_class2 = pd.DataFrame(outliers_class2, columns=['X', 'Y'])

print("Outliers class1: ")
print(df_outliers_class1)
print("\nOutliers class2: ")
print(df_outliers_class2)

# creating dataset 1 by putting class 1 and class 2 togheter
X1 = np.vstack((data1_class1, data1_class2))
# creating dataset 2 by putting dataset 1 and the outliers togheter
X2 = np.vstack((X1, outliers_class1, outliers_class2))
# setting labels for the data
y1 = np.array([0]*40 + [1]*40)
y2 = np.array([0]*40 + [1]*40 + [0]*4 + [1]*4)

# 3. Data Split:

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.25, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.25, random_state=42)

df_X1_train = pd.DataFrame(X1_train, columns=['X', 'Y'])
df_X1_train['Label'] = y1_train
print(f'\nX1_train\n{df_X1_train}')

df_X1_test = pd.DataFrame(X1_test, columns=['X', 'Y'])
df_X1_test['Label'] = y1_test
print(f'\nX1_test\n{df_X1_test}')

df_X2_train = pd.DataFrame(X2_train, columns=['X', 'Y'])
df_X2_train['Label'] = y2_train
print(f'\nX2_train\n{df_X2_train}')

df_X2_test = pd.DataFrame(X2_test, columns=['X', 'Y'])
df_X2_test['Label'] = y2_test
print(f'\nX2_test\n{df_X2_test}')

# 4. Train Models and Report Results:

def train_and_report(classifier, x_train, y_train, x_test, y_test, dataset_name):
    classifier.fit(x_train, y_train)
    y_pred_train = classifier.predict(x_train)
    y_pred_test = classifier.predict(x_test)
    
    print(f"\n{dataset_name} - {classifier.__class__.__name__} Training Accuracy:", accuracy_score(y_train, y_pred_train))
    print(f"{dataset_name} - {classifier.__class__.__name__} Testing Accuracy:", accuracy_score(y_test, y_pred_test))
    print(f"Confusion Matrix ({dataset_name}):\n", confusion_matrix(y_test, y_pred_test))

# Dataset 1 and Dataset 2 using k-NN
knn1 = KNeighborsClassifier(n_neighbors=3)
train_and_report(knn1, X1_train, y1_train, X1_test, y1_test, "Dataset 1")
knn2 = KNeighborsClassifier(n_neighbors=3)
train_and_report(knn2, X2_train, y2_train, X2_test, y2_test, "Dataset 2")

# Dataset 1 and Dataset 2 using Naive Bayes
nb1 = GaussianNB()
train_and_report(nb1, X1_train, y1_train, X1_test, y1_test, "Dataset 1")
nb2 = GaussianNB()
train_and_report(nb2, X2_train, y2_train, X2_test, y2_test, "Dataset 2")

# Dataset 1 and Dataset 2 using Decision Trees
dt1 = DecisionTreeClassifier(random_state=42)
train_and_report(dt1, X1_train, y1_train, X1_test, y1_test, "Dataset 1")
dt2 = DecisionTreeClassifier(random_state=42)
train_and_report(dt2, X2_train, y2_train, X2_test, y2_test, "Dataset 2")

# Dataset 1 and Dataset 2 using Random Forests
rf1 = RandomForestClassifier(random_state=42)
train_and_report(rf1, X1_train, y1_train, X1_test, y1_test, "Dataset 1")
rf2 = RandomForestClassifier(random_state=42)
train_and_report(rf2, X2_train, y2_train, X2_test, y2_test, "Dataset 2")

def plot_decision_boundaries(models, model_names, xs, ys):

    _ , axes = plt.subplots(2, 4, figsize=(15, 8))

    for idx, (model, name, X, y) in enumerate(zip(models, model_names, xs * 4, ys * 4,)):
        row, col = divmod(idx, 4)
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, .1),np.arange(y_min, y_max, .1))

        # Obtain labels for each point in the mesh using the trained model.
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot the decision boundary on the current subplot
        axes[row, col].contourf(xx, yy, Z, alpha=0.75)

        # Plot the training points on the current subplot
        axes[row, col].scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=50, linewidth=1)
        axes[row, col].set_title(name)

    plt.tight_layout()
    plt.show()

models = [knn1, nb1, dt1, rf1, knn2, nb2, dt2, rf2]
model_names = [
    "k-NN Dataset 1", "Naive Bayes 1", "Decision Trees 1", "Random Forests 1",
    "k-NN Dataset 2", "Naive Bayes 2", "Decision Trees 2", "Random Forests 2"
]
Xs = [X1, X2] * 4
ys = [y1, y2] * 4

plot_decision_boundaries(models, model_names, Xs, ys)