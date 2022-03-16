# %% [markdown]
# # Question 2.
# Implement the Principal Component Analysis algorithm for reducing the dimensionality of the points
# given in the datasets: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.
# data. Each point of this dataset is a 4-dimensional vector (d = 4) given in the first column of the datafile.
# Reduce the dimensionality to 2 (k = 2). This dataset contains 3 clusters. Ground-truth cluster IDs are
# given as the fifth column of the data file. In order to evaluate the performance of the PCA algorithm,
# perform clustering (in 3 clusters) before and after dimensionality reduction using the Spectral Clustering
# algorithm and then find the percentage of points for which the estimated cluster label is correct. Report
# the accuracy of the Spectral Clustering algorithm before and after the dimensionality reduction. Report
# the reconstruction error for k = 1, 2, 3. [15 Marks]
# 1
#

# %%

import sys
import numpy as np
import numpy.linalg as la
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'axes.facecolor': 'lightblue', 'figure.facecolor': 'lightblue'})


# %% [markdown]
# - KMeans Cluster From Scratch

# %%


class KMeans:
    def __init__(self, n_clusters=2, tollerance=0.001, max_iter=10):
        self.k = n_clusters
        self.tollerance = tollerance
        self.max_iter = max_iter

    def fit_predict(self, data):

        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureIndex, featureset in enumerate(data):
                distances = [la.norm(featureset-self.centroids[centroid])
                             for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureIndex)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(
                    data[self.classifications[classification]], axis=0)

            optimized = True

            for c in self.centroids:
                centroid_shift = np.sum(
                    (self.centroids[c]-prev_centroids[c])/prev_centroids[c]*100.0)
                if centroid_shift > self.tollerance:
                    optimized = False

            if optimized:
                break

        predictions = np.empty([len(data)])
        for classification in self.classifications:
            predictions[self.classifications[classification]] = classification
        return predictions


# %% [markdown]
# - Utility functions for Spectral  Clustering from scratch

# %%
def pairwise_distances(X, Y):

    # Calculate distances from every point of X to every point of Y

    # start with all zeros
    distances = np.empty((X.shape[0], Y.shape[0]), dtype='float')

    # compute adjacencies
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            distances[i, j] = la.norm(X[i]-Y[j])

    return distances


def nearest_neighbor_graph(X):
    '''
    Calculates nearest neighbor adjacency graph.
    https://en.wikipedia.org/wiki/Nearest_neighbor_graph
    '''
    X = np.array(X)

    # for smaller datasets use sqrt(#samples) as n_neighbors. max n_neighbors = 10
    n_neighbors = min(int(np.sqrt(X.shape[0])), 10)

    # calculate pairwise distances
    A = pairwise_distances(X, X)

    # sort each row by the distance and obtain the sorted indexes
    sorted_rows_ix_by_dist = np.argsort(A, axis=1)

    # pick up first n_neighbors for each point (i.e. each row)
    # start from sorted_rows_ix_by_dist[:,1] because because sorted_rows_ix_by_dist[:,0] is the point itself
    nearest_neighbor_index = sorted_rows_ix_by_dist[:, 1:n_neighbors+1]

    # initialize an nxn zero matrix
    W = np.zeros(A.shape)

    # for each row, set the entries corresponding to n_neighbors to 1
    for row in range(W.shape[0]):
        W[row, nearest_neighbor_index[row]] = 1

    # make matrix symmetric by setting edge between two points if at least one point is in n nearest neighbors of the other
    for r in range(W.shape[0]):
        for c in range(W.shape[0]):
            if(W[r, c] == 1):
                W[c, r] = 1

    return W


def compute_laplacian(W):
    '''
    Reference for simple: https://en.wikipedia.org/wiki/Laplacian_matrix
        simple:
            L = D - W
    '''
    # calculate row sums
    d = W.sum(axis=1)

    # create degree matrix
    D = np.diag(d)
    L = D - W
    return L


def get_eigvecs(L, k):
    '''
    Calculate Eigenvalues and EigenVectors of the Laplacian Matrix.
    Return k eigenvectors corresponding to the smallest k eigenvalues.
    Uses real part of the complex numbers in eigenvalues and vectors.
    The Eigenvalues and Vectors will be complex numbers when using
    NearestNeighbor adjacency matrix for W.
    '''

    eigvals, eigvecs = la.eig(L)
    # sort eigenvalues and select k smallest values - get their indices
    ix_sorted_eig = np.argsort(eigvals)[:k]

    # select k eigenvectors corresponding to k-smallest eigenvalues
    return eigvecs[:, ix_sorted_eig]


# %% [markdown]
# - Spectral  Clustering from scratch

# %%
def spectral_clustering(X, k):

    # create weighted adjacency matrix
    W = nearest_neighbor_graph(X)

    # create unnormalized graph Laplacian matrix
    L = compute_laplacian(W)

    # create projection matrix with first k eigenvectors of L
    E = get_eigvecs(L, k)

    # return clusters using k-means on rows of projection matrix
    f = KMeans(n_clusters=k).fit_predict(E)  # k_means_clustering(E,k)
    return np.ndarray.tolist(f)

# %% [markdown]
# - Utility function for confusion Matrix And Accuracy Report

# %%
def confusion_matrix(actual, pred):
    classes = np.unique(actual)
    no_of_classes = len(classes)
    actual = np.array([np.where(classes==x)[0][0] for x in actual])
    pred = np.array([np.where(classes==x)[0][0] for x in pred])
    
    cm = np.zeros((no_of_classes,no_of_classes))

    for i in range(len(actual)):
        cm[actual[i]][pred[i]]+=1

    return cm

def confusionMatrixAndAccuracyReport(Y_test, Y_pred, title):
    cm = confusion_matrix(Y_test, Y_pred)
    overallAccuracy = np.trace(cm)/sum(cm.flatten())

    classwiseAccuracy = np.zeros(len(cm))
    for n in range(len(cm)):
        for i in range(len(cm)):
            for j in range(len(cm)):
                if (i != n and j != n) or (i == n and j == n):
                    classwiseAccuracy[n] += cm[i][j]

    classwiseAccuracy /= sum(cm.flatten())

    plt.figure(figsize=(6, 6))
    plt.title('{0} Accuracy Score: {1:3.3f}'.format(
        title, overallAccuracy), size=12)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    sns.heatmap(data=cm, annot=True, square=True,  cmap='Blues')

    plt.show()
    print('Overall Accuracy Score: {0:3.3f}'.format(overallAccuracy))
    print('Classwise Accuracy Score: {0}'.format(classwiseAccuracy))


# %% [markdown]
# - Data load

# %%

data_path = sys.argv[1] if len(sys.argv) > 1 else 'data-ques-2/iris.data'

dataset = pd.read_csv(data_path, names=[
                      'Sepal.Length', 'Sepal.Width', ' Petal.Length', 'Petal.Width', 'Class'])
#print (sys.argv)
dataset.head()


# %%
features = ['Sepal.Length', 'Sepal.Width', ' Petal.Length', 'Petal.Width']
X = dataset[features].values
Y = dataset['Class'].values


# %%
X = StandardScaler().fit_transform(X)
Y_bin = LabelEncoder().fit_transform(Y)


# %% [markdown]
# - PCA Decomposition from scratch

# %%


class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit_transform(self, X_data):
       # centering data
        self.X_mean = np.mean(X_data, axis=0)
        x_centered = X_data - self.X_mean
        # calculating covariance matrix
        x_cov = np.cov(x_centered.T)
        # eigendecomposition
        eigenvals, eigenvecs = la.eig(x_cov)
        # sorting
        i = np.argsort(eigenvals)[::-1]
        self.eigenvecs = eigenvecs[:, i]
        eigenvals = eigenvals[i]
        # retaining the eigenvectors for first n PCs
        self.X_evecs_n = self.eigenvecs[:, :self.n_components]

        return np.dot(X_data - self.X_mean, self.X_evecs_n)

    def inverse_transform(self, data):
        return np.dot(data, self.X_evecs_n.T)+self.X_mean


# %%
#from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data=principalComponents, columns=['PC 1', 'PC 2'])


# %%
sns.scatterplot(data=principalDf, x='PC 1',
                y='PC 2', hue=Y,  palette='rocket_r')


# %%

print('\nBefore Dimensionality Reduction:\n')
pred = spectral_clustering(X, 3)

confusionMatrixAndAccuracyReport(Y_bin, pred,'\nBefore Dimensionality Reduction:\n')


# %%
print('\nAfter Dimensionality Reduction:\n')
predPca = spectral_clustering(principalDf, 3)

confusionMatrixAndAccuracyReport(Y_bin, predPca,'\nAfter Dimensionality Reduction:\n')

print()
# %%
def reconstructionError(X_train, X_projected):
    return np.round(np.sum((X_train - X_projected) ** 2, axis=1).mean(), 3)


# %%
for k in range(3):
    pca_k = PCA(n_components=k)
    pc_x_train = pca_k.fit_transform(X)
    pc_x_projected = pca_k.inverse_transform(pc_x_train)
    print(
        f'The reconstruction error for k = {k+1} is :: {reconstructionError(X,pc_x_projected)}')
