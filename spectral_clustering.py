# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 11:13:07 2024

@author: Evelina
"""

"""
Work with Spectral clustering.
Do not use global variables!
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pickle
from scipy.special import comb


######################################################################
#####     CHECK THE PARAMETERS     ########
######################################################################
def compute_proximity_matrix(data, sigma):
    n = len(data)
    proximity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            dist = np.linalg.norm(data[i] - data[j])
            proximity_matrix[i][j] = np.exp(-dist**2 / (2 * sigma**2))
            proximity_matrix[j][i] = proximity_matrix[i][j]  # symmetric matrix
    return proximity_matrix

def compute_laplacian(proximity_matrix):
    n = len(proximity_matrix)
    D = np.diag(np.sum(proximity_matrix, axis=1))
    laplacian = D - proximity_matrix
    return laplacian

def compute_eigenvalues(laplacian, num_eigen):
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    return eigenvalues[:num_eigen], eigenvectors[:, :num_eigen]

def kmeans_clustering(eigenvectors, k, max_iter=300):
    # Step 1: Initialize centroids randomly
    np.random.seed(42)  # for reproducibility
    initial_indices = np.random.choice(eigenvectors.shape[0], k, replace=False)
    centroids = eigenvectors[initial_indices]

    for _ in range(max_iter):
        # Step 2: Assign clusters
        distances = np.sqrt(((eigenvectors - centroids[:, np.newaxis])**2).sum(axis=2))
        closest_cluster = np.argmin(distances, axis=0)

        # Step 3: Update centroids
        new_centroids = np.array([eigenvectors[closest_cluster == k].mean(axis=0) for k in range(centroids.shape[0])])
        
        # Check for convergence
        if np.allclose(centroids, new_centroids, rtol=1e-6):
            break
        
        centroids = new_centroids

    return closest_cluster

def calculate_SSE(data, labels, centroids):
    SSE = 0
    for k in range(centroids.shape[0]):
        cluster_data = data[labels == k]
        if cluster_data.size == 0:
            continue
        SSE += np.sum((cluster_data - centroids[k])**2)
    return SSE

def adjusted_rand_index(labels_true, labels_pred):
    # Create contingency table
    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    n = len(labels_true)
    contingency = np.histogram2d(class_idx, cluster_idx, bins=(len(classes), len(clusters)))[0]

    # Sum over rows & columns
    sum_comb_c = sum(comb(n_c, 2) for n_c in np.sum(contingency, axis=1))
    sum_comb_k = sum(comb(n_k, 2) for n_k in np.sum(contingency, axis=0))

    # Sum of the contingency table (n_ij choose 2)
    sum_comb = sum(comb(n_ij, 2) for n_ij in contingency.flatten())

    # Calculate the expected index
    expected_index = sum_comb_c * sum_comb_k / comb(n, 2)
    max_index = (sum_comb_c + sum_comb_k) / 2

    # Adjusted Rand Index
    if max_index == expected_index:
        return 0.0
    else:
        return (sum_comb - expected_index) / (max_index - expected_index)
    
def plot_clusters(data, labels, title='Cluster Plot'):
    # Unique clusters
    unique_labels = np.unique(labels)
    
    # Set up the plot
    plt.figure(figsize=(10, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))  # Color map for clusters
    
    # Plot each cluster
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)

        xy = data[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=8, label=f'Cluster {k}')

    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()


def spectral(
    data: NDArray[np.floating], labels: NDArray[np.int32], params_dict: dict
) -> tuple[
    NDArray[np.int32] | None, float | None, float | None, NDArray[np.floating] | None
]:
    """
    Implementation of the Spectral clustering  algorithm only using the `numpy` module.

    Arguments:
    - data: a set of points of shape 50,000 x 2.
    - dict: dictionary of parameters. The following two parameters must
       be present: 'sigma', and 'k'. There could be others.
       params_dict['sigma']:  in the range [.1, 10]
       params_dict['k']: the number of clusters, set to five.

    Return values:
    - computed_labels: computed cluster labels
    - SSE: float, sum of squared errors
    - ARI: float, adjusted Rand index
    - eigenvalues: eigenvalues of the Laplacian matrix
    """
    sigma = params_dict['sigma']
    k = params_dict['k']
    Nc = 5  # Number of clusters

    # Compute proximity matrix, laplacian, and eigendecomposition
    proximity_matrix = compute_proximity_matrix(data, sigma)
    laplacian = compute_laplacian(proximity_matrix)
    eigenvalues, eigenvectors = compute_eigenvalues(laplacian, Nc)
    
    # Cluster the eigenvectors using our KMeans implementation
    computed_labels = kmeans_clustering(eigenvectors, k)

    # Calculate centroids
    centroids = np.array([data[computed_labels == i].mean(axis=0) for i in range(k)])
    
    # Calculate SSE
    SSE = calculate_SSE(data, computed_labels, centroids)

    # Calculate ARI
    ARI = adjusted_rand_index(labels, computed_labels)


    # computed_labels: NDArray[np.int32] | None = None
    # SSE: float | None = None
    # ARI: float | None = None
    # eigenvalues: NDArray[np.floating] | None = None

    return computed_labels, SSE, ARI, eigenvalues


def spectral_clustering():
    """
    Performs spectral clustering on a dataset.

    Returns:
        answers (dict): A dictionary containing the clustering results.
    """

    answers = {}

    # Return your `spectral` function
    answers["spectral_function"] = spectral

    # Work with the first 10,000 data points: data[0:10000]
    # Do a parameter study of this data using Spectral clustering.
    # Minimmum of 10 pairs of parameters ('sigma' and 'xi').
    
    # Load the data
    data = np.load('question1_cluster_data.npy')
    labels = np.load('question1_cluster_labels.npy')

    # # Create a dictionary for each parameter pair ('sigma' and 'xi').
    # groups = {}
    
    # Parameter study with a range of sigma values
    sigmas = np.linspace(0.1, 1.0, 10)  # Example range for sigma
    groups = {}
    metrics = {"ARI": [], "SSE": []}

    for i, sigma in enumerate(sigmas):
        params = {"sigma": sigma, "k": 5}
        computed_labels, SSE, ARI, eigenvalues = spectral(data[:5000], labels[:5000], params)
        
        # Plot the results
        plot_clusters(data[:5000], computed_labels, title='Spectral Clustering Results')
        
        groups[i] = {"sigma": sigma, "ARI": ARI, "SSE": SSE}
        metrics["ARI"].append(ARI)
        metrics["SSE"].append(SSE)

    # For the spectral method, perform your calculations with 5 clusters.
    # In this cas,e there is only a single parameter, σ.

    # data for data group 0: data[0:10000]. For example,
    # groups[0] = {"sigma": 0.1, "ARI": 0.1, "SSE": 0.1}

    # data for data group i: data[10000*i: 10000*(i+1)], i=1, 2, 3, 4.
    # For example,
    # groups[i] = {"sigma": 0.1, "ARI": 0.1, "SSE": 0.1}

    # groups is the dictionary above
    answers["cluster parameters"] = groups
    answers["1st group, SSE"] = groups[0]["SSE"]

    # Identify the cluster with the lowest value of ARI. This implies
    # that you set the cluster number to 5 when applying the spectral
    # algorithm.

    # Create two scatter plots using `matplotlib.pyplot`` where the two
    # axes are the parameters used, with \sigma on the horizontal axis
    # and \xi and the vertical axis. Color the points according to the SSE value
    # for the 1st plot and according to ARI in the second plot.

    # Choose the cluster with the largest value for ARI and plot it as a 2D scatter plot.
    # Do the same for the cluster with the smallest value of SSE.
    # All plots must have x and y labels, a title, and the grid overlay.

    # Plotting the metrics
    sigmas = [group["sigma"] for group in groups.values()]
    ARIs = [group["ARI"] for group in groups.values()]
    SSEs = [group["SSE"] for group in groups.values()]

    plot_ARI = plt.scatter(sigmas, ARIs, c='blue')
    plt.xlabel('Sigma')
    plt.ylabel('ARI')
    plt.title('ARI by Sigma')
    plt.grid(True)
    plt.show()
    answers["cluster scatterplot with largest ARI"] = plot_ARI

    plot_SSE = plt.scatter(sigmas, SSEs, c='red')
    plt.xlabel('Sigma')
    plt.ylabel('SSE')
    plt.title('SSE by Sigma')
    plt.grid(True)
    plt.show()
    answers["cluster scatterplot with smallest SSE"] = plot_SSE


    # Identify the best parameters (here selecting based on the highest ARI)
    best_index = np.argmax(ARIs)
    best_sigma = groups[best_index]["sigma"]
    all_ARIs = []
    all_SSEs = []

    # Apply the best sigma to other data slices
    for i in range(5):
        params = {"sigma": best_sigma, "k": 5}
        data_slice = data[5000*i:5000*(i+1)]
        label_slice = labels[5000*i:5000*(i+1)]
        _, SSE, ARI, _ = spectral(data_slice, label_slice, params)
        all_ARIs.append(ARI)
        all_SSEs.append(SSE)
        
        

    # Plot of the eigenvalues (smallest to largest) as a line plot.
    # Use the plt.plot() function. Make sure to include a title, axis labels, and a grid.
    # eigenvalues = groups[best_index]
    # plot_eig = plt.plot(eigenvalues)
    # plt.title('Eigenvalues')
    # plt.xlabel('Index')
    # plt.ylabel('Eigenvalue')
    # plt.grid(True)
    # plt.show()
    # answers["eigenvalue plot"] = plot_eig
    
    plt.figure()
    plot_eig = plt.plot(np.arange(len(eigenvalues)), eigenvalues, marker='o')
    plt.title("Eigenvalues of the Laplacian")
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.grid(True)
    plt.show()
    answers["eigenvalue plot"] = plot_eig

    # Pick the parameters that give the largest value of ARI, and apply these
    # parameters to datasets 1, 2, 3, and 4. Compute the ARI for each dataset.
    # Calculate mean and standard deviation of ARI for all five datasets.

    # A single float
    answers["mean_ARIs"] = np.mean(all_ARIs)

    # A single float
    answers["std_ARIs"] = np.std(all_ARIs)

    # A single float
    answers["mean_SSEs"] = np.mean(all_SSEs)

    # A single float
    answers["std_SSEs"] = np.std(all_SSEs)

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    all_answers = spectral_clustering()
    with open("spectral_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)