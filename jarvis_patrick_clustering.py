# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 09:30:40 2024

@author: Evelina
"""

"""
Work with Jarvis-Patrick clustering.
Do not use global variables!
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pickle
from scipy.spatial import KDTree
from scipy.special import comb

######################################################################
#####     CHECK THE PARAMETERS     ########
######################################################################


def compute_k_nearest_neighbors(data, k):
    tree = KDTree(data)
    _, indices = tree.query(data, k=k+1)  # k+1 because the point itself is included
    return indices[:, 1:]  # Exclude the point itself

def count_shared_neighbors(indices):
    num_points = indices.shape[0]
    shared_neighbors = np.zeros((num_points, num_points), dtype=int)

    for i in range(num_points):
        neighbors_i = set(indices[i])
        for j in range(num_points):
            if i != j:
                neighbors_j = set(indices[j])
                shared_neighbors[i, j] = len(neighbors_i.intersection(neighbors_j))

    return shared_neighbors

def form_clusters(shared_neighbors, smin):
    num_points = shared_neighbors.shape[0]
    labels = -np.ones(num_points, dtype=int)
    cluster_id = 0

    for i in range(num_points):
        if labels[i] == -1:  # Point not yet assigned to a cluster
            seen = set()
            stack = [i]
            while stack:
                current = stack.pop()
                if current not in seen:
                    seen.add(current)
                    labels[current] = cluster_id
                    stack.extend([j for j in range(num_points) if shared_neighbors[current, j] >= smin and labels[j] == -1])
            cluster_id += 1

    return labels

def calculate_sse(data, labels):
    unique_labels = np.unique(labels)
    sse = 0
    for label in unique_labels:
        cluster_data = data[labels == label]
        centroid = np.mean(cluster_data, axis=0)
        sse += np.sum((cluster_data - centroid) ** 2)
    return sse

def adjusted_rand_index(labels_true, labels_pred):
    # Find the unique classes
    classes_true = np.unique(labels_true)
    classes_pred = np.unique(labels_pred)

    # Create the contingency table
    contingency_matrix = np.zeros((len(classes_true), len(classes_pred)), dtype=int)
    for i, class_true in enumerate(classes_true):
        for j, class_pred in enumerate(classes_pred):
            contingency_matrix[i, j] = np.sum((labels_true == class_true) & (labels_pred == class_pred))

    # Sum over rows and columns
    sum_rows = np.sum(contingency_matrix, axis=1)
    sum_cols = np.sum(contingency_matrix, axis=0)
    total = np.sum(contingency_matrix)

    # Compute combinations of sums
    comb_sum_rows = comb(sum_rows, 2)
    comb_sum_cols = comb(sum_cols, 2)
    comb_contingency_matrix = comb(contingency_matrix, 2)
    
    # Calculate the terms needed for the Rand Index formula
    sum_comb = np.sum(comb_contingency_matrix)
    expected_index = np.sum(comb_sum_rows) * np.sum(comb_sum_cols) / comb(total, 2)
    max_index = 0.5 * (np.sum(comb_sum_rows) + np.sum(comb_sum_cols))

    # Compute the Adjusted Rand Index
    ari = (sum_comb - expected_index) / (max_index - expected_index)
    return ari

def plot_clustering(data, labels, title, cmap='viridis', alpha=0.5):
    """
    Plots the clustering assignments for a given dataset and its labels.
    
    Parameters:
        data (NDArray[np.floating]): The dataset (usually 2D for visualization).
        labels (NDArray[np.int32]): The cluster labels for the dataset.
        title (str): Title for the plot.
        cmap (str): Color map for the plot. Default is 'viridis'.
        alpha (float): Transparency of the points. Default is 0.5.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=cmap, alpha=alpha)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Cluster Label')
    plt.grid(True)
    plt.show()

def jarvis_patrick(
    data: NDArray[np.floating], labels: NDArray[np.int32], params_dict: dict
) -> tuple[NDArray[np.int32] | None, float | None, float | None]:
    """
    Implementation of the Jarvis-Patrick algorithm only using the `numpy` module.

    Arguments:
    - data: a set of points of shape 50,000 x 2.
    - dict: dictionary of parameters. The following two parameters must
       be present: 'k', 'smin', There could be others.
    - params_dict['k']: the number of nearest neighbors to consider. This determines the size of the neighborhood used to assess the similarity between datapoints. Choose values in the range 3 to 8
    - params_dict['smin']:  the minimum number of shared neighbors to consider two points in the same cluster.
       Choose values in the range 4 to 10.

    Return values:
    - computed_labels: computed cluster labels
    - SSE: float, sum of squared errors
    - ARI: float, adjusted Rand index

    Notes:
    - the nearest neighbors can be bidirectional or unidirectional
    - Bidirectional: if point A is a nearest neighbor of point B, then point B is a nearest neighbor of point A).
    - Unidirectional: if point A is a nearest neighbor of point B, then point B is not necessarily a nearest neighbor of point A).
    - In this project, only consider unidirectional nearest neighboars for simplicity.
    - The metric  used to compute the the k-nearest neighberhood of all points is the Euclidean metric
    """

    # Extract the parameters
    k = params_dict['k']
    smin = params_dict['smin']

    # Step 1: Compute k-nearest neighbors
    indices = compute_k_nearest_neighbors(data, k)

    # Step 2: Count shared neighbors
    shared_neighbors = count_shared_neighbors(indices)

    # Step 3: Form clusters
    computed_labels = form_clusters(shared_neighbors, smin)

    # Step 4: Calculate SSE
    SSE = calculate_sse(data, computed_labels)

    # Step 5: Calculate ARI
    ARI = adjusted_rand_index(labels, computed_labels)

    
    # computed_labels: NDArray[np.int32] | None = None
    # SSE: float | None = None
    # ARI: float | None = None

    return computed_labels, SSE, ARI


def jarvis_patrick_clustering():
    """
    Performs Jarvis-Patrick clustering on a dataset.

    Returns:
        answers (dict): A dictionary containing the clustering results.
    """
    # Load the data
    data = np.load('question1_cluster_data.npy')
    labels_true = np.load('question1_cluster_labels.npy')

    answers = {}

    # Return your `jarvis_patrick` function
    answers["jarvis_patrick_function"] = jarvis_patrick

    # Work with the first 5,000 data points: data[0:5000]
    # Do a parameter study of this data using Jarvis-Patrick.
    # Minimmum of 10 pairs of parameters ('sigma' and 'xi').
    
    k_values = [3, 4, 5, 6, 7, 8]
    smin_values = [4, 5, 6, 7, 8, 9, 10]
    results = []
    
    for k in k_values:
        for smin in smin_values:
            params_dict = {'k': k, 'smin': smin}
            computed_labels, SSE, ARI = jarvis_patrick(data[0:1000], labels_true[0:1000], params_dict)
            results.append((k, smin, SSE, ARI))
            
            # # Plot clustering after each run
            # plot_title = f'Clustering (k={k}, smin={smin})'
            # plot_clustering(data[0:5000], computed_labels, plot_title)

    # Create a dictionary for each parameter pair ('sigma' and 'xi').
    groups = {}

    # data for data group 0: data[0:5000]. For example,
    # groups[0] = {"sigma": 0.1, "xi": 0.1, "ARI": 0.1, "SSE": 0.1}

    # data for data group i: data[5000*i: 5000*(i+1)], i=1, 2, 3, 4.
    # For example,
    # groups[i] = {"sigma": 0.1, "xi": 0.1, "ARI": 0.1, "SSE": 0.1}
    
    # Select the best and worst according to ARI and SSE
    best_ari = max(results, key=lambda x: x[3])
    worst_sse = min(results, key=lambda x: x[2])

    # Visualization of SSE and ARI
    k_values, smin_values, SSEs, ARIs = zip(*results)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    plot_SSE = axs[0].scatter(k_values, smin_values, c=SSEs, cmap='viridis')
    fig.colorbar(plot_SSE, ax=axs[0], label='SSE')
    axs[0].set_xlabel('k')
    axs[0].set_ylabel('smin')
    axs[0].set_title('Parameter Scatter Plot (SSE)')
    axs[0].grid(True)

    plot_ARI = axs[1].scatter(k_values, smin_values, c=ARIs, cmap='plasma')
    fig.colorbar(plot_ARI, ax=axs[1], label='ARI')
    axs[1].set_xlabel('k')
    axs[1].set_ylabel('smin')
    axs[1].set_title('Parameter Scatter Plot (ARI)')
    axs[1].grid(True)

    plt.show()
        
    # Group parameters and results
    groups = {f'Group {i}': {'sigma': k, 'xi': smin, 'SSE': SSE, 'ARI': ARI}
              for i, (k, smin, SSE, ARI) in enumerate(results)}
    answers["cluster parameters"] = groups
    answers["1st group, SSE"] = groups['Group 0']['SSE']  # Assuming the first result corresponds to the first group

    

    # Create two scatter plots using `matplotlib.pyplot`` where the two
    # axes are the parameters used, with # \sigma on the horizontal axis
    # and \xi and the vertical axis. Color the points according to the SSE value
    # for the 1st plot and according to ARI in the second plot.

    # Choose the cluster with the largest value for ARI and plot it as a 2D scatter plot.
    # Do the same for the cluster with the smallest value of SSE.
    # All plots must have x and y labels, a title, and the grid overlay.

    

    # Pick the parameters that give the largest value of ARI, and apply these
    # parameters to datasets 1, 2, 3, and 4. Compute the ARI for each dataset.
    # Calculate mean and standard deviation of ARI for all five datasets.
    
    # Mean and standard deviation of ARIs and SSEs for subsequent applications
    # Assume we are applying the best ARI parameters across the next five data slices
    ARIs = []
    SSEs = []
    for i in range(5):
        data_slice = data[i * 1000:(i + 1) * 1000]
        labels_slice = labels_true[i * 1000:(i + 1) * 1000]
        _, SSE, ARI = jarvis_patrick(data_slice, labels_slice, {'k': best_ari[0], 'smin': best_ari[1]})
        ARIs.append(ARI)
        SSEs.append(SSE)
        
    # A single float
    answers["mean_ARIs"] = np.mean(ARIs)

    # A single float
    answers["std_ARIs"] = np.std(ARIs)

    # A single float
    answers["mean_SSEs"] = np.mean(SSEs)

    # A single float
    answers["std_SSEs"] = np.std(SSEs)

    return answers




# ----------------------------------------------------------------------
if __name__ == "__main__":
    all_answers = jarvis_patrick_clustering()
    with open("jarvis_patrick_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)