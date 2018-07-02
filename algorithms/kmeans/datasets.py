"""Create datasets for exploring k-means with."""
import numpy as np
import matplotlib.pyplot as plt
from lib import KMean
from scipy import random


def create_dataset(n_clusters, n_dim, n_samples, mean=None, std=None, seed=None):
    """Create n_clusters clusters of n_dim dimensions, n_samples of it."""
    if seed is not None:
        np.random.seed(seed)

    if mean is None:
        means = [np.random.randint(-(i + 1), (i + 1),
                                   n_dim).astype(np.float32) for i in range(n_clusters)]
    if std is None:
        stds = []
        for _ in range(n_clusters):
            a = np.diag(random.rand(n_dim))
            stds.append(a)

    data_set = []
    cluster_id = []
    for i, mean, std in zip(range(n_clusters), means, stds):
        data_set.append(np.random.multivariate_normal(mean, std, n_samples))
        cluster_id.append(np.ones(n_samples) * i)

    data_set = np.concatenate(data_set)
    cluster_id = np.concatenate(cluster_id)
    sample_idx = np.random.choice(data_set.shape[0], n_samples, replace=False)

    # Return correct amount of samples
    return data_set[sample_idx, :], cluster_id[sample_idx]


def plot_clusters(ax, data_points, cluster_ids):
    """Plot the data points according to cluster_ids."""
    cluster_handles = []
    for cluster_id in np.unique(cluster_ids):
        loc = cluster_id == cluster_ids
        cluster_handles.append(
            ax.plot(data_points[loc, 0], data_points[loc, 1], '.')[0])
        mean = np.mean(data_points[loc, ::], axis=0)
        ax.plot(mean[0], mean[1], 'kx')

    return cluster_handles


def update_plot_clusters(cluster_handles, data_points, cluster_ids):
    """Update cluster plots according to the newcluster_ids."""
    for handle, cluster_id in zip(cluster_handles, np.unique(cluster_ids)):
        loc = cluster_id == cluster_ids
        handle.set_data(data_points[loc, 0], data_points[loc, 1])


if __name__ == "__main__":
    import pandas as pd
    N_CLUSTERS = 10

    # data_set, cluster_ids = create_dataset(N_CLUSTERS, 2, 5000, seed=53)
    data_set = pd.read_hdf('/home/adamf/repos/machine_learning_examples/uploads/random_file_2.hdf').values
    cluster_ids = np.ones(data_set.shape[0])

    kmean = KMean(data_set, N_CLUSTERS)
    kernels, silhouettes, optimal_k, collect_clusters = kmean.explore(
        k_range=[5, 43], n_kmean_iters=15, n_max_iter=40)

    kmean.get_plotly(collect_clusters, kernels.index(optimal_k))

    fig = plt.figure(dpi=150)
    for i, (k, sa) in enumerate(zip(kernels, silhouettes)):
        ax1 = fig.add_subplot(len(kernels), 1, i + 1)
        plt.plot(sa)
        plt.plot([0, len(sa)], [sa.mean()] * 2)
        # Calculate how many are ABOVE the average
        perc_above = np.sum(sa >= sa.mean()).astype(np.float32) / len(sa)
        plt.title('Number of kernels: %d\nAverage: %1.4f\nAbove: %1.4f' %
                  (k, sa.mean(), perc_above))
        plt.axis('off')
    plt.xlabel('Kernels')
    plt.ylabel('Variance')

    # Fit the optimal
    kmean = KMean(data_set, optimal_k)
    kmean.fit(n_kmean_iters=50, n_max_iter=50)
    print("Optimal cluster size is %d" % optimal_k)

    # plt.ion()
    fig = plt.figure(dpi=150)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    plot_clusters(ax1, data_set, cluster_ids)
    cluster_handles = plot_clusters(ax2, kmean._data, kmean._cluster_id)

    plt.show()
