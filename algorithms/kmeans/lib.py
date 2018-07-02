"""Classes and methods needed for KMeans clustering."""
import numpy as np
from scipy.spatial import distance


class KMean():
    """Summary."""

    def __init__(self, data=None, n_clusters=10):
        """Summary.

        Parameters
        ----------
        data : TYPE
            Description
        """
        if data is not None:
            self.input(data, n_clusters)

    def input(self, data, n_clusters=10):
        """Input data."""
        self._data = data
        self._n_clusters = n_clusters
        self._n_data = data.shape[0]
        self._data_idx = np.arange(self._n_data)
        self._init_clusters()

    def _get_random_cluster(self, weights=None):
        """Select a datapoint as a cluster randomly, scaled with weights."""
        idx = np.random.choice(self._data_idx, p=weights)
        return self._data[idx, :]

    def _add_cluster(self, cluster_center):
        """Summary.

        Parameters
        ----------
        cluster_center : TYPE
            Description
        """
        self._clusters[self._i_cluster, :] = cluster_center
        self._i_cluster += 1

    def _update_cluster(self, cluster_id, new_center, new_var):
        """Update a specific cluster id."""
        self._clusters[cluster_id, :] = new_center
        self._clusters_var[cluster_id, :] = new_var

    def _get_distance(self):
        """Calculate shortest distance from data to a cluster."""
        # Calculate euclidan distance, returns a [self._i_cluster x self._data.shape[0]]
        # sized array.
        dist = distance.cdist(
            self._clusters[:self._i_cluster, :],
            self._data)

        idx = np.argmin(dist, axis=0)

        return dist[idx, self._data_idx], idx

    def _get_weights_by_distance(self):
        """Calculate weights proportinal to the distance squared."""
        min_dist, _ = self._get_distance()
        min_dist_2 = np.square(min_dist)
        min_dist_2_sum = min_dist_2.sum()
        if min_dist_2_sum == 0:
            return None

        return np.divide(min_dist_2, min_dist_2.sum()).flatten()

    def _update_cluster_membership(self):
        """Calculate which cluster a datapoint belongs too."""
        _, cluster_id = self._get_distance()
        self._cluster_id = cluster_id

    def _update_cluster_centers(self):
        """Update cluster centers."""
        # First update cluster memberships
        self._update_cluster_membership()

        # Loop and update
        for cluster in range(self._n_clusters):
            new_center = np.mean(
                self._data[self._cluster_id == cluster, :], axis=0)
            new_var = np.var(
                self._data[self._cluster_id == cluster, :], axis=0)
            self._update_cluster(cluster, new_center, new_var)

    def _init_clusters(self):
        """Initialize the clusters."""
        self._clusters = np.zeros((self._n_clusters, self._data.shape[1]))
        self._clusters_var = np.zeros((self._n_clusters, self._data.shape[1]))
        self._i_cluster = 0

        # Weights are first simply uniformly chosen
        weights = np.divide(
            np.ones(self._data.shape[0], dtype=np.float32), self._data.shape[0])
        for _ in range(self._n_clusters):
            # Add a cluster using the weights
            self._add_cluster(self._get_random_cluster(weights))
            # Calculate new weights
            weights = self._get_weights_by_distance()

        self._update_cluster_membership()

    def _get_fit_indicator(self):
        # return np.mean(silhouette_analysis(self._data, self._clusters, self._cluster_id))
        return np.mean(self._clusters_var)

    def _plot_update(self, fig, handles=None):
        """Update cluster plots."""
        if handles is None:
            return

        # If cluster_plot_handle is not None, use it to plot
        if handles is not None:
            for h, i, _mean in zip(handles, range(self._n_clusters), self._clusters):
                loc = i == self._cluster_id
                h.set_data(self._data[loc, 0], self._data[loc, 1])
            fig.canvas.draw()

    def fit(self, n_kmean_iters, n_max_iter, eps=1.0 / 1000, fig=None, cluster_plot_handle=None):
        """Start optimizing the clusters, using n_iterations and stopping at n_eps."""
        best_indicator = np.inf
        for _ in range(n_kmean_iters):
            # Re-Initialize
            self._init_clusters()

            for _ in range(n_max_iter):
                # Get last centeroids
                last_centers = self._clusters.copy()
                # Update cluster
                self._update_cluster_centers()
                # Check for change
                diff = np.sum(np.abs(last_centers - self._clusters))
                # Update plots
                self._plot_update(fig, cluster_plot_handle)
                # Break if eps reached
                if diff <= eps:
                    print("A difference of %1.5f reached" % diff)
                    print("\tIndicator is %1.5f" % self._get_fit_indicator())
                    break

            # Update best clusters and var
            if best_indicator > self._get_fit_indicator():
                print("Indicator before %1.4f and now %1.4f" %
                      (best_indicator, self._get_fit_indicator()))
                best_clusters = self._clusters.copy()
                best_indicator = self._get_fit_indicator()

        # End reached, set the clusters
        self._clusters = best_clusters
        self._update_cluster_membership()
        self._plot_update(fig, cluster_plot_handle)

    def explore(self, k_range, n_kmean_iters, n_max_iter,
                eps=1.0 / 1000, fig=None, cluster_plot_handle=None):
        """Explore different number of kernels and return the result."""
        assert k_range[0] > 1

        # If k_range is >= n_sample, decrease
        if k_range[1] > self._data.shape[0]:
            k_range[1] = self._data.shape[0] - 1

        sa_analysis = []
        k_clusters = list(range(k_range[0], k_range[1] + 1))
        collect_clusters = -np.ones(shape=(self._data.shape[0], len(k_clusters)))

        for i, k in enumerate(k_clusters):
            # Initialize class
            self.input(self._data, k)
            # Start the fit
            self.fit(n_kmean_iters, n_max_iter, eps, fig, cluster_plot_handle)
            # Collect silhouette scores
            sa_analysis.append(silhouette_analysis(self._data, self._clusters, self._cluster_id))
            # Collect cluster results
            collect_clusters[:, i] = self._cluster_id.copy()

        # Collect the means
        means = np.array([sa.mean() for sa in sa_analysis])

        # Return optimal cluster size according to largest mean
        optimal_size = k_clusters[np.where(means == means.max())[0][0]]

        return k_clusters, sa_analysis, optimal_size, collect_clusters

    def get_clustering_for_plotly(self, collected_clusters, get_cluster):
        """Given data from explore and desired cluster, return plotly data."""
        cluster_ids = collected_clusters[:, get_cluster]
        data = []
        for _id in np.unique(cluster_ids):
            this_cluster = self._data[cluster_ids == _id, :]
            _data = arrays_to_plotly_data(this_cluster, name='%d' % _id)
            data.append(_data)

        return data

    def get_silhouette_for_plotly(self, sa_analysis):
        """Given data from explore and desired cluster, return plotly data."""
        cluster_ids = collected_clusters[:, get_cluster]
        data = []
        for _id in np.unique(cluster_ids):
            this_cluster = self._data[cluster_ids == _id, :]
            _data = arrays_to_plotly_data(this_cluster, name='%d' % _id)
            data.append(_data)

        return data


def arrays_to_plotly_data(array, axis_names=None, name=None, marker={'size': 12}):
    shape = array.shape
    if shape[1] > 2:
        plot_type = 'scatter3d'
    else:
        plot_type = 'scatter2d'

    plt_data = {}
    if name is not None:
        plt_data['name'] = name
    plt_data['marker'] = marker
    plt_data['type'] = plot_type
    plt_data['mode'] = 'markers'

    for ax, column in zip(['x', 'y', 'z'], range(array.shape[1])):
        plt_data[ax] = array[:, column].tolist()

    return plt_data


def silhouette_analysis(data, cluster_means, cluster_ids):
    """Perform silhouette analysis."""
    n_data = data.shape[0]

    # Calculate the distance from all datapoints to all clusters
    dist = distance.cdist(cluster_means, data)

    # Find the locations of the min and replace them with larger numbers,
    # we are after the SECOND smallest mean
    idx = np.argmin(dist, axis=0)
    dist[idx, range(n_data)] = np.inf
    idx = np.argmin(dist, axis=0)
    second_min_dist = dist[idx, range(n_data)].copy()

    # Now, for each cluster membership, calculate the AVERAGE distance
    # from each sample, to the cluster it belongs to
    sa = []
    for _id in np.sort(np.unique(cluster_ids)):
        loc = cluster_ids == _id

        # Calculate the distance from any one point, to all other points in this cluster
        intra_distance = distance.cdist(data[loc, ::], data[loc, ::], 'euclidean')
        intra_mean = np.divide(np.sum(intra_distance, axis=0), loc.sum())

        # Now calcualte the distance from the points in this cluster to the closest
        # cluster it's not apart of
        second_closest = second_min_dist[loc]

        # Set the value
        sa_val = np.divide(second_closest - intra_mean, np.maximum(intra_mean, second_closest))
        sa.append(np.sort(sa_val))

    return np.concatenate(sa)
