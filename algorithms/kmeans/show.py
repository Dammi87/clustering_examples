"""Contains methods and classes used for visualization."""
from .lib import KMean
import numpy as np
import dash_core_components as dcc
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(name='plotting')


class KMeanDash(KMean):
    """Wrapper over KMEans algorithm, takes care of generating plotly data."""

    def __init__(self):
        """Init."""
        KMean.__init__(self)

    def _is_run(self):
        """Return True if exploration has been done."""
        return hasattr(self, '_dash_data')

    def get_figure_1(self, style):
        """Figure 1 will show the score for each cluster size."""
        if not self._is_run():
            return False

        y_data = [sa.mean() for sa in self._dash_data['sa_analysis']]
        x_data = self._dash_data['k_clusters']
        data = np.stack([x_data, y_data]).transpose()
        plt_data = arrays_to_plotly_data(
            array=data,
            name='Silhouette mean scores')
        return dcc.Graph(
            style=style,
            id='basic-interactions',
            figure={
                'data': [plt_data],
                'layout': {
                    'title': 'Silhouette mean scores'
                }
            }
        )

    def get_figure_2(self, style, n_cluster=None):
        """Figure 2 will show the current clustering chosen."""
        if not self._is_run():
            return False

        if n_cluster is None or n_cluster not in self._dash_data['k_clusters']:
            n_cluster = self._dash_data['optimal_size']

        show_cluster = self._dash_data['k_clusters'].index(n_cluster)
        cluster_ids = self._dash_data['collected_clusters'][:, show_cluster]
        data = []
        for _id in np.unique(cluster_ids):
            this_cluster = self._data[cluster_ids == _id, :]
            _data = arrays_to_plotly_data(this_cluster, name='%d' % _id)
            data.append(_data)

        return dcc.Graph(
            style=style,
            id='figure_2_ChosenClustering',
            figure={
                'data': data,
                'layout': {
                    'title': 'Cluster results for %d clusters' % n_cluster
                }
            }
        )

    def explore_dash(self, **kwargs):
        """Wrap KMEans explore."""
        k_clusters, sa_analysis, optimal_size, collect_clusters = self.explore(
            **kwargs)

        self._dash_data = {
            'k_clusters': k_clusters,
            'sa_analysis': sa_analysis,
            'optimal_size': optimal_size,
            'collected_clusters': collect_clusters
        }

    @staticmethod
    def get_pre_slider_options():
        """Return slider options for initializing algorithm, used by DASH app to render."""
        return {
            'k_range': {
                'settings': create_slider_option(2, 100, [5, 10], slider_type='range'),
                'title': 'Cluster ranges [%d, %d]'
            },
            'n_kmean_iters': {
                'settings': create_slider_option(2, 100, [25], slider_type='value'),
                'title': 'How many attempts per cluster: %d'
            },
            'n_max_iter': {
                'settings': create_slider_option(25, 500, [50], slider_type='value'),
                'title': 'Maximum convergence iterations %d'
            }
        }

    @staticmethod
    def get_post_slider_options():
        """Return slider options for controlling plots, used by DASH app to render."""

        def n_cluster(k_range):
            return create_slider_option(k_range[0], k_range[1], [k_range[0]], slider_type='value')

        return {
            'n_cluster': {
                'settings': create_slider_option(2, 100, [5], slider_type='value'),
                'title': 'Cluster rendered: %d',
                'dependency': 'k_range',
                'method': n_cluster
            },
        }


def arrays_to_plotly_data(array, axis_names=None, name=None, marker={'size': 12}, mode='markers'):
    """Create plotly data from array."""
    plt_data = {}
    if array.shape[1] > 2:
        plt_data['type'] = 'scatter3d'

    if name is not None:
        plt_data['name'] = name
    plt_data['marker'] = marker
    plt_data['mode'] = mode

    for ax, column in zip(['x', 'y', 'z'], range(array.shape[1])):
        plt_data[ax] = array[:, column]

    return plt_data


def create_slider_option(min_val, max_val, value, step_size=1, slider_type='range'):
    """Create DASH slider option format."""
    assert slider_type in ['range', 'value']
    assert isinstance(value, list)
    if slider_type == 'range':
        assert len(value) == 2
    return {
        'min': min_val,
        'max': max_val,
        'value': value,
        'step': step_size,
        'type': slider_type
    }
