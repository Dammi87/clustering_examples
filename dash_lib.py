"""Contains important libraries to be used with the dash app."""
from dash.dependencies import Input, Output, State
import dash_html_components as html
from sklearn.preprocessing import StandardScaler
import dash_core_components as dcc
from server import app
import pandas as pd
import logging
from datacleaner import autoclean
import os
import numpy as np
import importlib


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(name='dash-lib')


class SliderStates():
    """SliderStates takes care of creating sliders and handling their callbacks."""

    def __init__(self):
        """Initialize slider container."""
        self._values = {}

    def add(self, slider_name, value, title=None):
        """Add a slider to the SliderStates class.

        If this is the first time this slider is added, a callback is created

        Parameters
        ----------
        slider_name : str
            Name of the slider, <CLASS_NAME>#ID#<SLIDER_NAME>
        value : list
            Value that the slider should take
        title : None, str
            Tooltip text of the slider, should contain %d / %f etc which "value" will
            be used for format with.
        """
        algo, val_name = slider_name.split('#ID#')
        if algo not in self._values:
            self._values[algo] = {}
        if isinstance(value, list) and len(value) == 1:
            value = value[0]

        is_first = False
        if val_name not in self._values[algo]:
            is_first = True

        # Set value
        self._values[algo][val_name] = value

        # If this is the first time this value is created, then
        # create the callback
        if is_first:
            text_id = 'text_%s_%s' % (algo, val_name)

            @app.callback(Output(text_id, 'children'), [Input(slider_name, 'value')])
            def show_hover(value, default=title, slider_id=slider_name):
                if value is None:
                    return default % -1
                _, val_name = slider_id.split('#ID#')
                self.add(slider_id, value)
   
   
                if isinstance(value, int):
                    return default % value
                else:
                    return default % tuple(value)

    def get(self, algo):
        """Get the slider values for the desired algorithm.

        Parameters
        ----------
        algo : str
            <CLASS_NAME> of the algorithm

        Returns
        -------
        dict
            Returns a dictionary with the settings
        """
        params = {}
        for key in self._values[algo]:
            params[key] = self._values[algo][key]
        return params

    def get_as_states(self):
        """Return the value of sliders as states."""
        for algo in self._values:
            states = []
            for key in self._values[algo]:
                slider_id = '%s#ID#%s' % (algo, key)
                states.append(State(slider_id, 'value'))
        return states

    def get_as_inputs(self):
        """Return the value of sliders as states."""
        for algo in self._values:
            inputs = []
            for key in self._values[algo]:
                slider_id = '%s#ID#%s' % (algo, key)
                inputs.append(Input(slider_id, 'value'))
        return inputs


def get_default_style(**kwarg):
    """Default DIV style used."""
    style = {
        'width': '50%',
        'height': '30px',
        'lineHeight': '30px',
        'borderWidth': '1px',
        'borderStyle': 'dashed',
        'borderRadius': '5px',
        'textAlign': 'center',
        'margin': '10px',
        'display': 'inline-block'
    }
    if len(kwarg) > 0:
        for key in kwarg:
            style[key] = kwarg[key]
    return style


class Dataset():
    """Class that takes care of any dataset related tasks."""

    def __init__(self):
        """Init class."""
        self._column_names = ['col_1', 'col_2', 'col_3']
        self._active_x = None
        self._active_y = None
        self._active_z = None
        self._clicks = 0
        # Create a callback that sets the column names if process files is pressed

        for col_nr, ax in enumerate(zip(['x', 'y', 'z'])):
            @app.callback(Output('%s-axis' % ax, 'options'), [Input('figure_1', 'children')])
            def update_axis(_, dataset_object=self):
   
                options = [{'label': col, 'value': i}
                           for i, col in enumerate(self._column_names)]

                return options

            @app.callback(Output('%s-axis' % ax, 'value'), [Input('%s-axis' % ax, 'options')])
            def update_axis_selected(_, col_nr=col_nr, self=self, ax=ax):
   
                return col_nr

            @app.callback(Output('hidden-div-%s' % ax, 'title'),
                          [Input('%s-axis' % ax, 'options'),
                           Input('%s-axis' % ax, 'value')])
            def update_active_column(options, value, self=self, ax=ax):
                if value is None:
                    return None
                key_sel = options[value]['label']
                setattr(self, '_active_%s' % ax, key_sel)
   
                return None

    def load_dataset(self, file_name):
        """Load the dataset into memory."""
        self._data = autoclean(pd.read_hdf(
            './uploads/%s.hdf' % file_name), drop_nans=True)
        self._column_names = self._data.columns.values

        self._active_x = self._column_names[0]
        self._active_y = self._column_names[1]
        if len(self._column_names) > 2:
            self._active_z = self._column_names[2]

    def get_data(self):
        """Get all the data."""
        return self._data.values

    def get_active(self):
        """Get the data according to column names."""
        col = [self._active_x]
        if self._active_y is not None:
            col.append(self._active_y)
        if self._active_z is not None:
            col.append(self._active_x)
        return self._data[col].values.astype(np.float32)


class Clustering():
    """Class that contains all available algorithms."""

    def __init__(self):
        """Init."""
        self._set_algos()
        self._set_clustering_options()
        self._setup_sliders()

    def _set_algos(self):
        """Fetch all exposed algorithms."""
        folders = os.listdir('./algorithms')
        package = __import__('algorithms')
        self._algorithms = {}
        for module_name in dir(package):
            if module_name.startswith("__"):
                continue
            if module_name in folders:
                continue
            # Import
            module = importlib.import_module('algorithms')
            method = getattr(module, module_name)

            self._algorithms[module_name] = method()

    def _set_clustering_options(self):
        """Fetch all options for all algorithms."""
        pre_opt = {}
        post_opt = {}
        for algo in self._algorithms:
            pre_opt[algo] = self._algorithms[algo].get_pre_slider_options()
            post_opt[algo] = self._algorithms[algo].get_post_slider_options()

        self._pre_slider_options = pre_opt
        self._post_slider_options = post_opt

    def _setup_sliders(self):
        """Setup the sliders."""
        self._sliders_pre = SliderStates()
        all_options = self.get_pre_options()
        for key_sel in all_options:
            options = all_options[key_sel]
            for opt in options:
                slider_id = '%s#ID#%s' % (key_sel, opt)
   
                self._sliders_pre.add(
                    slider_id,
                    options[opt]['settings']['value'],
                    options[opt]['title'])

        self._sliders_post = SliderStates()
        all_options = self.get_post_options()
        for key_sel in all_options:
            options = all_options[key_sel]
            for opt in options:
                slider_id = '%s#ID#%s' % (key_sel, opt)
   
                self._sliders_post.add(
                    slider_id,
                    options[opt]['settings']['value'],
                    options[opt]['title'])

    def get_algorithms_dropdown_options(self):
        """Return a list that is used to fill in dropdown menu."""
        return [{'label': algo, 'value': 0} for i, algo in enumerate(self._pre_slider_options)]

    def get_class(self, algo):
        """Return the initiialized algorithm."""
        return self._algorithms[algo]

    def get_pre_options(self, algo=None):
        """Return pre run slider options."""
        if algo is None:
            return self._pre_slider_options
        return self._pre_slider_options[algo]

    def get_post_options(self, algo=None, **kwargs):
        """Return post run slider options."""
        if algo is None:
            return self._post_slider_options
        return self._post_slider_options[algo]

    def create_sliders(self, algo, slider_type='pre', **kwargs):
        """Create the DIV of a certain algorithm."""
        if slider_type is 'pre':
            options = self.get_pre_options(algo)

        elif slider_type is 'post':
            options = self.get_post_options(algo, **kwargs)

        # Get current values of sliders for use in dependency cases
        all_values = merge_two_dicts(self._sliders_pre.get(algo), self._sliders_post.get(algo))

        divs = []
        for opt in options:
            if 'dependency' not in options[opt]:
                slider = options[opt]['settings']
            else:
                # No settings, slider depends on another slider
                other_slider = all_values[options[opt]['dependency']]
                slider = options[opt]['method'](other_slider)

            slider_id = '%s#ID#%s' % (algo, opt)
            div_id = '%s#DIV#%s' % (algo, opt)
            text_id = 'text_%s_%s' % (algo, opt)

            item = get_slider_from_settings(slider, slider_id)
            divs.append(html.Div(id=text_id))
            divs.append(html.Div(item, id=div_id))

        return html.Div(divs)


def get_slider_from_settings(slider_settings, _id):
    """Given slider settings for particular slider, return compnonent."""
    if slider_settings['type'] == 'range':
        item = dcc.RangeSlider(
            count=1,
            min=slider_settings['min'],
            max=slider_settings['max'],
            step=slider_settings['step'],
            value=slider_settings['value'],
            allowCross=False,
            id=_id
        )

    if slider_settings['type'] == 'value':
        item = dcc.Slider(
            min=slider_settings['min'],
            max=slider_settings['max'],
            step=slider_settings['step'],
            value=slider_settings['value'],
            id=_id
        )

    return item


def upload_message(msg):
    """Method that creates a DIV with a message."""
    return html.Div(
        [msg],
        style=get_default_style(
            borderStyle=None,
            margin=None))


def load_selected_file(file_name):
    """Load the selected file, make sure to clean it up before use.

    Parameters
    ----------
    file_name : str
        File name to load

    Returns
    -------
    pandas.dataframe
    """
    return autoclean(pd.read_hdf('./uploads/%s.hdf' % file_name), drop_nans=True)


def scale_data_array(data_array):
    """Scale data such that it has 0 mean and 1 variance."""
    z_scaler = StandardScaler()
    data_array = z_scaler.fit_transform(data_array)

    return data_array


def get_uploaded_files():
    """Get a list of uploaded files available."""
    files = os.listdir('uploads')
    if len(files) == 0:
        return {'label': None, 'value': 0}
    files = [file.replace('.hdf', '') for file in files]
    return [{'label': label, 'value': i} for i, label in enumerate(files)]


def axis_dropdowns():
    """Create Axis drop down menus."""
    htmls = []
    for ax in ['x', 'y', 'z']:
        htmls.append(html.Div([
            dcc.Dropdown(
                id='%s-axis' % ax,
                options=[{'label': 'col1', 'value': 0}],
                clearable=False,
                placeholder="%s-Axis" % ax
            )
        ],
            className='axis-dropdowns',
            style=get_default_style(
                width='33.3%',
                margin='0px',
                borderStyle=None),
        ))

    return htmls


def merge_two_dicts(x, y):
    """Python 2 dictionary merging."""
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None

    return z
