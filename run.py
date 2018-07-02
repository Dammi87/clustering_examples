"""Example script."""
import base64
import io
import logging
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from server import app
import dash_lib

# Get logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(name='testing')

# Get cluster algorithms
cluster_algos = dash_lib.Clustering()

# Dataset class
dataset_loader = dash_lib.Dataset()


# Layout
app.layout = html.Div([
    dcc.Upload(id='upload-data',
               children=html.Div(['Drag and Drop or ',
                                  html.A('Select Files')
                                  ]),
               style=dash_lib.get_default_style(),
               multiple=True
               ),
    html.Div(id='output-data-upload'),
    html.Div([html.Div([
        dcc.Dropdown(id='opt-cluster-method',
                     options=cluster_algos.get_algorithms_dropdown_options(),
                     clearable=False,
                     placeholder="Clustering method"
                     )
    ],
        className='two columns',
        style=dash_lib.get_default_style(width='50%',
                                         margin='0px',
                                         borderStyle=None),
    ), html.Div([
        dcc.Dropdown(id='opt-uploaded-files',
                     options=dash_lib.get_uploaded_files(),
                     clearable=False,
                     placeholder="Select a file to process"
                     )
    ],
        className='two columns',
        style=dash_lib.get_default_style(width='50%',
                                         margin='0px',
                                         borderStyle=None),
    )],
        style=dash_lib.get_default_style(width='50%',
                                         borderStyle=None),),
    html.Button('Process File',
                id='button:process-file',
                style=dash_lib.get_default_style(width='50%',
                                                 borderStyle=None)
                ),
    html.Div(id='empty_div2',
             style=dash_lib.get_default_style(width='15%',
                                              borderStyle=None,
                                              display=None)),
    html.Div(id='hidden-div', style={'display': None}),
    html.Div(id='hidden-div-x', style={'display': None}),
    html.Div(id='hidden-div-y', style={'display': None}),
    html.Div(id='hidden-div-z', style={'display': None}),
    html.Div(id='clustering_settings',
             style=dash_lib.get_default_style(
                width='50%',
                height='10px',
                display=None)),
    html.Div(id='figure_1',
             style=dash_lib.get_default_style(width='50%',
                                              height='800px',
                                              borderStyle=None)),
    html.Div(id='figure_2',
             style=dash_lib.get_default_style(width='50%',
                                              height='800px',
                                              borderStyle=None)),
    html.Div(dash_lib.axis_dropdowns(),
             id='axis_to_show',
             style=dash_lib.get_default_style(width='50%',
                                              height='60px',
                                              borderStyle=None)),
    html.Div(id='hidden-div-2', style={'display': None}),
    html.Div(id='router-process-figure1', style={'display': None}),
    html.Div(id='router-slider-figure1', style={'display': None}),
    html.Div(id='router-preslider-postslider', style={'display': None}),
    html.Div(id='plot_settings',
             style=dash_lib.get_default_style(
                width='50%',
                height='10px',
                display=None)),
])


def parse_contents(contents, filename, date):
    """Given a uploaded file, parse it into a dataframe and save as hdf.

    Parameters
    ----------
    contents : list
        List containing content type and the serialized string
    filename : str
        File name that is uploaded
    date : str
        Modified date

    Returns
    -------
    html.Div
        Returns a status message filled into the div section
    """
    if filename.split('.')[-1] not in ['xls', 'csv']:
        return dash_lib.upload_message('Only supports .xsl or .csv files.')

    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        msg = 'There was an error processing this file.\n\n%s' % e.message
        return dash_lib.upload_message(msg)

    # Save dataframe
    df.to_hdf('./uploads/%s.hdf' % filename.split('.')[-2], 'data', mode='w')

    return dash_lib.upload_message('Upload complete.')


@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents'),
               Input('upload-data', 'filename'),
               Input('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    """Summary.

    Parameters
    ----------
    list_of_contents : TYPE
        Description
    list_of_names : TYPE
        Description
    list_of_dates : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


@app.callback(Output('opt-uploaded-files', 'options'), [Input('upload-data', 'filename')])
def update_date_dropdown(list_of_names):
    """When a file is uploaded, the dropdown list gets updated."""
    return dash_lib.get_uploaded_files()


@app.callback(Output('opt-uploaded-files', 'clearable'), [Input('opt-uploaded-files', 'value')])
def dummy_update(value):
    """Simply a dummy callback, otherwise Dash doesn't render the object."""
    return False


@app.callback(Output('clustering_settings', 'children'), [
    Input('opt-cluster-method', 'options'),
    Input('opt-cluster-method', 'value')])
def update_pre_sliders(options, value):
    """Update the sliders based on what clustering algorithm is active."""
    global cluster_algos
    if value is None:
        return None

    key_sel = options[value]['label']
    return cluster_algos.create_sliders(key_sel, 'pre')


@app.callback(Output('plot_settings', 'children'),
              [Input('button:process-file', 'n_clicks')],
              state=[State('opt-cluster-method', 'options'),
                     State('opt-cluster-method', 'value')])
def update_post_sliders(n_clicks, options, value):
    """Update the sliders available after processing has completed."""
    global cluster_algos
    if value is None or n_clicks is None:
        return None

    key_sel = options[value]['label']
    return cluster_algos.create_sliders(key_sel, 'post')


@app.callback(Output('clustering_settings', 'style'), [
    Input('opt-cluster-method', 'options'),
    Input('opt-cluster-method', 'value')],
    state=[State('clustering_settings', 'style')])
def update_pre_settings_height(options, value, current_style):
    """Update the heights of the DIV containing the pre sliders."""
    global cluster_algos
    if value is None:
        return current_style

    key_sel = options[value]['label']
    options = cluster_algos.get_pre_options(key_sel)
    px_needed = 50 * len(options)
    current_style['height'] = '%dpx' % px_needed

    return current_style


@app.callback(Output('plot_settings', 'style'), [
    Input('opt-cluster-method', 'options'),
    Input('opt-cluster-method', 'value')],
    state=[State('plot_settings', 'style')])
def update_post_settings_height(options, value, current_style):
    """Update the heights of the DIV containing the post sliders."""
    global cluster_algos
    if value is None:
        return current_style

    key_sel = options[value]['label']
    options = cluster_algos.get_post_options(key_sel)
    px_needed = 50 * len(options)
    current_style['height'] = '%dpx' % px_needed
    return current_style


@app.callback(Output('router-process-figure1', 'children'),
              [Input('button:process-file', 'n_clicks')],
              state=[State('opt-cluster-method', 'options'),
                     State('opt-cluster-method', 'value'),
                     State('opt-uploaded-files', 'options'),
                     State('opt-uploaded-files', 'value')])
def callback_process_files(clc, options, value, f_opt, f_val):
    """Process that happens when a file gets processed."""
    global kmean
    global cluster_algos

    if clc is None:
        return None

    if value is None:
        return None

    if f_val is None:
        return None

    key_sel = options[value]['label']
    file_sel = f_opt[f_val]['label']

    dataset_loader.load_dataset(file_sel)
    cluster_algos.get_class(key_sel).input(
        dash_lib.scale_data_array(dataset_loader.get_data()))
    cluster_algos.get_class(key_sel).explore_dash(
        **cluster_algos._sliders_pre.get(key_sel))
    cluster_algos.get_class(key_sel).input(
        dash_lib.scale_data_array(dataset_loader.get_data()))

    return None


@app.callback(Output('router-slider-figure1', 'children'),
              cluster_algos._sliders_post.get_as_inputs())
def trigger_figure_update(*args):
    return None


@app.callback(Output('figure_1', 'children'),
              [Input('router-process-figure1', 'children'),
               Input('router-slider-figure1', 'children')],
              state=[State('opt-cluster-method', 'options'),
                     State('opt-cluster-method', 'value'),
                     State('opt-uploaded-files', 'options'),
                     State('opt-uploaded-files', 'value')])
def callback_show_figure_1(cld1, cld2, options, value, f_opt, f_val):
    global kmean
    global cluster_algos

    if value is None:
        return None

    if f_val is None:
        return None

    key_sel = options[value]['label']
    cluster_algos.get_class(key_sel)._data = dataset_loader.get_active()

    return cluster_algos.get_class(key_sel).get_figure_1(style={'height': '100%'})


@app.callback(Output('figure_2', 'children'),
              [Input('figure_1', 'children'),
               Input('x-axis', 'options'),
               Input('x-axis', 'value'),
               Input('y-axis', 'options'),
               Input('y-axis', 'value'),
               Input('z-axis', 'options'),
               Input('z-axis', 'value')],
              state=[State('opt-cluster-method', 'options'),
                     State('opt-cluster-method', 'value')])
def callback_figure_change(clc, xop, xval, yop, yval, zop, zval, options, value):
    """If figure 1 gets changed, query for figure 2."""
    global cluster_algos
    global dataset_loader
    if value is None:
        return None
    key_sel = options[value]['label']

    cluster_algos.get_class(key_sel)._data = dataset_loader.get_active()
    post_settings = cluster_algos._sliders_post.get(key_sel)

    return cluster_algos.get_class(key_sel).get_figure_2(style={'height': '100%'}, **post_settings)


# CSS pens
app.css.append_css(
    {'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})
app.css.append_css(
    {"external_url": "https://codepen.io/chriddyp/pen/brPBPO.css"})

app.config['suppress_callback_exceptions'] = False
if __name__ == '__main__':
    app.run_server(debug='False', port=8050, host='0.0.0.0')
