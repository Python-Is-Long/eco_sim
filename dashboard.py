import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pickle
import os
from datetime import datetime
from sim import EconomyStats

# Path to the pickle file
PICKLE_FILE_PATH = 'simulation_stats.pkl'  # Update with your actual path

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = 'Live Data Dashboard'

# Define the layout
app.layout = html.Div([
    html.H1('Live Data Dashboard'),

    # Div for last update time
    html.Div(id='last-update', style={'marginBottom': 20}),

    # Div for scalar values
    html.Div(id='scalar-values-div', style={'marginBottom': 40}),

    # Div for time-series plots
    html.Div(id='plots-div'),

    # Interval component for updates
    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # Update every 5 seconds
        n_intervals=0
    )
])

# Function to load data from the pickle file
def load_data():
    with open(PICKLE_FILE_PATH, 'rb') as f:
        data_class = pickle.load(f)
    return data_class

# Functions to get scalar and time-series attributes
def get_time_series_attributes(data_class):
    return {attr: value for attr, value in data_class.__dict__.items() if isinstance(value, list)}

def get_scalar_attributes(data_class):
    return {attr: value for attr, value in data_class.__dict__.items() if isinstance(value, int)}

# Callback to update scalar values and plots
@app.callback(
    [
        Output('scalar-values-div', 'children'),
        Output('plots-div', 'children'),
        Output('last-update', 'children'),
    ],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    data_class = load_data()
    last_update = 'Last Update: {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    # Get attributes
    scalar_attrs = get_scalar_attributes(data_class)
    time_series_attrs = get_time_series_attributes(data_class)

    # Create scalar value displays
    scalar_components = []
    for attr_name, value in scalar_attrs.items():
        component = html.Div([
            html.H4(attr_name.replace('_', ' ').title()),
            html.Div(str(value), style={'fontSize': 24})
        ], style={'width': '20%', 'display': 'inline-block', 'marginBottom': '20px'})
        scalar_components.append(component)

    # Time steps (assuming steps correspond to list indices)
    max_length = max([len(v) for v in time_series_attrs.values()] + [0])
    steps = list(range(max_length))

    # Create time-series plots
    plot_components = []
    plots_per_row = 2  # Adjust this value to change the number of plots per row
    row_children = []
    count = 0

    for attr_name, values in time_series_attrs.items():
        # Extend values to match max length if necessary
        if len(values) < max_length:
            values = values + [None] * (max_length - len(values))

        trace = go.Scatter(
            x=steps,
            y=values,
            mode='lines',
            name=attr_name.replace('_', ' ').title()
        )
        layout = go.Layout(
            title=attr_name.replace('_', ' ').title(),
            xaxis={'title': 'Step'},
            yaxis={'title': attr_name.replace('_', ' ').title()}
        )
        fig = go.Figure(data=[trace], layout=layout)

        graph = dcc.Graph(
            id=f'{attr_name}-graph',
            figure=fig,
            style={'width': f'{100 / plots_per_row}%', 'display': 'inline-block'}
        )

        row_children.append(graph)
        count += 1

        # Add a row when we reach the plots per row or at the end
        if count % plots_per_row == 0 or count == len(time_series_attrs):
            plot_row = html.Div(row_children, style={'display': 'flex', 'marginBottom': '20px'})
            plot_components.append(plot_row)
            row_children = []

    return scalar_components, plot_components, last_update

if __name__ == '__main__':
    app.run_server(debug=True)