import plotly
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import json
import dash_core_components as dcc


def create_line_plot_dash(data, x_axis_title, y_axis_title, x_range=None):
    return dcc.Graph(id='timeseries',
                     config={'displayModeBar': False},
                     animate=True,
                     figure={
                         'data': data,
                         'layout': get_plotly_layout(x_axis_title, y_axis_title, x_range=x_range)
                     })


def create_scatter_and_line_plot_dash(fig):
    return dcc.Graph(figure=fig)


def create_line_plot(x=None, y=None, line_name=None, line_color=None, line_width=1.5, show_legend=True):
    if x is None:
        x = np.arange(1, len(y) + 1, 1)

    df = pd.DataFrame({'x': x, 'y': y})  # creating a sample dataframe
    data_line = go.Line(
        x=df['x'],  # assign x as the dataframe column 'x'
        y=df['y'],
        line={
            'width': line_width
        },
        showlegend=show_legend
    )

    if line_name is not None:
        data_line['name'] = line_name
    if line_color is not None:
        data_line['line']['color'] = line_color

    return json.dumps([data_line], cls=plotly.utils.PlotlyJSONEncoder)


def create_line_plot_fill(x=None, y=None, line_name=None, line_color=None, line_width=1.5, show_legend=True):
    if x is None:
        x = np.arange(1, len(y) + 1, 1)

    df = pd.DataFrame({'x': x, 'y': y})  # creating a sample dataframe
    data_line = go.Line(
        x=df['x'],  # assign x as the dataframe column 'x'
        y=df['y'],
        line={
            'width': line_width
        },
        showlegend=show_legend,
        fill="tozerox",
        fillcolor="rgba(230,230,230,0.3)",
    )

    if line_name is not None:
        data_line['name'] = line_name
    if line_color is not None:
        data_line['line']['color'] = line_color

    return json.dumps([data_line], cls=plotly.utils.PlotlyJSONEncoder)


def create_hist_plot(x=None, line_name="line_plot"):
    data = [go.Histogram(
        x=x,
        opacity=0.75,
        name=line_name,
        nbinsx=50,
    )]

    return json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)


def get_plotly_layout(x_axis_title, y_axis_title, x_range=None):
    layout = {
        "annotations": [],
        "font": {
            "size": 14,
            "family": 'Calibri',
            "color": '#263238'
        },
        "xaxis": {
            "ticks": '',
            "side": 'bottom',
            "title": x_axis_title,
            "tickcolor": '#fff',
            "gridcolor": '#d0d0d0',
            "color": '#263238',
            "range": [x_range[0], x_range[1]] if x_range is not None else '',
        },
        "yaxis": {
            "ticks": '',
            "ticksuffix": ' ',
            "autosize": False,
            "title": y_axis_title,
            "tickcolor": '#fff',
            "gridcolor": '#d0d0d0',
            "color": '#263238'
        },
        'paper_bgcolor': '#fafafa',
        'plot_bgcolor': '#fafafa'
    }

    return layout


def get_layout_density(x_axis_title, y_axis_title):
    layout = {
        "font": {
            "size": 12,
            "family": 'Calibri',
            "color": '#263238'
        },
        "showlegend": True,
        "autosize": True,
        "paper_bgcolor": '#fafafa',
        "plot_bgcolor": '#fafafa',
        'shapes': [],
        "xaxis": {
            "domain": [0, 50],
            "showgrid": True,
            "zeroline": False,
            "title": x_axis_title,
            "ticks": '',
            "side": 'bottom',
            "tickcolor": '#fff',
            "gridcolor": '#d0d0d0',
            "color": '#263238'
        },
        "yaxis": {
            # "domain": [0, 1],
            "showgrid": True,
            "zeroline": False,
            "title": y_axis_title,
            "ticks": '',
            "side": 'bottom',
            "tickcolor": '#fff',
            "gridcolor": '#d0d0d0',
            "color": '#263238'
        },
        "bargap": 0.0,
        "bargroupgap": 0.1,
    }

    return layout


def create_heatmap(x=None, y=None, z=None):
    data = [
        go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale='Viridis'
        )
    ]

    return json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)


def create_annotated_heatmap(x=None, y=None, z=None):
    data = go.Heatmap(z=z, x=x, y=y, colorscale='Viridis')

    annotations = go.Annotations()
    for n, row in enumerate(z):
        for m, val in enumerate(row):
            annotations.append(go.Annotation(text=str(z[n][m]), x=x[m], y=y[n], xref='x1', yref='y1', showarrow=False))

    fig = go.Figure(data=go.Data([data]))
    fig['layout'].update(
        annotations=annotations,
        xaxis=go.XAxis(ticks='', title='y_pred', side='bottom'),
        yaxis=go.YAxis(ticks='', title='y_true', ticksuffix='  '),  # ticksuffix is a workaround to add a bit of padding
        autosize=False
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
