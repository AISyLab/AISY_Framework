from aisy_database.db_tables import *
from webapp.mvc.views.plotly import *
from webapp.mvc.views.Utils import *
import numpy as np
import plotly.express as px
import pandas as pd


def metric_early_stopping_plots(all_metric_plots, analysis_id, db_select, metric):
    analysis = db_select.select_analysis(Analysis, analysis_id)
    hyper_parameters = db_select.select_all_from_analysis(HyperParameter, analysis_id)
    rows = db_select.select_all_from_analysis(Metric, analysis_id)

    utils = Utils(analysis.settings)

    elem_values = utils.get_rows(rows, metric)

    min_elem_values = None
    max_elem_values = None
    epochs = analysis.settings["epochs"]

    metric_plots = {metric: []}
    best_epoch_metric = []
    histogram_best_epoch_metric_plots = []

    """ Order elements putting Best Model as the last """
    elem_values_ordererd = []
    for elem in elem_values:
        if "Best Model" not in elem["label"]:
            elem_values_ordererd.append(elem)
    for elem in elem_values:
        if "Best Model" in elem["label"]:
            elem_values_ordererd.append(elem)

    for elem in elem_values_ordererd:

        if "Best Model" in elem["label"]:
            show_legend = True,
            line_color = None
            line_width = 2.5
        elif len(hyper_parameters) == 1:
            show_legend = True
            line_color = "b71c1c" if "val" in elem["label"] else "1565c0"
            line_width = 2.5
        else:
            show_legend = False
            line_color = "rgba(210,210,210,0.8)"
            line_width = 1.0

        if "values" in elem:
            min_elem_values, max_elem_values = utils.get_min_max_ge(min_elem_values, max_elem_values, elem)
            metric_plots[metric].append(create_line_plot(y=elem['values'], line_name=elem['label'], line_color=line_color,
                                                         line_width=line_width, show_legend=show_legend))

            if analysis.settings["early_stopping"]["metrics"][metric]["direction"] == "max":
                best_epoch_metric.append(np.argmax(elem['values']) + 1)
            else:
                best_epoch_metric.append(np.argmin(elem['values']) + 1)

    histogram_best_epoch_metric_plots.append(
        create_hist_plot(x=best_epoch_metric, line_name=f"Best Epochs {metric}")
    )

    x = np.concatenate([range(1, epochs + 1), list(reversed(range(1, epochs + 1)))])
    y = np.concatenate([max_elem_values, list(reversed(min_elem_values))])
    metric_plots[metric].append(create_line_plot_fill(x=x, y=y, line_name=metric, line_color="transparent", line_width=1.0,
                                                      show_legend=False))

    x_range = [1, epochs]

    all_metric_plots.append({
        "title": f"ES {metric} vs Epochs",
        "layout_plotly": get_plotly_layout("Epochs", metric, x_range=x_range),
        "plots": metric_plots[metric]
    })

    if len(hyper_parameters) > 1:
        all_metric_plots.append({
            "title": f"ES Best Epochs {metric}",
            "layout_plotly": get_layout_density("Epochs", "Frequency"),
            "plots": histogram_best_epoch_metric_plots
        })

    return all_metric_plots


def append_to_list(metric, elem, n_profiling_traces, metric_list, profiling_traces_list):
    if metric == "Guessing Entropy" or metric == "Success Rate":
        metric_list.append(elem['values'][len(elem['values']) - 1])
    else:
        metric_list.append(elem['values'])
    profiling_traces_list.append(n_profiling_traces)


def metric_profiling_analyzer_plots(analysis_id, db_select, metric, y_label):
    analysis = db_select.select_analysis(Analysis, analysis_id)
    utils = Utils(analysis.settings)

    if metric == "Guessing Entropy":
        elem_values = db_select.select_all_guessing_entropy_from_analysis(GuessingEntropy, analysis_id)
    elif metric == "Number of Attack Traces":
        elem_values = db_select.select_all_guessing_entropy_from_analysis(GuessingEntropy, analysis_id)
    elif metric == "Success Rate":
        elem_values = db_select.select_all_success_rate_from_analysis(SuccessRate, analysis_id)
    else:
        rows = db_select.select_all_from_analysis(Metric, analysis_id)
        elem_values = utils.get_rows(rows, metric)

    if metric == "Number of Attack Traces":
        for elem in elem_values:
            if elem['values'][len(elem['values']) - 1] < 2:
                elem['values'] = analysis.settings["key_rank_attack_traces"] - np.searchsorted(elem["values"][::-1], 2) * \
                                 analysis.settings["key_rank_report_interval"]
            else:
                elem['values'] = analysis.settings["key_rank_attack_traces"]

    profiling_traces_list = []
    metric_list = []
    type_list = []
    size_list = []

    best_model = {}
    best_model["Validation"] = {}
    best_model["Validation"]["x"] = []
    best_model["Validation"]["y"] = []
    best_model["Attack"] = {}
    best_model["Attack"]["x"] = []
    best_model["Attack"]["y"] = []

    if analysis.settings["use_early_stopping"]:
        for es_metric in analysis.settings["early_stopping"]["metrics"]:
            best_model[f"{es_metric} Validation"] = {}
            best_model[f"{es_metric} Validation"]["x"] = []
            best_model[f"{es_metric} Validation"]["y"] = []
            best_model[f"{es_metric} Attack"] = {}
            best_model[f"{es_metric} Attack"]["x"] = []
            best_model[f"{es_metric} Attack"]["y"] = []

    number_of_searches = None
    if analysis.settings["use_grid_search"]:
        number_of_searches = analysis.settings["grid_search"]["max_trials"]

    if analysis.settings["use_random_search"]:
        number_of_searches = analysis.settings["random_search"]["max_trials"]

    for n_profiling_traces in analysis.settings["profiling_analyzer_steps"]:

        for elem in elem_values:

            if "values" in elem:

                if number_of_searches is not None:

                    for search_index in range(number_of_searches):
                        for data_set in ["Validation", "Attack"]:
                            if f"{data_set} Set {search_index} {n_profiling_traces} traces" in elem["label"]:
                                append_to_list(metric, elem, n_profiling_traces, metric_list, profiling_traces_list)
                                type_list.append(data_set)
                                size_list.append(1)
                        if analysis.settings["use_early_stopping"]:
                            for es_metric in analysis.settings["early_stopping"]["metrics"]:
                                for data_set in ["Validation", "Attack"]:
                                    if f"ES {data_set} Set {es_metric} {search_index} {n_profiling_traces} traces" in elem["label"]:
                                        append_to_list(metric, elem, n_profiling_traces, metric_list, profiling_traces_list)
                                        type_list.append(f"{es_metric} {data_set}")
                                        size_list.append(1)

                for data_set in ["Validation", "Attack"]:
                    if f"{data_set} Set Best Model {n_profiling_traces} traces" in elem["label"]:
                        best_model[data_set]["x"].append(n_profiling_traces)
                        if metric == "Guessing Entropy" or metric == "Success Rate":
                            best_model[data_set]["y"].append(elem['values'][len(elem['values']) - 1])
                        else:
                            best_model[data_set]["y"].append(elem['values'])

                if analysis.settings["use_early_stopping"]:
                    for es_metric in analysis.settings["early_stopping"]["metrics"]:
                        for data_set in ["Validation", "Attack"]:
                            if f"ES {data_set} Set {es_metric} Best Model {n_profiling_traces} traces" in elem["label"]:
                                best_model[f"{es_metric} {data_set}"]["x"].append(n_profiling_traces)
                                if metric == "Guessing Entropy" or metric == "Success Rate":
                                    best_model[f"{es_metric} {data_set}"]["y"].append(elem['values'][len(elem['values']) - 1])
                                else:
                                    best_model[f"{es_metric} {data_set}"]["y"].append(elem['values'])

    data = {
        "Profiling Traces": profiling_traces_list,
        f"{metric}": metric_list,
        "Set": type_list,
        "Size": size_list
    }
    dataframe = pd.DataFrame(data)
    print(dataframe)
    fig = px.scatter(dataframe, x="Profiling Traces", y=f"{metric}", color="Set", size="Size", size_max=10)
    fig.update_layout({
        'plot_bgcolor': '#fafafa',
        'paper_bgcolor': '#fafafa',
        "xaxis": {
            "ticks": '',
            "side": 'bottom',
            "tickcolor": '#fff',
            "gridcolor": '#d0d0d0',
            "color": '#263238'
        },
        "yaxis": {
            "ticks": '',
            "ticksuffix": ' ',
            "tickcolor": '#fff',
            "gridcolor": '#d0d0d0',
            "color": '#263238'
        }
    })

    fig_to_add = go.Line(x=best_model["Validation"]["x"], y=best_model["Validation"]["y"], name="Best Model Validation")
    fig.add_trace(fig_to_add)
    fig_to_add = go.Line(x=best_model["Attack"]["x"], y=best_model["Attack"]["y"], name="Best Model Attack")
    fig.add_trace(fig_to_add)

    if analysis.settings["use_early_stopping"]:
        for metric in analysis.settings["early_stopping"]["metrics"]:
            fig_to_add = go.Line(x=best_model[f"{metric} Validation"]["x"], y=best_model[f"{metric} Validation"]["y"],
                                 name=f"Best Model {metric} Validation")
            fig.add_trace(fig_to_add)
            fig_to_add = go.Line(x=best_model[f"{metric} Attack"]["x"], y=best_model[f"{metric} Attack"]["y"],
                                 name=f"Best Model {metric} Attack")
            fig.add_trace(fig_to_add)

    if not analysis.settings["use_grid_search"] and not analysis.settings["use_random_search"]:
        hyper_parameters = db_select.select_all_from_analysis(HyperParameter, analysis_id)

        for hp in hyper_parameters:

            y_list = []
            elem_values = []

            if metric == "Guessing Entropy":
                elem_ids = db_select.select_guessing_entropy_from_hyperparameter(HyperParameterGuessingEntropy, hp.id, analysis_id)
                for elem in elem_ids:
                    elem_values.append(db_select.select_guessing_entropy_from_id(GuessingEntropy, elem.id))
            elif metric == "Success Rate":
                elem_ids = db_select.select_guessing_entropy_from_hyperparameter(HyperParameterSuccessRate, hp.id, analysis_id)
                for elem in elem_ids:
                    elem_values.append(db_select.select_success_rate_from_id(SuccessRate, elem.id))
            if metric == "Number of Attack Traces":
                elem_ids = db_select.select_guessing_entropy_from_hyperparameter(HyperParameterGuessingEntropy, hp.id, analysis_id)
                for elem in elem_ids:
                    elem_values.append(db_select.select_guessing_entropy_from_id(GuessingEntropy, elem.id))
                for elem in elem_values:
                    if elem['values'][len(elem['values']) - 1] < 2:
                        y_list.append(analysis.settings["key_rank_attack_traces"] - np.searchsorted(elem["values"][::-1], 2) * \
                                      analysis.settings["key_rank_report_interval"])
                    else:
                        y_list.append(analysis.settings["key_rank_attack_traces"])

            if metric == "Guessing Entropy" or metric == "Success Rate":
                for elem in elem_values:
                    y_list.append(elem['values'][len(elem['values']) - 1])

            fig_to_add = go.Line(x=analysis.settings["profiling_analyzer_steps"], y=y_list, name=f"Attack Set hp_id:{hp.id}")
            fig.add_trace(fig_to_add)

    return create_scatter_and_line_plot_dash(fig)


def metric_plots(analysis_id, db_select, metric, y_label):
    data = []

    analysis = db_select.select_analysis(Analysis, analysis_id)
    hyper_parameters = db_select.select_all_from_analysis(HyperParameter, analysis_id)
    utils = Utils(analysis.settings)

    x_range = None
    epochs = None
    report_interval = None
    min_elem_values = None
    max_elem_values = None

    if metric == "Guessing Entropy":
        elem_values = db_select.select_all_guessing_entropy_from_analysis(GuessingEntropy, analysis_id)
        x_label = "Traces"
    elif metric == "Success Rate":
        elem_values = db_select.select_all_success_rate_from_analysis(SuccessRate, analysis_id)
        x_label = "Traces"
    else:
        rows = db_select.select_all_from_analysis(Metric, analysis_id)
        elem_values = utils.get_rows(rows, metric)

        epochs = analysis.settings["epochs"]
        x_range = [1, epochs]
        x_label = "Epochs"

    """ Order elements putting Best Model as the last """
    elem_values_ordererd = []
    for elem in elem_values:
        if "Best Model" not in elem["label"] and "Ensemble" not in elem["label"]:
            elem_values_ordererd.append(elem)
    for elem in elem_values:
        if "Best Model" in elem["label"] or "Ensemble" in elem["label"]:
            elem_values_ordererd.append(elem)

    for elem_index, elem in enumerate(elem_values_ordererd):

        if "values" in elem:

            if "report_interval" in elem:
                report_interval = elem['report_interval']
                x_range = [report_interval, len(elem['values']) * report_interval]

            min_elem_values, max_elem_values = utils.get_min_max_ge(min_elem_values, max_elem_values, elem)

            x = utils.get_x(elem, metric)
            show_legend = utils.get_legend(elem)
            line_color = utils.get_color(elem)
            line_width = utils.get_line_width(elem)
            line_label = utils.get_label(elem, db_select, metric)

            data.append({'x': x, 'y': elem['values'], 'type': 'line', 'name': line_label, 'showlegend': show_legend, 'line': {}})
            if line_color is not None:
                data[len(data) - 1]['line']['color'] = line_color
            data[len(data) - 1]['line']['width'] = line_width

    if len(hyper_parameters) > 1:

        if metric == "Guessing Entropy" or metric == "Success Rate":
            x = np.concatenate(
                [range(report_interval, len(min_elem_values) * report_interval + report_interval, report_interval),
                 list(reversed(range(report_interval, len(min_elem_values) * report_interval + report_interval, report_interval)))])
            y = np.concatenate([min_elem_values, list(reversed(max_elem_values))])
        else:
            x = np.concatenate([range(1, epochs + 1), list(reversed(range(1, epochs + 1)))])
            y = np.concatenate([min_elem_values, list(reversed(max_elem_values))])

        data.append(
            {
                'x': x,
                'y': y,
                'fill': "tozerox",
                'fillcolor': "rgba(230,230,230,0.3)",
                'type': "scatter",
                'name': 'min_max__key_rank',
                'showlegend': False,
                'line': {},
            }
        )
        data[len(data) - 1]['line']['color'] = "transparent"
        data[len(data) - 1]['line']['width'] = 1.0

    return create_line_plot_dash(data, x_label, y_label, x_range=x_range)


def visualization_plots(analysis_id, db_select, hp):
    data = []

    analysis = db_select.select_analysis(Analysis, analysis_id)
    utils = Utils(analysis.settings)

    x_range = None
    min_elem_values = None
    max_elem_values = None

    visualization_rows = db_select.select_visualization_from_analysis(Visualization, hp.id, analysis_id)
    x = np.arange(1, analysis.settings["number_of_samples"] + 1)

    for idx in range(analysis.settings['epochs']):
        min_elem_values, max_elem_values = utils.get_min_max_ge(min_elem_values, max_elem_values, visualization_rows[idx])

        show_legend = False
        line_color = "rgba(210,210,210,0.8)"
        line_width = 1.0
        line_label = visualization_rows[idx]["label"]

        data.append(
            {'x': x, 'y': visualization_rows[idx]["values"], 'type': 'line', 'name': line_label,
             'showlegend': show_legend,
             'line': {}})
        if line_color is not None:
            data[len(data) - 1]['line']['color'] = line_color
        data[len(data) - 1]['line']['width'] = line_width

    show_legend = True
    line_color = None
    line_width = 1.5
    line_label = f"IG epoch {analysis.settings['epochs']} hp_id: {hp.id}"

    data.append(
        {'x': x, 'y': visualization_rows[analysis.settings['epochs'] - 1]["values"], 'type': 'line', 'name': line_label,
         'showlegend': show_legend,
         'line': {}})
    if line_color is not None:
        data[len(data) - 1]['line']['color'] = line_color
    data[len(data) - 1]['line']['width'] = line_width

    x = np.concatenate(
        [range(1, analysis.settings["number_of_samples"] + 1), list(reversed(range(1, analysis.settings["number_of_samples"] + 1)))])
    y = np.concatenate([min_elem_values, list(reversed(max_elem_values))])

    data.append(
        {
            'x': x,
            'y': y,
            'fill': "tozerox",
            'fillcolor': "rgba(230,230,230,0.3)",
            'type': "scatter",
            'name': 'min_max__key_rank',
            'showlegend': False,
            'line': {},
        }
    )
    data[len(data) - 1]['line']['color'] = "transparent"
    data[len(data) - 1]['line']['width'] = 1.0

    return create_line_plot_dash(data, "Samples", "Input Gradient", x_range=x_range)


def visualization_plots_heatmap(analysis_id, db_select):
    all_visualization_plots = []
    analysis = db_select.select_analysis(Analysis, analysis_id)
    hyper_parameters = db_select.select_all_from_analysis(HyperParameter, analysis_id)

    for hp in hyper_parameters:

        visualization_plots = []
        visualization_rows = db_select.select_visualization_from_analysis(Visualization, hp.id, analysis_id)

        if len(visualization_rows) > 0:
            visualization_plots_metrics = []
            z = []
            x = list(range(len(visualization_rows[0]['values'])))
            y = []
            epoch = 0
            for idx in range(analysis.settings['epochs']):
                z.append(visualization_rows[idx]['values'])
                y.append(epoch)
                epoch += 1
            visualization_plots_metrics.append(create_heatmap(x=x, y=y, z=z))
            visualization_plots.append(visualization_plots_metrics)

            all_visualization_plots.append({
                "title": f"Visualization hp: {hp.id}",
                "layout_plotly": get_plotly_layout("Samples", "Epoch"),
                "plots": visualization_plots,
                "hp_id": hp.id
            })

    return all_visualization_plots


def confusion_matrix_plots(analysis_id, db_select):
    all_confusion_matrix_plots = []
    hyper_parameters = db_select.select_all_from_analysis(HyperParameter, analysis_id)

    for hp in hyper_parameters:
        confusion_matrix_plots = []

        confusion_matrix_all_key_bytes = db_select.select_confusion_matrix_from_analysis(ConfusionMatrix, hp.id, analysis_id)
        if len(confusion_matrix_all_key_bytes) > 0:
            z = []
            y = []
            x = list(range(len(confusion_matrix_all_key_bytes[0]['values'])))
            confusion_matrix_plots_metrics = []

            for y_true, confusion_matrix_key_byte in enumerate(confusion_matrix_all_key_bytes):
                z.append(confusion_matrix_key_byte['values'])
                y.append(y_true)
            if len(x) > 9:
                confusion_matrix_plots_metrics.append(create_heatmap(x=x, y=y, z=z))
            else:
                confusion_matrix_plots_metrics.append(create_annotated_heatmap(x=x, y=y, z=z))
            confusion_matrix_plots.append(confusion_matrix_plots_metrics)

            all_confusion_matrix_plots.append({
                "title": f"Confusion Matrix hp_id:{hp.id}",
                "layout_plotly": get_plotly_layout("Y_true", "Y_pred"),
                "plots": confusion_matrix_plots,
                "hp_id": hp.id
            })

    return all_confusion_matrix_plots
