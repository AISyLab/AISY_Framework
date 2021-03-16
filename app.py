from flask import Flask, render_template
from aisy_database.db_select import *
from aisy_database.db_delete import *
from custom.custom_datasets.datasets import *
from webapp.mvc.controllers.generate_script import generate_script
from webapp.mvc.views import views
from webapp.mvc.views import plotly
import os
import time
import pytz
import h5py
import numpy as np
from datetime import datetime
import flaskcode
import hiplot as hip
import matplotlib.pyplot as plt
import dash
import dash_html_components as html
import dash_bootstrap_components as dbc

app = Flask(__name__,
            static_url_path='',
            static_folder='webapp/static',
            template_folder='webapp/templates')

app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config.from_object(flaskcode.default_config)
app.config['FLASKCODE_RESOURCE_BASEPATH'] = 'scripts'
app.register_blueprint(flaskcode.blueprint, url_prefix='/scripts')

dash_app_key_ranks = dash.Dash(
    __name__,
    server=app,
    routes_pathname_prefix='/dash/key_ranks/',
    external_stylesheets=[dbc.themes.CERULEAN]
)
dash_app_key_ranks.layout = html.Div(children=[])

dash_app_success_rates = dash.Dash(
    __name__,
    server=app,
    routes_pathname_prefix='/dash/success_rates/',
    external_stylesheets=[dbc.themes.CERULEAN]
)
dash_app_success_rates.layout = html.Div(children=[])

dash_app_loss = dash.Dash(
    __name__,
    server=app,
    routes_pathname_prefix='/dash/loss/',
    external_stylesheets=[dbc.themes.CERULEAN]
)
dash_app_loss.layout = html.Div(children=[])

dash_app_accuracy = dash.Dash(
    __name__,
    server=app,
    routes_pathname_prefix='/dash/accuracy/',
    external_stylesheets=[dbc.themes.CERULEAN]
)
dash_app_accuracy.layout = html.Div(children=[])


@app.before_request
def before_request():
    app.jinja_env.cache = {}


app.before_request(before_request)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/dashboard')
def dashboard():
    return render_template("dashboard/index.html")


@app.route('/scripts')
def scripts():
    return "ok"


databases_root_folder = "my_path/AISY_framework/resources/databases/"
datasets_root_folder = "my_dataset_folder/"
resources_root_folder = "my_path/AISY_framework/resources/"


@app.route('/tables')
def table():
    db_files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(databases_root_folder):
        for file in f:
            if file.endswith(".sqlite"):
                db_files.append(file)

    all_tables = []
    all_tables_names = []

    for db_file in db_files:

        if os.path.exists(databases_root_folder + db_file):

            db_select = DBSelect(databases_root_folder + db_file)
            analysis_all = db_select.select_all(Analysis)
            analyses = []

            for analysis in analysis_all:
                if not analysis.deleted:
                    localtimezone = pytz.timezone(os.getenv("TIME_ZONE"))
                    analysis_datetime = datetime.strptime(str(analysis.datetime), "%Y-%m-%d %H:%M:%S.%f").astimezone(
                        localtimezone).__format__(
                        "%b %d, %Y %H:%M:%S")

                    key_ranks = db_select.select_key_ranks_from_analysis(KeyRank, analysis.id)
                    success_rates = db_select.select_success_rates_from_analysis(SuccessRate, analysis.id)
                    neural_network = db_select.select_from_analysis(NeuralNetwork, analysis.id)
                    leakage_model = db_select.select_from_analysis(LeakageModel, analysis.id)

                    analyses.append({
                        "id": analysis.id,
                        "datetime": analysis_datetime,
                        "dataset": analysis.dataset,
                        "settings": analysis.settings,
                        "elapsed_time": time.strftime('%H:%M:%S', time.gmtime(analysis.elapsed_time)),
                        "key_ranks": key_ranks,
                        "success_rates": success_rates,
                        "leakage_model": leakage_model,
                        "neural_network_name": "not ready" if neural_network is None else neural_network.model_name
                    })

            all_tables.append(analyses)
            all_tables_names.append(db_file)

    return render_template("tables.html", all_tables=all_tables, all_tables_names=all_tables_names)


@app.route('/search/<string:table_name>')
def search(table_name):
    if os.path.exists(databases_root_folder + table_name):

        db_select = DBSelect(databases_root_folder + table_name)
        analysis_all = db_select.select_all(Analysis)
        analyses = []

        hp = []

        for analysis in analysis_all:

            final_key_ranks = db_select.select_key_ranks_from_analysis(KeyRank, analysis.id)

            if len(final_key_ranks) > 0:
                hyper_parameters = db_select.select_from_analysis(HyperParameter, analysis.id)
                training_hyper_parameters = hyper_parameters.hyper_parameters
                training_hyper_parameters[0]['guessing_entropy'] = final_key_ranks[0][0]['key_rank']
                hp.append(training_hyper_parameters[0])

        exp = hip.Experiment().from_iterable(hp)
        exp.display_data(hip.Displays.PARALLEL_PLOT).update({
            'hide': ['uid', 'key_rank', 'key'],  # Hide some columns
            'order': ['guessing_entropy'],  # Put column time first on the left
        })
        exp.validate()
        exp.to_html("webapp/templates/hiplot.html")

        return render_template("dashboard/search.html", analyses=analyses)
    return render_template("dashboard/search.html", analyses=[])


@app.route('/result/<int:analysis_id>/<string:table_name>')
def result(analysis_id, table_name):
    db_select = DBSelect(databases_root_folder + table_name)

    # get neural network information from database
    analysis = db_select.select_analysis(Analysis, analysis_id)

    all_metric_plots = views.metric_plots(analysis.id, db_select)

    all_accuracy_plots = views.accuracy_plots(analysis.id, db_select)
    dash_app_accuracy.layout = html.Div(children=[all_accuracy_plots])
    all_loss_plots = views.loss_plots(analysis.id, db_select)
    dash_app_loss.layout = html.Div(children=[all_loss_plots])
    if "ensemble" in analysis.settings:
        all_key_rank_plots = views.ensemble_plots_key_rank(analysis.id, db_select)
    else:
        all_key_rank_plots = views.key_rank_plots(analysis.id, db_select)
    dash_app_key_ranks.layout = html.Div(children=[all_key_rank_plots])
    if "ensemble" in analysis.settings:
        all_success_rate_plots = views.ensemble_plots_success_rate(analysis.id, db_select)
    else:
        all_success_rate_plots = views.success_rate_plots(analysis.id, db_select)
    dash_app_success_rates.layout = html.Div(children=[all_success_rate_plots])

    # get neural network information from database
    neural_network_model = db_select.select_from_analysis(NeuralNetwork, analysis_id)

    # get training hyper-parameters information from database
    hyper_parameters = db_select.select_all_from_analysis(HyperParameter, analysis_id)
    training_hyper_parameters = []
    hyper_parameters_table = []
    for i, hp in enumerate(hyper_parameters):
        training_hyper_parameters.append(hp.hyper_parameters)

        hp_struct = {}

        filter_list = []
        stride_list = []
        kernel_list = []
        pooling_type_list = []
        pooling_size_list = []
        pooling_stride_list = []

        if "conv_layers" in hp.hyper_parameters[0]:
            conv_layers = hp.hyper_parameters[0]["conv_layers"]
        else:
            conv_layers = 0

        for hp_key in hp.hyper_parameters[0]:
            if "filter" in hp_key and len(filter_list) < conv_layers:
                filter_list.append(hp.hyper_parameters[0][hp_key])
            elif "pooling_type" in hp_key and len(pooling_type_list) < conv_layers:
                pooling_type_list.append(hp.hyper_parameters[0][hp_key])
            elif "pooling_size" in hp_key and len(pooling_size_list) < conv_layers:
                pooling_size_list.append(hp.hyper_parameters[0][hp_key])
            elif "pooling_stride" in hp_key and len(pooling_stride_list) < conv_layers:
                pooling_stride_list.append(hp.hyper_parameters[0][hp_key])
            elif "stride" in hp_key and len(stride_list) < conv_layers:
                stride_list.append(hp.hyper_parameters[0][hp_key])
            elif "kernel" in hp_key and len(kernel_list) < conv_layers:
                kernel_list.append(hp.hyper_parameters[0][hp_key])
            else:
                hp_struct[str(hp_key)] = hp.hyper_parameters[0][hp_key]

        if len(filter_list) > 0:
            hp_struct["filters"] = filter_list
        if len(stride_list) > 0:
            hp_struct["strides"] = stride_list
        if len(kernel_list) > 0:
            hp_struct["kernels"] = kernel_list
        if len(pooling_type_list):
            hp_struct["pooling_types"] = pooling_type_list
        if len(pooling_size_list) > 0:
            hp_struct["pooling_sizes"] = pooling_size_list
        if len(pooling_stride_list) > 0:
            hp_struct["pooling_strides"] = pooling_stride_list
        hp_struct["id"] = hp.id
        hyper_parameters_table.append(hp_struct)

    hyper_parameter_search = []
    if len(hyper_parameters) > 1:
        hyper_parameter_search = db_select.select_from_analysis(HyperParameterSearch, analysis.id)

    # get leakage model information from database
    leakage_model = db_select.select_from_analysis(LeakageModel, analysis_id)
    leakage_model_parameters = leakage_model.leakage_model

    # get visualization plots
    all_visualization_plots = views.visualization_plots(analysis.id, db_select)
    all_visualization_heatmap_plots = views.visualization_plots_heatmap(analysis.id, db_select)

    # confusion matrix plot
    all_confusion_matrix_plots = views.confusion_matrix_plots(analysis.id, db_select)

    return render_template("dashboard/result.html",
                           all_plots=all_metric_plots,
                           all_key_rank_plots=all_key_rank_plots,
                           all_success_rate_plots=all_success_rate_plots,
                           neural_network_description=neural_network_model.description,
                           training_hyper_parameters=training_hyper_parameters,
                           hyper_parameters=hyper_parameters,
                           hyper_parameters_table=hyper_parameters_table,
                           hyper_parameter_search=hyper_parameter_search,
                           leakage_model_parameters=leakage_model_parameters,
                           all_visualization_plots=all_visualization_plots,
                           all_visualization_heatmap_plots=all_visualization_heatmap_plots,
                           all_confusion_matrix_plots=all_confusion_matrix_plots,
                           analysis=analysis)


@app.route('/datasets')
def datasets():
    all_plots = []

    for ds in datasets_dict:
        in_file = h5py.File(datasets_root_folder + ds, "r")
        profiling_samples = np.array(in_file['Profiling_traces/traces'], dtype=np.float64)

        all_plots.append({
            "title": ds,
            "layout_plotly": plotly.get_plotly_layout("Samples", "Amplitude (mV)"),
            "plots": plotly.create_line_plot(y=profiling_samples[0], line_name=ds)
        })

    return render_template("dashboard/datasets.html", all_plots=all_plots)


@app.route("/generate_script/<int:analysis_id>/<string:table_name>/<reproducible>/<from_db>")
def gen_script(analysis_id, table_name, reproducible, from_db):
    if reproducible == "true":
        if from_db == "true":
            generate_script("script_aes_{}_db_based_reproducible".format(analysis_id), databases_root_folder, table_name, analysis_id, True,
                            True)
        else:
            generate_script("script_aes_{}_reproducible".format(analysis_id), databases_root_folder, table_name, analysis_id, True, False)
    else:
        generate_script("script_aes_{}".format(analysis_id), databases_root_folder, table_name, analysis_id, False, False)
    return "ok"


@app.route("/generate_plot/<int:analysis_id>/<string:table_name>/<metric>")
def gen_plot(analysis_id, table_name, metric):
    db_select = DBSelect(databases_root_folder + table_name)

    analysis = db_select.select_analysis(Analysis, analysis_id)
    hyper_parameters = db_select.select_all_from_analysis(HyperParameter, analysis_id)
    best_epoch_metric = []

    if metric == "Guessing_Entropy":
        result_key_byte = db_select.select_key_ranks_from_analysis(KeyRank, analysis_id)
    elif metric == "Success_Rate":
        result_key_byte = db_select.select_success_rates_from_analysis(SuccessRate, analysis_id)
    else:
        result_key_byte = []
        all_metrics_names = db_select.select_metric_names_from_analysis(Metric, analysis_id)

        if metric == "accuracy":
            for metric_name in all_metrics_names:
                if metric in metric_name:
                    if "grid_search" in analysis.settings or "random_search" in analysis.settings:
                        if "best" in metric_name:
                            result_key_byte.append(db_select.select_metric_from_analysis(Metric, metric_name, analysis_id)[0])
                    else:
                        result_key_byte.append(db_select.select_metric_from_analysis(Metric, metric_name, analysis_id)[0])
        elif metric == "loss":
            for metric_name in all_metrics_names:
                if metric in metric_name:
                    if "grid_search" in analysis.settings or "random_search" in analysis.settings:
                        if "best" in metric_name:
                            result_key_byte.append(db_select.select_metric_from_analysis(Metric, metric_name, analysis_id)[0])
                    else:
                        result_key_byte.append(db_select.select_metric_from_analysis(Metric, metric_name, analysis_id)[0])
        else:
            if "Best Epochs" in metric:
                metric = metric.replace("ES ", "")
                metric = metric.replace("Best Epochs ", "")
                for m in range(len(hyper_parameters) - 1):
                    if "early_stopping" in analysis.settings:
                        metric_value = db_select.select_metric_from_analysis(Metric, "{} {}".format(metric, m), analysis_id)[0]
                        if analysis.settings["early_stopping"]["metrics"][metric.replace("val_", "")]["direction"] == "max":
                            best_epoch_metric.append(np.argmax(metric_value['values']) + 1)
                        else:
                            best_epoch_metric.append(np.argmin(metric_value['values']) + 1)
            else:
                if len(hyper_parameters) > 1:
                    metric = metric.replace("ES ", "")
                    for m in range(len(hyper_parameters)):
                        if m == len(hyper_parameters) - 1:
                            result_key_byte.append(db_select.select_metric_from_analysis(Metric, "{} best".format(metric), analysis_id)[0])
                        else:
                            result_key_byte.append(db_select.select_metric_from_analysis(Metric, "{} {}".format(metric, m), analysis_id)[0])
                else:
                    metric = metric.replace("ES ", "")
                    result_key_byte.append(db_select.select_metric_from_analysis(Metric, metric, analysis_id)[0])

    my_dpi = 100
    plt.figure(figsize=(800 / my_dpi, 600 / my_dpi), dpi=my_dpi)
    dir_analysis_id = "resources/figures/{}".format(analysis_id)
    if not os.path.exists(dir_analysis_id):
        os.makedirs(dir_analysis_id)
    if metric == "Guessing_Entropy" or metric == "Success_Rate":
        for res in result_key_byte:
            if "Best" in res['label'] or "Ensemble" in res['label'] or "Attack" in res['label']:
                plt.grid(ls='--')
                plt.plot(np.arange(res['report_interval'], (len(res['values']) + 1) * res['report_interval'], res['report_interval']),
                         res['values'],
                         label=res['label'])
                plt.legend(loc='best', fontsize=13)
                plt.xlim([1, len(res['values']) * res['report_interval']])
                plt.ylabel(metric.replace("_", " "), fontsize=13)
                if metric == "Guessing_Entropy" or metric == "Success_Rate":
                    plt.xlabel("Attack Traces", fontsize=13)
                else:
                    plt.xlabel("Epochs", fontsize=13)
        plt.savefig("{}/{}_{}_{}.png".format(dir_analysis_id, metric, analysis_id, table_name.replace(".sqlite", "")), format="png")
    else:
        if len(best_epoch_metric) > 0:
            max_number_of_epochs = hyper_parameters[0].hyper_parameters[0]['epochs']
            for hp in hyper_parameters:
                if hp.hyper_parameters[0]['epochs'] > max_number_of_epochs:
                    max_number_of_epochs = hp.hyper_parameters[0]['epochs']
            metric = "Best epochs {}".format(metric.replace("ES ", ""))
            plt.hist(best_epoch_metric, label=metric, bins=hyper_parameters[0].hyper_parameters[0]['epochs'])
            plt.legend(loc='best', fontsize=13)
            plt.ylabel("Frequency", fontsize=13)
            plt.xlabel("Epochs", fontsize=13)
            plt.grid(ls='--')
            plt.savefig("{}/best_epochs_{}_{}_{}.png".format(dir_analysis_id, metric, analysis_id, table_name.replace(".sqlite", "")),
                        format="png")
        else:
            for res in result_key_byte:
                if len(hyper_parameters) == 1 or "best" in res['label']:
                    plt.plot(np.arange(1, len(res['values']) + 1, 1), res['values'], label=res['label'])
                    plt.legend(loc='best', fontsize=13)
                else:
                    plt.plot(np.arange(1, len(res['values']) + 1, 1), res['values'], color="grey", linestyle="--", linewidth=1.0)
                plt.xlim([1, len(res['values'])])
                plt.ylabel(metric.replace("_", " "), fontsize=13)
                plt.xlabel("Epochs", fontsize=13)
                plt.grid(ls='--')
            plt.savefig("{}/{}_{}_{}.png".format(dir_analysis_id, metric, analysis_id, table_name.replace(".sqlite", "")), format="png")

    return "ok"


@app.route("/delete_analysis/<int:analysis_id>/<string:table_name>")
def delete_analysis(analysis_id, table_name):
    db_delete = DBDelete(databases_root_folder + table_name)
    db_delete.soft_delete_analysis_from_table(Analysis, analysis_id)

    return "ok"


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port='5001')
    dash_app_accuracy.run_server(debug=True)
    dash_app_loss.run_server(debug=True)
    dash_app_key_ranks.run_server(debug=True)
    dash_app_success_rates.run_server(debug=True)
