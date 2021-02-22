from flask import Flask, render_template
from commons.sca_database import ScaDatabase
from commons.sca_generate_script import generate_script
from commons.sca_views import ScaViews
from commons.sca_tables import *
from commons.plotly.PlotlyPlots import PlotlyPlots
from custom.custom_datasets.datasets import *
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


databases_root_folder = "C:/Users/guilh/PycharmProjects/aisy_framework/resources/databases/"
datasets_root_folder = "D:/traces/"


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

            db = ScaDatabase(databases_root_folder + db_file)
            analysis_all = db.select_all(Analysis)
            analyses = []

            for analysis in analysis_all:
                if not analysis.deleted:
                    localtimezone = pytz.timezone(os.getenv("TIME_ZONE"))
                    analysis_datetime = datetime.strptime(str(analysis.datetime), "%Y-%m-%d %H:%M:%S.%f").astimezone(localtimezone).__format__(
                        "%b %d, %Y %H:%M:%S")

                    final_key_ranks = db.select_final_key_rank_json_from_analysis(KeyRank, analysis.id)
                    final_success_rates = db.select_final_success_rate_from_analysis(SuccessRate, analysis.id)
                    neural_network = db.select_from_analysis(NeuralNetwork, analysis.id)

                    analyses.append({
                        "id": analysis.id,
                        "datetime": analysis_datetime,
                        "dataset": analysis.dataset,
                        "settings": analysis.settings,
                        "elapsed_time": time.strftime('%H:%M:%S', time.gmtime(analysis.elapsed_time)),
                        "key_ranks": final_key_ranks,
                        "success_rates": final_success_rates,
                        "neural_network_name": "not ready" if neural_network is None else neural_network.model_name
                    })

            all_tables.append(analyses)
            all_tables_names.append(db_file)

    return render_template("tables.html", all_tables=all_tables, all_tables_names=all_tables_names)


@app.route('/search/<string:table_name>')
def search(table_name):
    if os.path.exists(databases_root_folder + table_name):

        db = ScaDatabase(databases_root_folder + table_name)
        analysis_all = db.select_all(Analysis)
        analyses = []

        hp = []

        for analysis in analysis_all:

            final_key_ranks = db.select_final_key_rank_json_from_analysis(KeyRank, analysis.id)

            if len(final_key_ranks) > 0:
                hyper_parameters = db.select_from_analysis(HyperParameter, analysis.id)
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
    db = ScaDatabase(databases_root_folder + table_name)

    sca_views = ScaViews(analysis_id, db)

    all_metric_plots = sca_views.metric_plots()

    all_accuracy_plots = sca_views.accuracy_plots()
    dash_app_accuracy.layout = html.Div(children=[all_accuracy_plots])
    all_loss_plots = sca_views.loss_plots()
    dash_app_loss.layout = html.Div(children=[all_loss_plots])
    all_key_rank_plots = sca_views.key_rank_plots()
    dash_app_key_ranks.layout = html.Div(children=[all_key_rank_plots])
    all_success_rate_plots = sca_views.success_rate_plots()
    dash_app_success_rates.layout = html.Div(children=[all_success_rate_plots])

    # get neural network information from database
    analysis = db.select_analysis(Analysis, analysis_id)

    # get neural network information from database
    neural_network_model = db.select_from_analysis(NeuralNetwork, analysis_id)

    # get training hyper-parameters information from database
    hyper_parameters = db.select_all_from_analysis(HyperParameter, analysis_id)
    training_hyper_parameters = []
    for hp in hyper_parameters:
        training_hyper_parameters.append(hp.hyper_parameters)

    hyper_parameter_search = []
    if len(hyper_parameters) > 1:
        hyper_parameter_search = db.select_from_analysis(HyperParameterSearch, analysis.id)

    # get leakage model information from database
    leakage_models = db.select_from_analysis(LeakageModel, analysis_id)
    leakage_model_parameters = leakage_models.leakage_model

    # get visualization plots
    all_visualization_plots = sca_views.visualization_plots()
    all_visualization_heatmap_plots = sca_views.visualization_plots_heatmap()

    # confusion matrix plot
    all_confusion_matrix_plots = sca_views.confusion_matrix_plots()

    return render_template("dashboard/result.html",
                           all_plots=all_metric_plots,
                           all_key_rank_plots=all_key_rank_plots,
                           all_success_rate_plots=all_success_rate_plots,
                           neural_network_description=neural_network_model.description,
                           training_hyper_parameters=training_hyper_parameters,
                           hyper_parameters=hyper_parameters,
                           hyper_parameter_search=hyper_parameter_search,
                           leakage_model_parameters=leakage_model_parameters,
                           all_visualization_plots=all_visualization_plots,
                           all_visualization_heatmap_plots=all_visualization_heatmap_plots,
                           all_confusion_matrix_plots=all_confusion_matrix_plots,
                           analysis=analysis)


@app.route('/datasets')
def datasets():
    plotly_plots = PlotlyPlots()
    all_plots = []

    for ds in datasets_dict:
        in_file = h5py.File(datasets_root_folder + ds, "r")
        profiling_samples = np.array(in_file['Profiling_traces/traces'], dtype=np.float64)

        all_plots.append({
            "title": ds,
            "layout_plotly": plotly_plots.get_plotly_layout("Samples", "Amplitude (mV)"),
            "plots": plotly_plots.create_line_plot(y=profiling_samples[0], line_name=ds)
        })

    return render_template("dashboard/datasets.html", all_plots=all_plots)


@app.route("/generate_script/<int:analysis_id>/<string:table_name>")
def gen_script(analysis_id, table_name):
    generate_script("script_aes_{}".format(analysis_id), databases_root_folder, table_name, analysis_id)
    return "ok"


@app.route("/generate_plot/<int:analysis_id>/<string:table_name>/<metric>")
def gen_plot(analysis_id, table_name, metric):
    db = ScaDatabase(databases_root_folder + table_name)

    if metric == "Guessing_Entropy":
        result_key_byte = db.select_values_from_analysis_json(KeyRank, analysis_id)
    elif metric == "Success_Rate":
        result_key_byte = db.select_values_from_analysis_json(SuccessRate, analysis_id)
    else:
        result_key_byte = []
        if metric == "accuracy" or metric == "loss":
            result_key_byte.append(db.select_values_from_metric(Metric, metric, analysis_id)[0])
            result_key_byte.append(db.select_values_from_metric(Metric, "val_" + metric, analysis_id)[0])
        else:
            result_key_byte.append(db.select_values_from_metric(Metric, metric, analysis_id)[0])
    my_dpi = 100
    plt.figure(figsize=(800 / my_dpi, 600 / my_dpi), dpi=my_dpi)
    if metric == "Guessing_Entropy" or metric == "Success_Rate":
        for res in result_key_byte:
            plt.plot(np.arange(res['report_interval'], (len(res['values']) + 1) * res['report_interval'], res['report_interval']),
                     res['values'],
                     label=res['metric'])
            plt.legend(loc='best', fontsize=13)
            plt.xlim([1, len(res['values']) * res['report_interval']])
            plt.ylabel(metric.replace("_", " "), fontsize=13)
            if metric == "Guessing_Entropy" or metric == "Success_Rate":
                plt.xlabel("Attack Traces", fontsize=13)
            else:
                plt.xlabel("Epochs", fontsize=13)
            plt.grid(ls='--')
        plt.savefig("resources/{}_{}_{}.png".format(metric, analysis_id, table_name.replace(".sqlite", "")), format="png")
    else:
        for res in result_key_byte:
            plt.plot(np.arange(1, len(res['values']) + 1, 1), res['values'], label=res['metric'])
            plt.legend(loc='best', fontsize=13)
            plt.xlim([1, len(res['values'])])
            plt.ylabel(metric.replace("_", " "), fontsize=13)
            plt.xlabel("Epochs", fontsize=13)
            plt.grid(ls='--')
        plt.savefig("resources/{}_{}_{}.png".format(metric, analysis_id, table_name.replace(".sqlite", "")), format="png")

    return "ok"


@app.route("/delete_analysis/<int:analysis_id>/<string:table_name>")
def delete_analysis(analysis_id, table_name):
    db = ScaDatabase(databases_root_folder + table_name)
    db.soft_delete_analysis_from_table(Analysis, analysis_id)

    return "ok"


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port='5001')
    dash_app_accuracy.run_server(debug=True)
    dash_app_loss.run_server(debug=True)
    dash_app_key_ranks.run_server(debug=True)
    dash_app_success_rates.run_server(debug=True)
