from flask import Flask, render_template
from aisy_database.db_select import *
from aisy_database.db_delete import *
from custom.custom_datasets.datasets import *
from webapp.mvc.controllers.script_controller import *
from webapp.mvc.views import views
from webapp.mvc.controllers.app_controller import AppController
from webapp.mvc.controllers.figures_controller import FigureController
import os
import numpy as np
import flaskcode
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

dash_app_profiling_analyzer_ge = dash.Dash(
    __name__,
    server=app,
    routes_pathname_prefix='/dash/profiling_analyzer_ge/',
    external_stylesheets=[dbc.themes.CERULEAN]
)
dash_app_profiling_analyzer_ge.layout = html.Div(children=[])

dash_app_profiling_analyzer_sr = dash.Dash(
    __name__,
    server=app,
    routes_pathname_prefix='/dash/profiling_analyzer_sr/',
    external_stylesheets=[dbc.themes.CERULEAN]
)
dash_app_profiling_analyzer_sr.layout = html.Div(children=[])

dash_app_profiling_analyzer_nt = dash.Dash(
    __name__,
    server=app,
    routes_pathname_prefix='/dash/profiling_analyzer_nt/',
    external_stylesheets=[dbc.themes.CERULEAN]
)
dash_app_profiling_analyzer_nt.layout = html.Div(children=[])

dash_app_visualization = dash.Dash(
    __name__,
    server=app,
    routes_pathname_prefix='/dash/visualization/',
    external_stylesheets=[dbc.themes.CERULEAN]
)
dash_app_visualization.layout = html.Div(children=[])


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


databases_root_folder = "my_database_location"
datasets_root_folder = "my_datasets_location"
resources_root_folder = "my_resources_location"

def adapt_folder_path(path):
    if "/" in path and path[len(path) - 1] != "/":
        path += "/"
    if "\\" in path and path[len(path) - 1] != "\\":
        path += "\\"
    return path


@app.route('/databases')
def databases():
    app_controller = AppController()
    all_tables, all_tables_names = app_controller.get_databases_tables(adapt_folder_path(databases_root_folder))
    return render_template("tables.html", all_tables=all_tables, all_tables_names=all_tables_names)


@app.route('/search/<string:table_name>')
def search(table_name):
    app_controller = AppController()
    analyses = app_controller.get_search(adapt_folder_path(databases_root_folder), table_name)

    return render_template("dashboard/search.html", analyses=analyses)


@app.route('/result/<int:analysis_id>/<string:table_name>')
def result(analysis_id, table_name):
    db_select = DBSelect(adapt_folder_path(databases_root_folder) + table_name)

    analysis = db_select.select_analysis(Analysis, analysis_id)

    all_metric_plots = []
    if analysis.settings["use_early_stopping"]:
        for metric in analysis.settings["early_stopping"]["metrics"]:
            all_metric_plots = views.metric_early_stopping_plots(all_metric_plots, analysis.id, db_select, metric)
    all_accuracy_plots = views.metric_plots(analysis.id, db_select, "accuracy", "Accuracy")
    dash_app_accuracy.layout = html.Div(children=[all_accuracy_plots])
    all_loss_plots = views.metric_plots(analysis.id, db_select, "loss", "Loss")
    dash_app_loss.layout = html.Div(children=[all_loss_plots])
    all_key_rank_plots = views.metric_plots(analysis.id, db_select, "Guessing Entropy", "Guessing Entropy")
    dash_app_key_ranks.layout = html.Div(children=[all_key_rank_plots])
    all_success_rate_plots = views.metric_plots(analysis.id, db_select, "Success Rate", "Success Rate")
    dash_app_success_rates.layout = html.Div(children=[all_success_rate_plots])

    if analysis.settings["use_profiling_analyzer"]:
        all_profiling_analyzer_plots_ge = views.metric_profiling_analyzer_plots(analysis.id, db_select, "Guessing Entropy",
                                                                                "Guessing Entropy")
        all_profiling_analyzer_plots_sr = views.metric_profiling_analyzer_plots(analysis.id, db_select, "Success Rate", "Success Rate")
        all_profiling_analyzer_plots_nt = views.metric_profiling_analyzer_plots(analysis.id, db_select, "Number of Attack Traces",
                                                                                "Number of Attack Traces")
        dash_app_profiling_analyzer_ge.layout = html.Div(children=[all_profiling_analyzer_plots_ge])
        dash_app_profiling_analyzer_sr.layout = html.Div(children=[all_profiling_analyzer_plots_sr])
        dash_app_profiling_analyzer_nt.layout = html.Div(children=[all_profiling_analyzer_plots_nt])

    app_controller = AppController()
    all_visualization_plots, all_visualization_heatmap_plots = app_controller.get_visualization_plots(db_select, analysis_id)
    if "visualization" in analysis.settings:
        dash_app_visualization.layout = all_visualization_plots
    training_settings = app_controller.get_training_settings(analysis)
    hyper_parameters = db_select.select_all_from_analysis(HyperParameter, analysis_id)
    hyper_parameters_table = app_controller.get_hyperparameter_table(hyper_parameters, analysis)

    neural_network_model = db_select.select_from_analysis(NeuralNetwork, analysis_id)

    neural_network_description = None
    if neural_network_model is not None:
        neural_network_description = neural_network_model.description

    leakage_model = db_select.select_from_analysis(LeakageModel, analysis_id)
    leakage_model_parameters = leakage_model.leakage_model

    all_confusion_matrix_plots = views.confusion_matrix_plots(analysis.id, db_select)

    return render_template("dashboard/result.html",
                           all_plots=all_metric_plots,
                           all_key_rank_plots=all_key_rank_plots,
                           all_success_rate_plots=all_success_rate_plots,
                           neural_network_description=neural_network_description,
                           training_settings=training_settings,
                           hyper_parameters_table=hyper_parameters_table,
                           leakage_model_parameters=leakage_model_parameters,
                           all_visualization_heatmap_plots=all_visualization_heatmap_plots,
                           all_confusion_matrix_plots=all_confusion_matrix_plots,
                           analysis=analysis)


@app.route('/documentation')
def documentation():
    return render_template("dashboard/documentation.html")


@app.route("/generate_reproducible_script/<int:analysis_id>/<string:table_name>")
def generate_reproducible_script(analysis_id, table_name):
    write_reproducible_script(f"script_reproducible_{analysis_id}", adapt_folder_path(databases_root_folder), table_name, analysis_id)
    return "ok"


@app.route("/generate_fully_reproducible_script/<int:analysis_id>/<string:table_name>")
def generate_fully_reproducible_script(analysis_id, table_name):
    write_fully_reproducible_script(f"script_fully_reproducible_{analysis_id}", adapt_folder_path(databases_root_folder), table_name, analysis_id)
    write_fully_reproducible_script(f"script_fully_reproducible_from_db_{analysis_id}", adapt_folder_path(databases_root_folder), table_name, analysis_id,
                                    from_db=True)
    return "ok"


@app.route("/generate_plot/<int:analysis_id>/<string:table_name>/<metric>")
def gen_plot(analysis_id, table_name, metric):
    dir_analysis_id = f"{adapt_folder_path(resources_root_folder)}figures/{table_name}_{analysis_id}"
    if not os.path.exists(dir_analysis_id):
        os.makedirs(dir_analysis_id)

    db_select = DBSelect(adapt_folder_path(databases_root_folder) + table_name)
    figure_controller = FigureController(dir_analysis_id, analysis_id, db_select)

    if metric == "accuracy":
        figure_controller.save_accuracy_plot()
    if metric == "loss":
        figure_controller.save_loss_plot()
    if metric == "guessing_entropy":
        figure_controller.save_guessing_entropy_plot()
    if metric == "success_rate":
        figure_controller.save_success_rate_plot()
    if metric == "visualization":
        hyper_parameters = db_select.select_all_from_analysis(HyperParameter, analysis_id)
        for hp in hyper_parameters:
            figure_controller.save_visualization_plot(hp.id)
    if metric == "visualization_heatmap":
        hyper_parameters = db_select.select_all_from_analysis(HyperParameter, analysis_id)
        for hp in hyper_parameters:
            figure_controller.save_visualization_heatmap_plot(hp.id)
    if metric == "pa_guessing_entropy":
        figure_controller.save_profiling_analyzer_guessing_entropy_plot()
    if metric == "pa_success_rate":
        figure_controller.save_profiling_analyzer_success_rate_plot()
    if metric == "pa_number_of_traces":
        figure_controller.save_profiling_analyzer_number_of_traces_plot()
    return "ok"


@app.route("/delete_analysis/<int:analysis_id>/<string:table_name>")
def delete_analysis(analysis_id, table_name):
    db_delete = DBDelete(adapt_folder_path(databases_root_folder) + table_name)
    db_delete.soft_delete_analysis_from_table(Analysis, analysis_id)

    return "ok"


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port='5001')
    dash_app_accuracy.run_server(debug=True)
    dash_app_loss.run_server(debug=True)
    dash_app_key_ranks.run_server(debug=True)
    dash_app_success_rates.run_server(debug=True)
    dash_app_profiling_analyzer_ge.run_server(debug=True)
    dash_app_profiling_analyzer_sr.run_server(debug=True)
    dash_app_profiling_analyzer_nt.run_server(debug=True)
    dash_app_visualization.run_server(debug=True)
