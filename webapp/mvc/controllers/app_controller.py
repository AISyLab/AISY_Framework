from aisy_database.db_select import *
from aisy_database.db_delete import *
from datetime import datetime
from webapp.mvc.views import views
import numpy as np
import time
import pytz
import os
import hiplot as hip
import dash_html_components as html


class AppController:

    def __init__(self):
        pass

    def get_databases_tables(self, databases_root_folder):
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

                        key_ranks = db_select.select_all_guessing_entropy_from_analysis(GuessingEntropy, analysis.id)
                        success_rates = db_select.select_all_success_rate_from_analysis(SuccessRate, analysis.id)
                        neural_network = db_select.select_from_analysis(NeuralNetwork, analysis.id)
                        leakage_model = db_select.select_from_analysis(LeakageModel, analysis.id)

                        analyses.append({
                            "analysis_id": analysis.id,
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

        return all_tables, all_tables_names

    def get_search(self, databases_root_folder, table_name):

        analyses = []
        if os.path.exists(f"{databases_root_folder}{table_name}"):

            db_select = DBSelect(f"{databases_root_folder}{table_name}")
            analysis_all = db_select.select_all(Analysis)

            hp = []

            for analysis in analysis_all:

                final_key_ranks = db_select.select_all_guessing_entropy_from_analysis(GuessingEntropy, analysis.id)

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

        return analyses

    def get_visualization_plots(self, db_select, analysis_id):

        analysis = db_select.select_analysis(Analysis, analysis_id)
        if "visualization" in analysis.settings:
            hyper_parameters = db_select.select_all_from_analysis(HyperParameter, analysis_id)
            all_visualization_plots = []
            for hp in hyper_parameters:
                all_visualization_plots.append(views.visualization_plots(analysis.id, db_select, hp))
            visualization_plots = html.Div([
                html.Div(children=[
                    html.Div(children=all_visualization_plots)
                ])
            ])
            visualization_heatmap_plots = views.visualization_plots_heatmap(analysis.id, db_select)
            return visualization_plots, visualization_heatmap_plots
        else:
            return [], []

    def get_training_settings(self, analysis):
        training_settings = {
            "Profiling Traces": analysis.settings["number_of_profiling_traces"],
            "Attack Traces": analysis.settings["number_of_attack_traces"],
            "Index of first sample": analysis.settings["first_sample"],
            "Number of samples": analysis.settings["number_of_samples"],
            "Key": analysis.settings["key"],
            "Good key": hex(analysis.settings["good_key"])
        }
        if not analysis.settings["use_grid_search"] or not analysis.settings["use_random_search"]:
            training_settings["Batch-size"] = analysis.settings["batch_size"]
            training_settings["Epochs"] = analysis.settings["epochs"]
        return training_settings

    def get_hyperparameter_table(self, hyper_parameters, analysis):

        hyper_parameters_table = []
        for i, hp in enumerate(hyper_parameters):
            hp_struct = {}

            filter_list = []
            stride_list = []
            kernel_list = []
            pooling_type_list = []
            pooling_size_list = []
            pooling_stride_list = []

            if "Val GE" in hp.hyperparameters:
                hp_struct["Val GE"] = hp.hyperparameters["Val GE"]
            if "Val SR" in hp.hyperparameters:
                hp_struct["Val SR"] = hp.hyperparameters["Val SR"]

            if "conv_layers" in hp.hyperparameters:
                conv_layers = hp.hyperparameters["conv_layers"]
                hp_struct["conv_layers"] = conv_layers
            else:
                conv_layers = 0
                if not analysis.settings["use_grid_search"] or not analysis.settings["use_random_search"]:
                    hp_struct["conv_layers"] = "-"

            for hp_key in hp.hyperparameters:
                if "filter" in hp_key and len(filter_list) < conv_layers:
                    filter_list.append(hp.hyperparameters[hp_key])
                elif "pooling_type" in hp_key and len(pooling_type_list) < conv_layers:
                    pooling_type_list.append(hp.hyperparameters[hp_key])
                elif "pooling_size" in hp_key and len(pooling_size_list) < conv_layers:
                    pooling_size_list.append(hp.hyperparameters[hp_key])
                elif "pooling_stride" in hp_key and len(pooling_stride_list) < conv_layers:
                    pooling_stride_list.append(hp.hyperparameters[hp_key])
                elif "stride" in hp_key and len(stride_list) < conv_layers:
                    stride_list.append(hp.hyperparameters[hp_key])
                elif "kernel" in hp_key and len(kernel_list) < conv_layers:
                    kernel_list.append(hp.hyperparameters[hp_key])
                else:
                    if "structure" not in hp_key:
                        if "layers" in hp_key and "conv_layers" not in hp_key:
                            hp_struct["dense_layers"] = hp.hyperparameters[hp_key]
                        if "neurons" in hp_key:
                            hp_struct[str(hp_key)] = hp.hyperparameters[hp_key]
                        if "activation" in hp_key:
                            hp_struct[str(hp_key)] = hp.hyperparameters[hp_key]
                        if "learning_rate" in hp_key:
                            hp_struct[str(hp_key)] = hp.hyperparameters[hp_key]
                        if "dropout_rate" in hp_key:
                            hp_struct[str(hp_key)] = hp.hyperparameters[hp_key]
                        if "optimizer" in hp_key:
                            hp_struct[str(hp_key)] = hp.hyperparameters[hp_key]

            if analysis.settings["use_grid_search"] or analysis.settings["use_random_search"]:
                if len(filter_list) > 0:
                    hp_struct["filters"] = filter_list[0]
                if len(stride_list) > 0:
                    hp_struct["strides"] = stride_list[0]
                if len(kernel_list) > 0:
                    hp_struct["kernels"] = kernel_list[0]
                if len(pooling_type_list):
                    hp_struct["pooling_types"] = pooling_type_list[0]
                if len(pooling_size_list) > 0:
                    hp_struct["pooling_sizes"] = pooling_size_list[0]
                if len(pooling_stride_list) > 0:
                    hp_struct["pooling_strides"] = pooling_stride_list[0]
            else:
                if len(filter_list) > 0:
                    hp_struct["filters"] = filter_list[0]
                else:
                    hp_struct["filters"] = "-"
                if len(stride_list) > 0:
                    hp_struct["strides"] = stride_list[0]
                else:
                    hp_struct["strides"] = "-"
                if len(kernel_list) > 0:
                    hp_struct["kernels"] = kernel_list[0]
                else:
                    hp_struct["kernels"] = "-"
                if len(pooling_type_list):
                    hp_struct["pooling_types"] = pooling_type_list[0]
                else:
                    hp_struct["pooling_types"] = "-"
                if len(pooling_size_list) > 0:
                    hp_struct["pooling_sizes"] = pooling_size_list[0]
                else:
                    hp_struct["pooling_sizes"] = "-"
                if len(pooling_stride_list) > 0:
                    hp_struct["pooling_strides"] = pooling_stride_list[0]
                else:
                    hp_struct["pooling_strides"] = "-"

            hp_struct["id"] = hp.id
            hyper_parameters_table.append(hp_struct)
        return hyper_parameters_table
