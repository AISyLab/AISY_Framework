from aisy_database.db_tables import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class FigureController:

    def __init__(self, dir_analysis_id, analysis_id, db_select):
        self.dir_analysis_id = dir_analysis_id
        self.analysis_id = analysis_id
        self.db_select = db_select

    def save_accuracy_plot(self):
        analysis = self.db_select.select_analysis(Analysis, self.analysis_id)
        accuracy_rows = self.db_select.select_accuracy_from_analysis(self.analysis_id)
        best_accuracy_rows = self.db_select.select_best_accuracy_from_analysis(self.analysis_id)
        all_rows = []

        x_range = np.arange(1, analysis.settings["epochs"] + 1)
        plt_obj = plt
        for row in accuracy_rows:
            if len(best_accuracy_rows) > 0:
                plt_obj.plot(x_range, row["values"], color="lightgrey", linewidth=0.5)
            else:
                plt_obj.plot(x_range, row["values"], linewidth=1.5, label=row["label"])
            all_rows.append(row["values"])
        for row in best_accuracy_rows:
            plt_obj.plot(x_range, row["values"], linewidth=1.5, label=row["label"])
            all_rows.append(row["values"])

            plt_obj.fill_between(x_range, np.min(all_rows, axis=0), np.max(all_rows, axis=0), color="lightgrey", alpha=0.05)

        self.save_figure_to_png(plt_obj, "Epochs", "Accuracy", [1, analysis.settings["epochs"]], f"accuracy_{self.analysis_id}")

    def save_loss_plot(self):
        analysis = self.db_select.select_analysis(Analysis, self.analysis_id)
        loss_rows = self.db_select.select_loss_from_analysis(self.analysis_id)
        best_loss_rows = self.db_select.select_best_loss_from_analysis(self.analysis_id)
        all_rows = []

        x_range = np.arange(1, analysis.settings["epochs"] + 1)
        plt_obj = plt
        for row in loss_rows:
            if len(best_loss_rows) > 0:
                plt_obj.plot(x_range, row["values"], color="lightgrey", linewidth=0.5)
            else:
                plt_obj.plot(x_range, row["values"], linewidth=1.5, label=row["label"])
            all_rows.append(row["values"])
        for row in best_loss_rows:
            plt_obj.plot(x_range, row["values"], linewidth=1.5, label=row["label"])
            all_rows.append(row["values"])

            plt_obj.fill_between(x_range, np.min(all_rows, axis=0), np.max(all_rows, axis=0), color="lightgrey", alpha=0.05)

        self.save_figure_to_png(plt_obj, "Epochs", "Loss", [1, analysis.settings["epochs"]], f"loss_{self.analysis_id}")

    def save_guessing_entropy_plot(self):
        analysis = self.db_select.select_analysis(Analysis, self.analysis_id)
        ge_rows = self.db_select.select_guessing_entropy_from_analysis(self.analysis_id)
        best_ge_rows = self.db_select.select_best_guessing_entropy_from_analysis(self.analysis_id)
        all_rows = []

        x_range = np.arange(analysis.settings["key_rank_report_interval"],
                            analysis.settings["key_rank_attack_traces"] + analysis.settings["key_rank_report_interval"],
                            analysis.settings["key_rank_report_interval"])
        plt_obj = plt
        for row in ge_rows:
            if len(best_ge_rows) > 0:
                plt_obj.plot(x_range, row["values"], color="lightgrey", linewidth=0.5)
            else:
                plt_obj.plot(x_range, row["values"], linewidth=1.5,
                             label=row["label"])
            all_rows.append(row["values"])
        for row in best_ge_rows:
            plt_obj.plot(x_range, row["values"], linewidth=1.5, label=row["label"])
            all_rows.append(row["values"])

            plt_obj.fill_between(x_range, np.min(all_rows, axis=0), np.max(all_rows, axis=0), color="lightgrey", alpha=0.05)

        x_lim = [analysis.settings["key_rank_report_interval"], analysis.settings["key_rank_attack_traces"]]
        self.save_figure_to_png(plt_obj, "Traces", "Guessing Entropy", x_lim, f"guessing_entropy_{self.analysis_id}")

    def save_success_rate_plot(self):
        analysis = self.db_select.select_analysis(Analysis, self.analysis_id)
        sr_rows = self.db_select.select_success_rate_from_analysis(self.analysis_id)
        best_sr_rows = self.db_select.select_best_success_rate_from_analysis(self.analysis_id)
        all_rows = []

        x_range = np.arange(analysis.settings["key_rank_report_interval"],
                            analysis.settings["key_rank_attack_traces"] + analysis.settings["key_rank_report_interval"],
                            analysis.settings["key_rank_report_interval"])
        plt_obj = plt
        for row in sr_rows:
            if len(best_sr_rows) > 0:
                plt_obj.plot(x_range, row["values"], color="lightgrey", linewidth=0.5)
            else:
                plt_obj.plot(x_range, row["values"], linewidth=1.5, label=row["label"])
            all_rows.append(row["values"])
        for row in best_sr_rows:
            plt_obj.plot(x_range, row["values"], linewidth=1.5, label=row["label"])
            all_rows.append(row["values"])

            plt_obj.fill_between(x_range, np.min(all_rows, axis=0), np.max(all_rows, axis=0), color="lightgrey", alpha=0.05)

        x_lim = [analysis.settings["key_rank_report_interval"], analysis.settings["key_rank_attack_traces"]]
        self.save_figure_to_png(plt_obj, "Traces", "Success Rate", x_lim, f"success_rate_{self.analysis_id}")

    def save_visualization_plot(self, hp_id):
        analysis = self.db_select.select_analysis(Analysis, self.analysis_id)
        visualization_rows = self.db_select.select_visualization_from_analysis(Visualization, hp_id, self.analysis_id)

        x_range = np.arange(1, analysis.settings["number_of_samples"] + 1)
        plt_obj = plt
        for idx in range(analysis.settings["epochs"]):
            plt_obj.plot(x_range, visualization_rows[idx]["values"], color="lightgrey", linewidth=0.5)
        plt_obj.plot(x_range, visualization_rows[analysis.settings["epochs"] - 1]["values"], color="blue", linewidth=1,
                     label=f"IG Epoch {analysis.settings['epochs']}")
        x_lim = [1, analysis.settings["number_of_samples"]]
        self.save_figure_to_png(plt_obj, "Samples", "Input Gradient", x_lim, f"input_gradient_hp_{hp_id}_{self.analysis_id}")

    def save_visualization_heatmap_plot(self, hp_id):
        analysis = self.db_select.select_analysis(Analysis, self.analysis_id)
        visualization_rows = self.db_select.select_visualization_from_analysis(Visualization, hp_id, self.analysis_id)
        all_rows = []
        plt_obj = plt

        figure = plt_obj.gcf()
        figure.set_size_inches(6, 4)
        x_lim = [1, analysis.settings["number_of_samples"]]
        plt_obj.xlim(x_lim)

        for idx in range(analysis.settings["epochs"]):
            all_rows.append(visualization_rows[idx]["values"])
        plt_obj.imshow(all_rows, cmap='viridis', interpolation='nearest', aspect='auto')
        plt_obj.colorbar()
        plt_obj.xlabel("Samples", fontsize=14)
        plt_obj.ylabel("Epoch", fontsize=14)
        plt_obj.tight_layout()
        plt_obj.savefig(f"{self.dir_analysis_id}/input_gradient_heatmap_hp_{hp_id}_{self.analysis_id}.png", dpi=300)
        plt_obj.close()

    def save_profiling_analyzer_guessing_entropy_plot(self):
        analysis = self.db_select.select_analysis(Analysis, self.analysis_id)
        steps = analysis.settings["profiling_analyzer_steps"]

        elem_values = self.db_select.select_all_guessing_entropy_from_analysis(GuessingEntropy, self.analysis_id)

        profiling_traces_list = []
        metric_list = []
        color_list = []
        ge_list = []

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

        for n_profiling_traces in steps:
            for elem in elem_values:
                if "values" in elem:
                    if number_of_searches is not None:
                        for search_index in range(number_of_searches):
                            for data_set in ["Validation", "Attack"]:
                                if f"{data_set} Set {search_index} {n_profiling_traces} traces" in elem["label"]:
                                    profiling_traces_list.append(n_profiling_traces)
                                    ge_list.append(elem['values'][len(elem['values']) - 1])
                                    metric_list.append(f"{data_set} Set")
                                    color_list.append("blue" if data_set == "Validation" else "red")
                            if analysis.settings["use_early_stopping"]:
                                for es_metric in analysis.settings["early_stopping"]["metrics"]:
                                    for data_set in ["Validation", "Attack"]:
                                        if f"ES {data_set} Set {es_metric} {search_index} {n_profiling_traces} traces" in elem["label"]:
                                            profiling_traces_list.append(n_profiling_traces)
                                            ge_list.append(elem['values'][len(elem['values']) - 1])
                                            metric_list.append(f"ES {data_set} Set")
                                            color_list.append("green" if data_set == "Validation" else "orange")
                    for data_set in ["Validation", "Attack"]:
                        if f"{data_set} Set Best Model {n_profiling_traces} traces" in elem["label"]:
                            best_model[data_set]["x"].append(n_profiling_traces)
                            best_model[data_set]["y"].append(elem['values'][len(elem['values']) - 1])
                    if analysis.settings["use_early_stopping"]:
                        for es_metric in analysis.settings["early_stopping"]["metrics"]:
                            for data_set in ["Validation", "Attack"]:
                                if f"ES {data_set} Set {es_metric} Best Model {n_profiling_traces} traces" in elem["label"]:
                                    best_model[f"{es_metric} {data_set}"]["x"].append(n_profiling_traces)
                                    best_model[f"{es_metric} {data_set}"]["y"].append(elem['values'][len(elem['values']) - 1])

        # data = {
        #     "ProfilingTraces": profiling_traces_list,
        #     "GE": ge_list,
        #     "metric": metric_list,
        #     "Set": color_list,
        # }
        # dataframe = pd.DataFrame(data)
        # plt.scatter(dataframe.ProfilingTraces, dataframe.GE, s=10, c=dataframe.Set)

        plt.plot(best_model["Validation"]["x"], best_model["Validation"]["y"], label="Best Model Validation", linewidth=1.0)
        plt.plot(best_model["Attack"]["x"], best_model["Attack"]["y"], label="Best Model Attack", linewidth=1.5)

        if analysis.settings["use_early_stopping"]:
            for es_metric in analysis.settings["early_stopping"]["metrics"]:
                plt.plot(best_model[f"{es_metric} Validation"]["x"], best_model[f"{es_metric} Validation"]["y"],
                         label=f"Best Model {es_metric} Validation", linestyle="--", linewidth=1.0)
                plt.plot(best_model[f"{es_metric} Attack"]["x"], best_model[f"{es_metric} Attack"]["y"],
                         label=f"Best Model {es_metric} Attack",
                         linestyle="--", linewidth=1.5)

        figure = plt.gcf()
        figure.set_size_inches(6, 2.5)
        plt.grid()
        plt.xlabel("Profiling Traces")
        plt.ylabel("Guessing Entropy")
        plt.xlim([min(profiling_traces_list), max(profiling_traces_list)])
        plt.xticks(steps, steps)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.dir_analysis_id}/pa_guessing_entropy_{self.analysis_id}.png", dpi=300)
        plt.close()

    def save_profiling_analyzer_success_rate_plot(self):
        analysis = self.db_select.select_analysis(Analysis, self.analysis_id)
        steps = analysis.settings["profiling_analyzer_steps"]

        elem_values = self.db_select.select_all_success_rate_from_analysis(SuccessRate, self.analysis_id)

        profiling_traces_list = []
        metric_list = []
        color_list = []
        ge_list = []

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

        for n_profiling_traces in steps:
            for elem in elem_values:
                if "values" in elem:
                    if number_of_searches is not None:
                        for search_index in range(number_of_searches):
                            for data_set in ["Validation", "Attack"]:
                                if f"{data_set} Set {search_index} {n_profiling_traces} traces" in elem["label"]:
                                    profiling_traces_list.append(n_profiling_traces)
                                    ge_list.append(elem['values'][len(elem['values']) - 1])
                                    metric_list.append(f"{data_set} Set")
                                    color_list.append("blue" if data_set == "Validation" else "red")
                            if analysis.settings["use_early_stopping"]:
                                for es_metric in analysis.settings["early_stopping"]["metrics"]:
                                    for data_set in ["Validation", "Attack"]:
                                        if f"ES {data_set} Set {es_metric} {search_index} {n_profiling_traces} traces" in elem["label"]:
                                            profiling_traces_list.append(n_profiling_traces)
                                            ge_list.append(elem['values'][len(elem['values']) - 1])
                                            metric_list.append(f"ES {data_set} Set")
                                            color_list.append("green" if data_set == "Validation" else "orange")
                    for data_set in ["Validation", "Attack"]:
                        if f"{data_set} Set Best Model {n_profiling_traces} traces" in elem["label"]:
                            best_model[data_set]["x"].append(n_profiling_traces)
                            best_model[data_set]["y"].append(elem['values'][len(elem['values']) - 1])
                    if analysis.settings["use_early_stopping"]:
                        for es_metric in analysis.settings["early_stopping"]["metrics"]:
                            for data_set in ["Validation", "Attack"]:
                                if f"ES {data_set} Set {es_metric} Best Model {n_profiling_traces} traces" in elem["label"]:
                                    best_model[f"{es_metric} {data_set}"]["x"].append(n_profiling_traces)
                                    best_model[f"{es_metric} {data_set}"]["y"].append(elem['values'][len(elem['values']) - 1])

        # data = {
        #     "ProfilingTraces": profiling_traces_list,
        #     "GE": ge_list,
        #     "metric": metric_list,
        #     "Set": color_list,
        # }
        # dataframe = pd.DataFrame(data)
        # plt.scatter(dataframe.ProfilingTraces, dataframe.GE, s=10, c=dataframe.Set)

        plt.plot(best_model["Validation"]["x"], best_model["Validation"]["y"], label="Best Model Validation", linewidth=1.0)
        plt.plot(best_model["Attack"]["x"], best_model["Attack"]["y"], label="Best Model Attack", linewidth=1.5)

        if analysis.settings["use_early_stopping"]:
            for es_metric in analysis.settings["early_stopping"]["metrics"]:
                plt.plot(best_model[f"{es_metric} Validation"]["x"], best_model[f"{es_metric} Validation"]["y"],
                         label=f"Best Model {es_metric} Validation", linestyle="--", linewidth=1.0)
                plt.plot(best_model[f"{es_metric} Attack"]["x"], best_model[f"{es_metric} Attack"]["y"],
                         label=f"Best Model {es_metric} Attack",
                         linestyle="--", linewidth=1.5)

        figure = plt.gcf()
        figure.set_size_inches(6, 2.5)
        plt.grid()
        plt.xlabel("Profiling Traces")
        plt.ylabel("Success Rate")
        plt.xlim([min(profiling_traces_list), max(profiling_traces_list)])
        plt.xticks(steps, steps)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.dir_analysis_id}/pa_success_rate_{self.analysis_id}.png", dpi=300)
        plt.close()

    def save_profiling_analyzer_number_of_traces_plot(self):
        analysis = self.db_select.select_analysis(Analysis, self.analysis_id)
        steps = analysis.settings["profiling_analyzer_steps"]

        elem_values = self.db_select.select_all_guessing_entropy_from_analysis(GuessingEntropy, self.analysis_id)
        for elem in elem_values:
            if elem['values'][len(elem['values']) - 1] < 2:
                elem['values'] = analysis.settings["key_rank_attack_traces"] - np.searchsorted(elem["values"][::-1], 2) * \
                                 analysis.settings["key_rank_report_interval"]
            else:
                elem['values'] = analysis.settings["key_rank_attack_traces"]

        profiling_traces_list = []
        metric_list = []
        color_list = []
        ge_list = []

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

        for n_profiling_traces in steps:
            for elem in elem_values:
                if "values" in elem:
                    if number_of_searches is not None:
                        for search_index in range(number_of_searches):
                            for data_set in ["Validation", "Attack"]:
                                if f"{data_set} Set {search_index} {n_profiling_traces} traces" in elem["label"]:
                                    profiling_traces_list.append(n_profiling_traces)
                                    ge_list.append(elem['values'])
                                    metric_list.append(f"{data_set} Set")
                                    color_list.append("blue" if data_set == "Validation" else "red")
                            if analysis.settings["use_early_stopping"]:
                                for es_metric in analysis.settings["early_stopping"]["metrics"]:
                                    for data_set in ["Validation", "Attack"]:
                                        if f"ES {data_set} Set {es_metric} {search_index} {n_profiling_traces} traces" in elem["label"]:
                                            profiling_traces_list.append(n_profiling_traces)
                                            ge_list.append(elem['values'])
                                            metric_list.append(f"ES {data_set} Set")
                                            color_list.append("green" if data_set == "Validation" else "orange")
                    for data_set in ["Validation", "Attack"]:
                        if f"{data_set} Set Best Model {n_profiling_traces} traces" in elem["label"]:
                            best_model[data_set]["x"].append(n_profiling_traces)
                            best_model[data_set]["y"].append(elem['values'])
                    if analysis.settings["use_early_stopping"]:
                        for es_metric in analysis.settings["early_stopping"]["metrics"]:
                            for data_set in ["Validation", "Attack"]:
                                if f"ES {data_set} Set {es_metric} Best Model {n_profiling_traces} traces" in elem["label"]:
                                    best_model[f"{es_metric} {data_set}"]["x"].append(n_profiling_traces)
                                    best_model[f"{es_metric} {data_set}"]["y"].append(elem['values'])

        # data = {
        #     "ProfilingTraces": profiling_traces_list,
        #     "GE": ge_list,
        #     "metric": metric_list,
        #     "Set": color_list,
        # }
        # dataframe = pd.DataFrame(data)
        # plt.scatter(dataframe.ProfilingTraces, dataframe.GE, s=10, c=dataframe.Set)

        plt.plot(best_model["Validation"]["x"], best_model["Validation"]["y"], label="Best Model Validation", linewidth=1.0)
        plt.plot(best_model["Attack"]["x"], best_model["Attack"]["y"], label="Best Model Attack", linewidth=1.5)

        if analysis.settings["use_early_stopping"]:
            for es_metric in analysis.settings["early_stopping"]["metrics"]:
                plt.plot(best_model[f"{es_metric} Validation"]["x"], best_model[f"{es_metric} Validation"]["y"],
                         label=f"Best Model {es_metric} Validation", linestyle="--", linewidth=1.0)
                plt.plot(best_model[f"{es_metric} Attack"]["x"], best_model[f"{es_metric} Attack"]["y"],
                         label=f"Best Model {es_metric} Attack",
                         linestyle="--", linewidth=1.5)

        figure = plt.gcf()
        figure.set_size_inches(6, 2.5)
        plt.grid()
        plt.xlabel("Profiling Traces")
        plt.ylabel("Attack Traces for GE = 1")
        plt.xlim([min(profiling_traces_list), max(profiling_traces_list)])
        plt.xticks(steps, steps)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.dir_analysis_id}/pa_number_of_traces_{self.analysis_id}.png", dpi=300)
        plt.close()

    def save_figure_to_png(self, plt_obj, x_label, y_label, x_lim, figure_name, legend=True):
        figure = plt_obj.gcf()
        figure.set_size_inches(6, 4)
        plt_obj.grid()
        plt_obj.xlabel(x_label, fontsize=14)
        plt_obj.ylabel(y_label, fontsize=14)
        plt_obj.xlim(x_lim)
        if legend:
            plt_obj.legend()
        plt_obj.tight_layout()
        plt_obj.savefig(f"{self.dir_analysis_id}/{figure_name}.png", dpi=300)
        plt_obj.close()
