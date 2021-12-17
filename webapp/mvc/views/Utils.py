from aisy_database.db_tables import *
import numpy as np
import json


class Utils:

    def __init__(self, settings):
        self.settings = settings

    def get_min_max_ge(self, min_values, max_values, elem):
        if min_values is None and max_values is None:
            min_values = elem["values"].copy()
            max_values = elem["values"].copy()
        else:
            for i, v in enumerate(elem["values"].copy()):
                if v < min_values[i]:
                    min_values[i] = v
            for i, v in enumerate(elem["values"].copy()):
                if v > max_values[i]:
                    max_values[i] = v
        return min_values, max_values

    def get_rows(self, rows, metric):

        elem_values = []
        for row in rows:
            values_as_array = json.loads(row.values)
            values_list = []
            for index, value in values_as_array.items():
                values_list.append(values_as_array[str(index)])

            if f"{metric}" in row.label or f"val_{metric}" in row.label:
                elem_values.append({
                    "values": values_list,
                    "label": row.label,
                    "id": row.id
                })

        return elem_values

    def get_legend(self, elem):
        is_search = self.settings["use_grid_search"] or self.settings["use_random_search"]
        if "Best Model" in elem["label"] or "Ensemble" in elem["label"]:
            return True
        elif is_search is False:
            return True
        else:
            return False

    def get_color(self, elem):
        is_search = self.settings["use_grid_search"] or self.settings["use_random_search"]
        if "Best Model" in elem["label"] or "Ensemble" in elem["label"]:
            return None
        elif is_search is False:
            return None
        else:
            return "rgba(210,210,210,0.8)"

    def get_line_width(self, elem):
        is_search = self.settings["use_grid_search"] or self.settings["use_random_search"]
        if "Best Model" in elem["label"] or "Ensemble" in elem["label"]:
            return 3.0
        elif is_search is False:
            return 1.5
        else:
            return 1.0

    def get_label(self, elem, db_select, metric):
        is_search = self.settings["use_grid_search"] or self.settings["use_random_search"]
        if is_search is True:
            return elem["label"]
        else:
            hyper_parameters = db_select.select_all_from_analysis(HyperParameter, self.settings["analysis_id"])
            if len(hyper_parameters) > 1:
                if metric == "Guessing Entropy":
                    hp_id = db_select.select_hyperparameter_from_guessing_entropy(HyperParameterGuessingEntropy, elem['id'],
                                                                                  self.settings["analysis_id"]).hyperparameters_id
                elif metric == "Success Rate":
                    hp_id = db_select.select_hyperparameter_from_success_rate(HyperParameterSuccessRate, elem['id'],
                                                                              self.settings["analysis_id"]).hyperparameters_id
                else:
                    hp_id = db_select.select_hyperparameter_from_metric(HyperParameterMetric, elem['id'],
                                                                        self.settings["analysis_id"]).hyperparameters_id
                return f"{elem['label']} hp_id: {hp_id}"
            else:
                return elem["label"]

    def get_x(self, elem, metric):
        if metric == "Guessing Entropy" or metric == "Success Rate":
            return np.linspace(elem['report_interval'], len(elem['values']) * elem['report_interval'], len(elem['values']))
        else:
            return np.arange(1, self.settings["epochs"] + 1)
