from commons.sca_tables import *
from commons.plotly.PlotlyPlots import PlotlyPlots
import numpy as np


class ScaViews:

    def __init__(self, analysis_id, db):
        self.db = db
        self.analysis_id = analysis_id

    def accuracy_plots(self):

        accuracy_metrics = ["accuracy", "val_accuracy"]
        hyper_parameters = self.db.select_all_from_analysis(HyperParameter, self.analysis_id)
        data = []
        show_legend = True

        for index in range(len(hyper_parameters)):
            for metric in accuracy_metrics:
                if len(hyper_parameters) > 1:
                    metric_name = metric + "_{}".format(index)
                    if index > 0:
                        show_legend = False
                else:
                    metric_name = metric
                metric_values = self.db.select_values_from_metric(Metric, metric_name, self.analysis_id)
                if len(metric_values) > 0:
                    for metric_value in metric_values:
                        data.append(
                            {'x': np.linspace(1, len(metric_value['values']), len(metric_value['values'])),
                             'y': metric_value['values'],
                             'type': 'line',
                             'name': metric_name,
                             'showlegend': show_legend,
                             'line': {}
                             }
                        )
                        if index != len(hyper_parameters) - 1:
                            data[len(data) - 1]['line']['color'] = "ababab"
                            data[len(data) - 1]['line']['width'] = 1.5
                        else:
                            data[len(data) - 1]['line']['color'] = "009688" if "val_accuracy" in metric else "b71c1c"
                            data[len(data) - 1]['line']['width'] = 2.5

        for metric in accuracy_metrics:
            metric_name = metric + "_best"
            metric_values = self.db.select_values_from_metric(Metric, metric_name, self.analysis_id)
            if len(metric_values) > 0:
                for metric_value in metric_values:
                    data.append(
                        {'x': np.linspace(1, len(metric_value['values']), len(metric_value['values'])),
                         'y': metric_value['values'],
                         'type': 'line',
                         'name': metric_name,
                         'showlegend': True,
                         'line': {}
                         }
                    )
                    data[len(data) - 1]['line']['color'] = "b71c1c"
                    data[len(data) - 1]['line']['width'] = 2.5

        return PlotlyPlots().create_line_plot_dash(data, "Epochs", "Accuracy")

    def loss_plots(self):

        loss_metrics = ["loss", "val_loss"]
        hyper_parameters = self.db.select_all_from_analysis(HyperParameter, self.analysis_id)
        data = []
        show_legend = True

        for index in range(len(hyper_parameters)):

            for metric in loss_metrics:
                if len(hyper_parameters) > 1:
                    metric_name = metric + "_{}".format(index)
                    if index > 0:
                        show_legend = False
                else:
                    metric_name = metric
                metric_values = self.db.select_values_from_metric(Metric, metric_name, self.analysis_id)
                if len(metric_values) > 0:
                    for metric_value in metric_values:
                        data.append(
                            {'x': np.linspace(1, len(metric_value['values']), len(metric_value['values'])),
                             'y': metric_value['values'],
                             'type': 'line',
                             'name': metric_name,
                             'showlegend': show_legend,
                             'line': {}
                             }
                        )

                        if index != len(hyper_parameters) - 1:
                            data[len(data) - 1]['line']['color'] = "ababab"
                            data[len(data) - 1]['line']['width'] = 1.5
                        else:
                            data[len(data) - 1]['line']['color'] = "009688" if "val_loss" in metric else "b71c1c"
                            data[len(data) - 1]['line']['width'] = 2.5

        for metric in loss_metrics:
            metric_name = metric + "_best"
            metric_values = self.db.select_values_from_metric(Metric, metric_name, self.analysis_id)
            if len(metric_values) > 0:
                for metric_value in metric_values:
                    data.append(
                        {'x': np.linspace(1, len(metric_value['values']), len(metric_value['values'])),
                         'y': metric_value['values'],
                         'type': 'line',
                         'name': metric_name,
                         'showlegend': True,
                         'line': {}
                         }
                    )
                    data[len(data) - 1]['line']['color'] = "b71c1c"
                    data[len(data) - 1]['line']['width'] = 2.5

        return PlotlyPlots().create_line_plot_dash(data, "Epochs", "Loss")

    def metric_plots(self):

        plotly_plots = PlotlyPlots()
        all_metric_plots = []
        hyper_parameters = self.db.select_all_from_analysis(HyperParameter, self.analysis_id)

        metrics = self.db.select_metrics(Metric, self.analysis_id)
        metric_plots = {}
        metric_names = []

        for metric in metrics:
            if not metric.startswith("accuracy") and not metric.startswith("loss") and not metric.startswith(
                    "val_accuracy") and not metric.startswith("val_loss") and "best" not in metric:
                if len(hyper_parameters) > 1:
                    metric_inverted = metric[::-1]
                    metric_without_index = metric_inverted.partition(" ")[2][::-1]
                    metric_plots[metric_without_index] = []
                    if metric_without_index not in metric_names:
                        metric_names.append(metric_without_index)
                else:
                    metric_plots[metric] = []
                    metric_names.append(metric)

        for metric_name in metric_names:

            best_epoch_metric = []
            histogram_best_epoch_metric_plots = []

            for index in range(len(hyper_parameters)):
                if len(hyper_parameters) > 1:
                    db_metric_name = "{} {}".format(metric_name, index)
                    if index == 0:
                        show_legend = True
                    else:
                        show_legend = False
                    line_color = "ababab"
                else:
                    db_metric_name = metric_name
                    show_legend = True
                    line_color = "b71c1c"
                metric_values = self.db.select_values_from_metric(Metric, db_metric_name, self.analysis_id)
                if len(metric_values) > 0:
                    for metric_value in metric_values:
                        metric_plots[metric_name].append(
                            plotly_plots.create_line_plot(y=metric_value['values'], line_name=metric_name, line_color=line_color,
                                                          show_legend=show_legend))
                        if len(hyper_parameters) > 1:
                            best_epoch_metric.append(np.argmax(metric_value['values']))

                if index == len(hyper_parameters) - 1:
                    metric_name_best = "{} best".format(metric_name)
                    metric_values = self.db.select_values_from_metric(Metric, metric_name_best, self.analysis_id)
                    line_color = "b71c1c"
                    if len(metric_values) > 0:
                        for metric_value in metric_values:
                            metric_plots[metric_name].append(
                                plotly_plots.create_line_plot(y=metric_value['values'], line_name=metric_name_best, line_color=line_color,
                                                              line_width=2.5))

                    if len(hyper_parameters) > 1:
                        histogram_best_epoch_metric_plots.append(
                            plotly_plots.create_hist_plot(x=best_epoch_metric, line_name="Best Epochs {}".format(metric_name))
                        )

            all_metric_plots.append({
                "title": metric_name,
                "layout_plotly": plotly_plots.get_plotly_layout("Epochs", metric_name),
                "plots": metric_plots[metric_name]
            })

            if len(hyper_parameters) > 1:
                all_metric_plots.append({
                    "title": "Best Epochs {}".format(metric_name),
                    "layout_plotly": plotly_plots.get_layout_density("Epochs", "Frequency"),
                    "plots": histogram_best_epoch_metric_plots
                })

        return all_metric_plots

    def key_rank_plots(self):
        data = []

        key_rank_metrics = self.db.select_key_rank_metrics(KeyRank, self.analysis_id)
        hyper_parameters = self.db.select_all_from_analysis(HyperParameter, self.analysis_id)

        if len(hyper_parameters) > 1:
            metric_names = []
            for metric in key_rank_metrics:
                if "Best" not in metric:
                    metric_inverted = metric[::-1]
                    metric_without_index = metric_inverted.partition(" ")[2][::-1]
                    if metric_without_index not in metric_names:
                        metric_names.append(metric_without_index)
        else:
            metric_names = key_rank_metrics

        for metric in metric_names:

            for index in range(len(hyper_parameters)):

                show_legend = True if index == 0 else False
                line_color = "b71c1c" if index == len(hyper_parameters) - 1 else "ababab"
                line_width = 2.5 if index == len(hyper_parameters) - 1 else 1.0
                line_name = metric
                if len(hyper_parameters) > 1:
                    if index == len(hyper_parameters) - 1:
                        db_metric_name = "{} Best Model".format(metric)
                        show_legend = True
                        line_name = "{} Best Model".format(metric)
                    else:
                        db_metric_name = "{} {}".format(metric, index)
                else:
                    db_metric_name = metric

                key_rank = self.db.select_key_rank_from_analysis_json(KeyRank, db_metric_name, self.analysis_id)

                if len(key_rank) > 0:
                    data.append(
                        {
                            'x': np.linspace(key_rank['report_interval'], len(key_rank['values']) * key_rank['report_interval'],
                                             len(key_rank['values'])),
                            'y': key_rank['values'],
                            'type': 'line',
                            'name': line_name,
                            'showlegend': show_legend,
                            'line': {}
                        }
                    )
                    if index != len(hyper_parameters) - 1:
                        data[len(data) - 1]['line']['color'] = line_color
                        data[len(data) - 1]['line']['width'] = line_width

        return PlotlyPlots().create_line_plot_dash(data, "Traces", "Guessing Entropy")

    def success_rate_plots(self):

        data = []

        key_rank_metrics = self.db.select_key_rank_metrics(SuccessRate, self.analysis_id)
        hyper_parameters = self.db.select_all_from_analysis(HyperParameter, self.analysis_id)

        if len(hyper_parameters) > 1:
            metric_names = []
            for metric in key_rank_metrics:
                if "Best" not in metric:
                    metric_inverted = metric[::-1]
                    metric_without_index = metric_inverted.partition(" ")[2][::-1]
                    if metric_without_index not in metric_names:
                        metric_names.append(metric_without_index)
        else:
            metric_names = key_rank_metrics

        for metric in metric_names:

            for index in range(len(hyper_parameters)):

                show_legend = True if index == 0 else False
                line_color = "b71c1c" if index == len(hyper_parameters) - 1 else "ababab"
                line_width = 2.5 if index == len(hyper_parameters) - 1 else 1.0
                line_name = metric
                if len(hyper_parameters) > 1:
                    if index == len(hyper_parameters) - 1:
                        db_metric_name = "{} Best Model".format(metric)
                        show_legend = True
                        line_name = "{} Best Model".format(metric)
                    else:
                        db_metric_name = "{} {}".format(metric, index)
                else:
                    db_metric_name = metric

                success_rate = self.db.select_success_rate_from_analysis_json(SuccessRate, db_metric_name, self.analysis_id)

                if len(success_rate) > 0:
                    data.append(
                        {
                            'x': np.linspace(success_rate['report_interval'], len(success_rate['values']) * success_rate['report_interval'],
                                             len(success_rate['values'])),
                            'y': success_rate['values'],
                            'type': 'line',
                            'name': line_name,
                            'showlegend': show_legend,
                            'line': {}
                        }
                    )
                    if index != len(hyper_parameters) - 1:
                        data[len(data) - 1]['line']['color'] = line_color
                        data[len(data) - 1]['line']['width'] = line_width

        return PlotlyPlots().create_line_plot_dash(data, "Traces", "Success Rate")

    def visualization_plots(self):
        plotly_plots = PlotlyPlots()

        all_visualization_plots = []
        visualization_plots = []

        visualization_rows = self.db.select_values_from_analysis_json(Visualization, self.analysis_id)

        if len(visualization_rows) > 0:
            visualization_plots_metrics = []
            visualization = visualization_rows[len(visualization_rows) - 1]
            visualization_plots_metrics.append(
                plotly_plots.create_line_plot(y=visualization['values'], line_name=str(visualization['key_byte']) + " IG - Sum"))
            visualization_plots.append(visualization_plots_metrics)

            all_visualization_plots.append({
                "title": "Visualization",
                "layout_plotly": plotly_plots.get_plotly_layout("Samples", "Gradient"),
                "plots": visualization_plots
            })

        return all_visualization_plots

    def visualization_plots_heatmap(self):
        plotly_plots = PlotlyPlots()

        all_visualization_plots = []
        visualization_plots = []

        visualization_rows = self.db.select_values_from_analysis_json(Visualization, self.analysis_id)
        if len(visualization_rows) > 0:
            visualization_plots_metrics = []
            z = []
            x = list(range(len(visualization_rows[0]['values'])))
            y = []
            epoch = 0
            for visualization in visualization_rows:
                z.append(visualization['values'])
                y.append(epoch)
                epoch += 1
            visualization_plots_metrics.append(plotly_plots.create_heatmap(x=x, y=y, z=z))
            visualization_plots.append(visualization_plots_metrics)

            all_visualization_plots.append({
                "title": "Visualization",
                "layout_plotly": plotly_plots.get_plotly_layout("Samples", "Epoch"),
                "plots": visualization_plots
            })

        return all_visualization_plots

    def confusion_matrix_plots(self):
        plotly_plots = PlotlyPlots()

        all_confusion_matrix_plots = []
        confusion_matrix_plots = []

        confusion_matrix_all_key_bytes = self.db.select_values_from_confusion_matrix_json(ConfusionMatrix, self.analysis_id)
        if len(confusion_matrix_all_key_bytes) > 0:
            z = []
            y = []
            x = list(range(len(confusion_matrix_all_key_bytes[0]['values'])))
            confusion_matrix_plots_metrics = []

            for y_true, confusion_matrix_key_byte in enumerate(confusion_matrix_all_key_bytes):
                z.append(confusion_matrix_key_byte['values'])
                y.append(y_true)
            if len(x) > 9:
                confusion_matrix_plots_metrics.append(plotly_plots.create_heatmap(x=x, y=y, z=z))
            else:
                confusion_matrix_plots_metrics.append(plotly_plots.create_annotated_heatmap(x=x, y=y, z=z))
            confusion_matrix_plots.append(confusion_matrix_plots_metrics)

            all_confusion_matrix_plots.append({
                "title": "Confusion Matrix",
                "layout_plotly": plotly_plots.get_plotly_layout("Y_true", "Y_pred"),
                "plots": confusion_matrix_plots
            })

        return all_confusion_matrix_plots
