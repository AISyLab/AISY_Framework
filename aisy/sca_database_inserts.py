from aisy.sca_database import ScaDatabase
from aisy.sca_tables import *
import aisy.sca_tables as tables
import custom.custom_tables.tables as custom_tables


class ScaDatabaseInserts:

    def __init__(self, database_name, db_filename, dataset, settings, elapsed_time):
        self.db = ScaDatabase(database_name)
        tables.base().metadata.create_all(self.db.engine)
        custom_tables.base().metadata.create_all(self.db.engine)
        new_insert = Analysis(db_filename=db_filename, dataset=dataset, settings=settings, elapsed_time=elapsed_time, deleted=False)
        self.analysis_id = self.db.insert(new_insert)

    def get_analysis_id(self):
        return self.analysis_id

    def update_elapsed_time_analysis(self, elapsed_time):
        self.db.session.query(Analysis).filter(Analysis.id == self.analysis_id).update({"elapsed_time": elapsed_time})
        self.db.session.commit()

    def save_hyper_parameters(self, hyper_parameters):
        new_insert = HyperParameter(hyper_parameters=hyper_parameters, analysis_id=self.analysis_id)
        return self.db.insert(new_insert)

    def save_neural_network(self, description, model_name):
        new_insert = NeuralNetwork(model_name=model_name, description=description, analysis_id=self.analysis_id)
        return self.db.insert(new_insert)

    def save_leakage_model(self, leakage_model):
        new_insert = LeakageModel(leakage_model=leakage_model, analysis_id=self.analysis_id)
        return self.db.insert(new_insert)

    def save_metric(self, data, key_byte, metric):
        for value in data:
            new_insert = Metric(value=value, key_byte=key_byte, metric=metric, analysis_id=self.analysis_id)
            self.db.insert(new_insert)

    def save_key_rank_json(self, values, key_byte, report_interval, metric):
        new_insert = KeyRank(values=values, key_byte=key_byte, report_interval=report_interval, metric=metric,
                             analysis_id=self.analysis_id)
        self.db.insert(new_insert)

    def save_success_rate_json(self, values, key_byte, report_interval, metric):
        new_insert = SuccessRate(values=values, key_byte=key_byte, report_interval=report_interval, metric=metric,
                                 analysis_id=self.analysis_id)
        self.db.insert(new_insert)

    def save_visualization(self, values, epoch, key_byte, report_interval, metric):
        new_insert = Visualization(values=values, epoch=epoch, key_byte=key_byte, report_interval=report_interval, metric=metric,
                                   analysis_id=self.analysis_id)
        self.db.insert(new_insert)

    def save_hyper_parameters_search(self, search_type, hyper_parameters, best_hyper_parameters):
        new_insert = HyperParameterSearch(search_type=search_type, hyper_parameters_settings=hyper_parameters,
                                          best_hyper_parameters=best_hyper_parameters, analysis_id=self.analysis_id)
        return self.db.insert(new_insert)

    def save_confusion_matrix(self, y_pred, y_true, key_byte):
        new_insert = ConfusionMatrix(y_pred=y_pred, y_true=y_true, key_byte=key_byte, analysis_id=self.analysis_id)
        self.db.insert(new_insert)

    def save_probability_rank(self, ranks, classes, correct_key_byte, key_guess, title, key_byte):
        new_insert = ProbabilityRank(ranks=ranks, classes=classes, correct_key_byte=correct_key_byte, key_guess=key_guess,
                                     title=title, key_byte=key_byte, analysis_id=self.analysis_id)
        self.db.insert(new_insert)

    def custom_insert(self, new_insert):
        self.db.insert(new_insert)
