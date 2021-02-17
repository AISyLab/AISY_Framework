from commons.sca_tables import *


class API:

    def __init__(self, db, analysis_id):
        self.db = db
        self.analysis_id = analysis_id

    def get_all_key_ranks(self):
        return self.db.select_values_from_analysis_json(KeyRank, self.analysis_id)

    def get_all_success_rates(self):
        return self.db.select_values_from_analysis_json(SuccessRate, self.analysis_id)

    def get_metric_names(self):
        return self.db.select_metrics(Metric, self.analysis_id)
