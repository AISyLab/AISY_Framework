from sqlalchemy import create_engine
from sqlalchemy import MetaData
from sqlalchemy.orm import sessionmaker
import json


class ScaDatabase:

    def __init__(self, database_name):
        self.engine = create_engine('sqlite:///' + str(database_name), echo=False)
        self.metadata = MetaData(self.engine)
        self.session = sessionmaker(bind=self.engine)()

    def insert(self, row):
        self.session.add(row)
        self.session.commit()
        return row.id

    def select_all(self, table_class):
        return self.session.query(table_class).all()

    def select_from_analysis(self, table_class, analysis_id):
        rows = self.session.query(table_class).filter_by(analysis_id=analysis_id).all()
        if rows:
            return rows[0]

    def select_all_from_analysis(self, table_class, analysis_id):
        return self.session.query(table_class).filter_by(analysis_id=analysis_id).all()

    def select_analysis(self, table_class, analysis_id):
        return self.session.query(table_class).filter_by(id=analysis_id).all()[0]

    def select_metrics(self, table_class, analysis_id):
        metric_names = self.session.query(table_class.metric).filter_by(analysis_id=analysis_id).distinct().all()
        return [value for value, in metric_names]

    def select_key_rank_metrics(self, table_class, analysis_id):
        metric_rows = self.session.query(table_class.metric).filter_by(analysis_id=analysis_id).distinct().all()
        metrics = [value for value, in metric_rows]
        return metrics

    def select_key_rank_from_analysis_json(self, table_class, metric, analysis_id):
        rows = self.session.query(table_class).filter_by(analysis_id=analysis_id, metric=metric).all()
        values = {}
        for row in rows:
            values_as_array = json.loads(row.values)
            values_list = []
            for index, value in values_as_array.items():
                values_list.append(values_as_array[str(index)])

            values["values"] = values_list
            values["metric"] = row.metric
            values["report_interval"] = row.report_interval

        return values

    def select_success_rate_from_analysis_json(self, table_class, metric, analysis_id):
        rows = self.session.query(table_class).filter_by(analysis_id=analysis_id, metric=metric).all()
        values = {}
        for row in rows:
            values_as_array = json.loads(row.values)
            values_list = []
            for index, value in values_as_array.items():
                values_list.append(values_as_array[str(index)])

            values["values"] = values_list
            values["metric"] = row.metric
            values["report_interval"] = row.report_interval

        return values

    def select_values_from_analysis_json(self, table_class, analysis_id):
        rows = self.session.query(table_class).filter_by(analysis_id=analysis_id).all()

        return_struct = []

        for row in rows:
            values_as_array = json.loads(row.values)
            values_list = []
            for index, value in values_as_array.items():
                values_list.append(values_as_array[str(index)])

            return_struct.append({
                "key_byte": row.key_byte,
                "values": values_list,
                "metric": row.metric,
                "report_interval": row.report_interval
            })
        return return_struct

    # def select_values_from_analysis_json(self, table_class, analysis_id):
    #     key_byte_rows = self.session.query(table_class.key_byte).filter_by(analysis_id=analysis_id).distinct().all()
    #     key_bytes = [value for value, in key_byte_rows]
    #     values_all_key_bytes = []
    #     for key_byte in key_bytes:
    #         rows = self.session.query(table_class).filter_by(analysis_id=analysis_id, key_byte=key_byte).all()
    #
    #         values_key_byte = []
    #
    #         for row in rows:
    #             values_as_array = json.loads(row.values)
    #             values_list = []
    #             for index, value in values_as_array.items():
    #                 values_list.append(values_as_array[str(index)])
    #
    #             values_key_byte.append({
    #                 "key_byte": key_byte,
    #                 "values": values_list,
    #                 "metric": row.metric,
    #                 "report_interval": row.report_interval
    #             })
    #         values_all_key_bytes.append(values_key_byte)
    #     return values_all_key_bytes

    def select_values_from_confusion_matrix_json(self, table_class, analysis_id):
        values_all = []
        rows = self.session.query(table_class).filter_by(analysis_id=analysis_id).all()
        for row in rows:
            values_as_array = json.loads(row.values)
            values_list = []
            for index, value in values_as_array.items():
                values_list.append(values_as_array[str(index)])

            values_all.append({
                "values": values_list
            })

        return values_all

    def select_values_from_metric(self, table_class, metric, analysis_id):
        key_byte_rows = self.session.query(table_class.key_byte).filter_by(analysis_id=analysis_id, metric=metric).distinct().all()
        key_bytes = [value for value, in key_byte_rows]
        values_all_key_bytes = []

        for key_byte in key_bytes:
            rows = self.session.query(table_class.value).filter_by(analysis_id=analysis_id, key_byte=key_byte, metric=metric).all()
            values = [value for value, in rows]
            values_all_key_bytes.append({
                "key_byte": key_byte,
                "values": values,
                "metric": metric
            })
        return values_all_key_bytes

    def select_final_key_rank_json_from_analysis(self, table_class, analysis_id):
        key_byte_rows = self.session.query(table_class.key_byte).filter_by(analysis_id=analysis_id).distinct().all()
        key_bytes = [value for value, in key_byte_rows]
        values_all_key_bytes = []
        for key_byte in key_bytes:
            rows = self.session.query(table_class).filter_by(analysis_id=analysis_id, key_byte=key_byte).all()
            values_key_byte = []
            for row in rows:
                values_as_array = json.loads(row.values)
                values_list = []
                for index, value in values_as_array.items():
                    values_list.append(values_as_array[str(index)])

                values_key_byte.append({
                    "key_byte": key_byte,
                    "metric": row.metric,
                    "key_rank": int(values_list[len(values_list) - 1])
                })
            values_all_key_bytes.append(values_key_byte)
        return values_all_key_bytes

    def select_final_success_rate_from_analysis(self, table_class, analysis_id):
        key_byte_rows = self.session.query(table_class.key_byte).filter_by(analysis_id=analysis_id).distinct().all()
        key_bytes = [value for value, in key_byte_rows]
        values_all_key_bytes = []
        for key_byte in key_bytes:
            rows = self.session.query(table_class).filter_by(analysis_id=analysis_id, key_byte=key_byte).all()
            values_key_byte = []
            for row in rows:
                values_as_array = json.loads(row.values)
                values_list = []
                for index, value in values_as_array.items():
                    values_list.append(values_as_array[str(index)])

                values_key_byte.append({
                    "key_byte": key_byte,
                    "success_rate": round(values_list[len(values_list) - 1], 2)
                })
            values_all_key_bytes.append(values_key_byte)
        return values_all_key_bytes

    def select_guessing_entropy_json(self, table_class, analysis_id):
        key_byte_rows = self.session.query(table_class.key_byte).filter_by(analysis_id=analysis_id).distinct().all()
        key_bytes = [value for value, in key_byte_rows]
        values_all_key_bytes = []
        for key_byte in key_bytes:
            rows = self.session.query(table_class).filter_by(analysis_id=analysis_id, key_byte=key_byte).all()
            values_key_byte = []
            for row in rows:
                values_as_array = json.loads(row.values)
                values_list = []
                for index, value in values_as_array.items():
                    values_list.append(values_as_array[str(index)])

                values_key_byte.append({
                    "key_byte": key_byte,
                    "values": values_list,
                    "metric": row.metric,
                    "key_guess": row.key_guess
                })
            values_all_key_bytes.append(values_key_byte)
        return values_all_key_bytes

    def select_max_pkc_accuracy_from_analysis(self, table_class, metric, analysis_id):
        rows = self.session.query(table_class).filter_by(analysis_id=analysis_id, name=metric).all()
        values_accuracy = None
        for row in rows:
            values_as_array = json.loads(row.values)
            values_list = []
            for index, value in values_as_array.items():
                values_list.append(values_as_array[str(index)])

            values_accuracy = {
                "metric": metric,
                "max": max(values_list)
            }
        return values_accuracy

    def soft_delete_analysis_from_table(self, table_class, analysis_id):
        self.session.query(table_class).filter_by(id=analysis_id).update({"deleted": True})
        self.session.commit()
        return
