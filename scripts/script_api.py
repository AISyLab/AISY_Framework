from api.api import API
from commons.sca_database import *

# set analysis id (this can be retrieved from web application)
analysis_id = 1
# start database
db_location = "C:/Users/guilh/PycharmProjects/aisy/databases"
db_name = "database_ascad.sqlite"
db = ScaDatabase("{}/{}".format(db_location, db_name))
# start api
api = API(db, analysis_id)

metrics = api.get_metric_names()
print(metrics)

# key_ranks = api.get_all_key_ranks()
#
# import matplotlib.pyplot as plt
#
# for key_rank in key_ranks:
#     plt.plot(key_rank['values'])
# plt.xlabel("Traces")
# plt.ylabel("Key Rank")
# plt.show()
#
# success_rates = api.get_all_success_rates()
#
# import matplotlib.pyplot as plt
#
# for success_rate in success_rates:
#     plt.plot(success_rate['values'])
# plt.xlabel("Traces")
# plt.ylabel("Success Rate")
# plt.show()
