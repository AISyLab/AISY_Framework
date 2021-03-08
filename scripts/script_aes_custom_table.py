from custom.custom_models.neural_networks import *
from aisy.sca_deep_learning_aes import AisyAes
from custom.custom_tables.tables import *
from app import databases_root_folder

aisy = AisyAes()
aisy.set_dataset("ascad-variable.h5")
aisy.set_database_name("database_ascad.sqlite")
aisy.set_aes_leakage_model(leakage_model="HW", byte=2)
aisy.set_number_of_profiling_traces(10000)
aisy.set_number_of_attack_traces(1000)
aisy.set_batch_size(400)
aisy.set_epochs(20)
aisy.set_neural_network(mlp)

aisy.run()

start_custom_tables(databases_root_folder + "database_ascad.sqlite")
session = start_custom_tables_session(databases_root_folder + "database_ascad.sqlite")

db_inserts = aisy.get_db_inserts()
new_insert = CustomTable(value1=10, value2=20, value3=30, analysis_id=aisy.get_analysis_id())
session.add(new_insert)
session.commit()
