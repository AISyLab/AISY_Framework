import aisy_sca
from app import *
from custom.custom_models.neural_networks import *
from custom.custom_tables.tables import *

aisy = aisy_sca.Aisy()
aisy.set_resources_root_folder(resources_root_folder)
aisy.set_database_root_folder(databases_root_folder)
aisy.set_datasets_root_folder(datasets_root_folder)
aisy.set_database_name("database_ascad.sqlite")
aisy.set_dataset(datasets_dict["ascad-variable.h5"])
aisy.set_aes_leakage_model(leakage_model="HW", byte=2)
aisy.set_batch_size(400)
aisy.set_epochs(20)
aisy.set_neural_network(mlp)

aisy.run()

start_custom_tables(databases_root_folder + "database_ascad.sqlite")
session = start_custom_tables_session(databases_root_folder + "database_ascad.sqlite")

new_insert = CustomTable(value1=10, value2=20, value3=30, analysis_id=aisy.settings["analysis_id"])
session.add(new_insert)
session.commit()
