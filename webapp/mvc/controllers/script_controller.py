from aisy_database.db_select import DBSelect
from aisy_database.db_tables import *
import json


def write_imports(script, analysis):
    script.write("import aisy_sca\n")
    script.write("from app import *\n")
    if not analysis.settings["hyperparameter_search"]:
        script.write("from tensorflow.keras.optimizers import *\n")
        script.write("from tensorflow.keras.layers import *\n")
        script.write("from tensorflow.keras.models import *\n")
    if "data_augmentation" in analysis.settings:
        script.write("from custom.custom_data_augmentation.data_augmentation import *\n")


def write_database_imports(script):
    script.write("from aisy_database.db_select import DBSelect\n")
    script.write("from aisy_database.db_tables import *\n")


def write_minimum_settings(script, analysis, table_name):
    script.write('\naisy = aisy_sca.Aisy()')
    script.write('\naisy.set_resources_root_folder(resources_root_folder)')
    script.write('\naisy.set_database_root_folder(databases_root_folder)')
    script.write('\naisy.set_datasets_root_folder(datasets_root_folder)')
    script.write(f'\naisy.set_dataset(datasets_dict["{analysis.dataset}"])')
    script.write(f'\naisy.set_database_name("{table_name}")')


def write_leakage_model(script, leakage_model):
    leakage_model_parameters = leakage_model.leakage_model
    script.write('\n\naisy.set_aes_leakage_model(')
    for index, key in enumerate(leakage_model_parameters):
        if isinstance(leakage_model_parameters[key], str):
            script.write(f'{key}="{leakage_model_parameters[key]}"')
        else:
            script.write(f'{key}={leakage_model_parameters[key]}')
        if index < len(leakage_model_parameters) - 1:
            script.write(', ')
    script.write(')')


def write_dataset_definitions(script, analysis):
    key = analysis.settings['key']
    script.write(f"\n\naisy.set_key('{key}')")
    script.write(f"\naisy.set_number_of_profiling_traces({analysis.settings['number_of_profiling_traces']})")
    script.write(f"\naisy.set_number_of_attack_traces({analysis.settings['number_of_attack_traces']})")
    script.write(f"\naisy.set_first_sample({analysis.settings['first_sample']})")
    script.write(f"\naisy.set_number_of_samples({analysis.settings['number_of_samples']})")
    script.write(f"\naisy.set_batch_size({analysis.settings['batch_size']})")
    script.write(f"\naisy.set_epochs({analysis.settings['epochs']})")


def write_read_analysis(script, db_path, analysis_id):
    script.write(f'\n\ndb_select = DBSelect("{db_path}")')
    script.write(f'\nanalysis = db_select.select_analysis(Analysis, {analysis_id})')


def write_neural_networks(script, analysis, neural_network_rows):
    if not analysis.settings["hyperparameter_search"]:
        models = analysis.settings["models"]
        for row in neural_network_rows:
            script.write(f'\n\n\n{row.description}')
        for idx in range(len(neural_network_rows)):
            script.write(f'\n\naisy.add_neural_network({models[f"{idx}"]["method_name"]}, seed={models[f"{idx}"]["seed"]})')


def write_search_definitions(script, analysis):
    if analysis.settings["use_grid_search"]:
        script.write(f'\n\ngrid_search = {analysis.settings["grid_search"]}')
    if analysis.settings["use_random_search"]:
        script.write(f'\n\nrandom_search = {analysis.settings["random_search"]}')


def write_early_stopping_defitions(script, analysis):
    if analysis.settings["use_early_stopping"]:
        script.write(f'\n\nearly_stopping = {analysis.settings["early_stopping"]}')


def write_callbacks_defitions(script, analysis):
    if analysis.settings["use_custom_callbacks"]:
        script.write(f'\n\ncustom_callbacks = {analysis.settings["custom_callbacks"]}')


def write_hyperparameters_combinations_from_db(script, analysis):
    if analysis.settings["hyperparameter_search"]:
        script.write('\n\nhyperparameters = db_select.select_all_from_analysis(HyperParameter, analysis.id)')
        script.write('\nhyperparameters_reproducible = []')
        script.write('\nfor row in hyperparameters:')
        script.write('\n    hyperparameters_reproducible.append(row.hyperparameters)')
        script.write('\naisy.set_hyperparameters_reproducible(hyperparameters_reproducible)')


def write_hyperparameters_combinations(script, hyperparameters_rows, analysis):
    if analysis.settings["hyperparameter_search"]:
        hyperparameters = []
        for row in hyperparameters_rows:
            hyperparameters.append(row.hyperparameters)
        script.write(f'\n\nhyperparameters = {hyperparameters}')
        script.write('\naisy.set_hyperparameters_reproducible(hyperparameters)')


def write_random_states_from_db(script):
    script.write('\n\nrandom_states_rows = db_select.select_all_from_analysis(RandomStatesHyperParameter, analysis.id)')
    script.write('\nrandom_states = {}')
    script.write('\nfor row in random_states_rows:')
    script.write('\n    if f"{row.index}" not in random_states:')
    script.write('\n        random_states[f"{row.index}"] = {}')
    script.write('\n    random_states[f"{row.index}"][f"{row.label}"] = json.loads(row.random_states)')
    script.write('\naisy.set_random_states(random_states)')


def write_random_states(script, random_states_rows):
    random_states = {}
    for row in random_states_rows:
        if f"{row.index}" not in random_states:
            random_states[f"{row.index}"] = {}
        random_states[f"{row.index}"][f"{row.label}"] = json.loads(row.random_states)
    script.write(f'\n\nrandom_states = {random_states}')
    script.write('\naisy.set_random_states(random_states)')


def write_run_method(script, analysis):
    script.write('\n\naisy.run(')
    script.write(f'\n    key_rank_executions={analysis.settings["key_rank_executions"]},')
    script.write(f'\n    key_rank_report_interval={analysis.settings["key_rank_report_interval"]},')
    script.write(f'\n    key_rank_attack_traces={analysis.settings["key_rank_attack_traces"]},')
    if analysis.settings["use_early_stopping"]:
        script.write('\n    early_stopping=early_stopping,')
    if analysis.settings["use_ensemble"]:
        script.write(f'\n    ensemble={analysis.settings["ensemble"]},')
    if analysis.settings["use_visualization"]:
        script.write(f'\n    visualization=[{analysis.settings["visualization"]}],')
    if analysis.settings["use_custom_callbacks"]:
        script.write('\n    callbacks=custom_callbacks')
    if analysis.settings["use_data_augmentation"]:
        script.write(
            f'\n    data_augmentation=[{analysis.settings["data_augmentation"][0]}, {analysis.settings["data_augmentation"][1]}]')
    if analysis.settings["use_grid_search"]:
        script.write('\n    grid_search=grid_search,')
    if analysis.settings["use_random_search"]:
        script.write('\n    random_search=random_search,')
    if analysis.settings["use_profiling_analyzer"]:
        steps = {"steps": analysis.settings["profiling_analyzer_steps"]}
        script.write(f'\n    profiling_analyzer={steps}')
    script.write('\n)\n')


def write_reproducible_script(script_filename, databases_root_folder, table_name, analysis_id):
    """ Read information from database """
    db_select = DBSelect(databases_root_folder + table_name)
    analysis = db_select.select_analysis(Analysis, analysis_id)
    neural_network_rows = db_select.select_all_from_analysis(NeuralNetwork, analysis_id)
    leakage_model = db_select.select_from_analysis(LeakageModel, analysis_id)

    """ Create .py file in the scripts folder """
    script_py_file = open(f"scripts/{script_filename}_{table_name.replace('.sqlite', '')}.py", "w+")

    """ write to file """
    write_imports(script_py_file, analysis)
    write_minimum_settings(script_py_file, analysis, table_name)
    write_leakage_model(script_py_file, leakage_model)
    write_dataset_definitions(script_py_file, analysis)
    write_neural_networks(script_py_file, analysis, neural_network_rows)
    write_search_definitions(script_py_file, analysis)
    write_early_stopping_defitions(script_py_file, analysis)
    write_callbacks_defitions(script_py_file, analysis)
    write_run_method(script_py_file, analysis)
    script_py_file.close()


def write_fully_reproducible_script(script_filename, databases_root_folder, table_name, analysis_id, from_db=False):
    """ Read information from database """
    db_select = DBSelect(databases_root_folder + table_name)
    analysis = db_select.select_analysis(Analysis, analysis_id)
    neural_network_rows = db_select.select_all_from_analysis(NeuralNetwork, analysis_id)
    leakage_model = db_select.select_from_analysis(LeakageModel, analysis_id)
    random_states_rows = db_select.select_all_from_analysis(RandomStatesHyperParameter, analysis_id)
    hyperparameters_rows = db_select.select_all_from_analysis(HyperParameter, analysis_id)

    """ Create .py file in the scripts folder """
    script_py_file = open(f"scripts/{script_filename}_{table_name.replace('.sqlite', '')}.py", "w+")

    """ write to file """
    write_imports(script_py_file, analysis)
    if from_db:
        write_database_imports(script_py_file)
    write_minimum_settings(script_py_file, analysis, table_name)
    write_leakage_model(script_py_file, leakage_model)
    write_dataset_definitions(script_py_file, analysis)
    if from_db:
        write_read_analysis(script_py_file, f"{databases_root_folder}{table_name}", analysis_id)
    write_neural_networks(script_py_file, analysis, neural_network_rows)
    write_search_definitions(script_py_file, analysis)
    write_early_stopping_defitions(script_py_file, analysis)
    write_callbacks_defitions(script_py_file, analysis)
    if from_db:
        write_hyperparameters_combinations_from_db(script_py_file, analysis)
        write_random_states_from_db(script_py_file)
    else:
        write_hyperparameters_combinations(script_py_file, hyperparameters_rows, analysis)
        write_random_states(script_py_file, random_states_rows)

    write_run_method(script_py_file, analysis)
    script_py_file.close()
