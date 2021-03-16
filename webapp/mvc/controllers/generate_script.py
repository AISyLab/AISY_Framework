from aisy_database.db_select import DBSelect
from aisy_database.db_tables import *


def generate_script(script_filename, databases_root_folder, table_name, analysis_id, reproducible, from_db):
    db_select = DBSelect(databases_root_folder + table_name)

    analysis = db_select.select_analysis(Analysis, analysis_id)
    neural_network_model = db_select.select_from_analysis(NeuralNetwork, analysis_id)

    # get training hyper-parameters information from database
    hyper_parameters = db_select.select_all_from_analysis(HyperParameter, analysis_id)
    hyper_parameters_single = hyper_parameters[0].hyper_parameters[0]

    hyper_parameter_search = None
    if len(hyper_parameters) > 1:
        hyper_parameter_search = db_select.select_from_analysis(HyperParameterSearch, analysis.id)

    leakage_models = db_select.select_from_analysis(LeakageModel, analysis_id)
    leakage_model_parameters = leakage_models.leakage_model

    script_py_file = open("scripts/{}_{}.py".format(script_filename, table_name.replace(".sqlite", "")), "w+")

    script_py_file.write("import aisy_sca\n")
    script_py_file.write("from app import *\n")
    if "grid_search" not in analysis.settings and "random_search" not in analysis.settings:
        script_py_file.write("from custom.custom_models.neural_networks import *\n")
    if "data_augmentation" in analysis.settings:
        script_py_file.write("from custom.custom_data_augmentation.data_augmentation import *\n")
    if reproducible and from_db:
        script_py_file.write("from aisy_database.db_select import DBSelect\n")
        script_py_file.write("from aisy_database.db_tables import *\n")

    script_py_file.write('\naisy = aisy_sca.Aisy()')
    script_py_file.write('\naisy.set_resources_root_folder(resources_root_folder)')
    script_py_file.write('\naisy.set_database_root_folder(databases_root_folder)')
    script_py_file.write('\naisy.set_datasets_root_folder(datasets_root_folder)')
    script_py_file.write('\naisy.set_dataset(datasets_dict["{}"])'.format(analysis.dataset))
    script_py_file.write('\naisy.set_database_name("{}")'.format(table_name))
    script_py_file.write('\naisy.set_aes_leakage_model(')
    for index, key in enumerate(leakage_model_parameters):
        if isinstance(leakage_model_parameters[key], str):
            script_py_file.write('{}="{}"'.format(key, leakage_model_parameters[key]))
        else:
            script_py_file.write('{}={}'.format(key, leakage_model_parameters[key]))
        if index < len(leakage_model_parameters) - 1:
            script_py_file.write(', ')
    script_py_file.write(')')
    script_py_file.write('\naisy.set_key("{}")'.format(hyper_parameters_single['key']))
    script_py_file.write('\naisy.set_number_of_profiling_traces({})'.format(hyper_parameters_single['profiling_traces']))
    script_py_file.write('\naisy.set_number_of_attack_traces({})'.format(hyper_parameters_single['attack_traces']))
    script_py_file.write('\naisy.set_first_sample({})'.format(hyper_parameters_single['first_sample']))
    script_py_file.write('\naisy.set_number_of_samples({})'.format(hyper_parameters_single['number_of_samples']))
    script_py_file.write('\naisy.set_batch_size({})'.format(hyper_parameters_single['batch_size']))
    script_py_file.write('\naisy.set_epochs({})'.format(hyper_parameters_single['epochs']))

    if from_db:
        script_py_file.write('\ndb = DBSelect("{}")'.format(databases_root_folder + table_name))
        script_py_file.write('\nanalysis = db.select_analysis(Analysis, {})'.format(analysis_id))

    if "grid_search" not in analysis.settings and "random_search" not in analysis.settings:
        if reproducible:
            script_py_file.write(
                '\naisy.set_model_seed(analysis.settings["seed"])' if from_db else '\naisy.set_model_seed({})'.format(
                    analysis.settings["seed"]))
    if len(hyper_parameters) == 1:
        script_py_file.write('\n\n\n{}'.format(neural_network_model.description))
        script_py_file.write('\n\naisy.set_neural_network({})'.format(neural_network_model.model_name))
    else:
        if "grid_search" in analysis.settings:
            script_py_file.write('\ngrid_search = {}'.format(analysis.settings["grid_search"]))
        if "random_search" in analysis.settings:
            script_py_file.write('\nrandom_search = {}'.format(analysis.settings["random_search"]))
    if "early_stopping" in analysis.settings:
        script_py_file.write('\nearly_stopping = {}'.format(analysis.settings["early_stopping"]))
    if "callbacks" in analysis.settings:
        script_py_file.write('\ncustom_callbacks = {}'.format(analysis.settings["callbacks"]))

    if reproducible:
        if from_db:
            script_py_file.write('\naisy.set_reproducible_analysis(True)')
            if "grid_search" in analysis.settings or "random_search" in analysis.settings:
                script_py_file.write('\naisy.set_ge_sr_random_states_search(analysis.settings["ge_sr_random_states_search"])')
                script_py_file.write('\naisy.set_hp_combinations_reproducible(analysis.settings["hp_combinations_reproducible"])')
            if "ensemble" in analysis.settings:
                script_py_file.write('\naisy.set_ge_sr_random_states_ensemble(analysis.settings["ge_sr_random_states_ensemble"])')
            if "early_stopping" in analysis.settings:
                script_py_file.write(
                    '\naisy.set_ge_sr_random_states_early_stopping(analysis.settings["ge_sr_random_states_early_stopping"])')
            if "ge_sr_random_states" in analysis.settings:
                script_py_file.write('\naisy.set_ge_sr_random_states(analysis.settings["ge_sr_random_states"])')
        else:
            script_py_file.write('\naisy.set_reproducible_analysis(True)')
            if "grid_search" in analysis.settings or "random_search" in analysis.settings:
                script_py_file.write('\naisy.set_ge_sr_random_states_search({})'.format(analysis.settings["ge_sr_random_states_search"]))
                script_py_file.write(
                    '\naisy.set_hp_combinations_reproducible({})'.format(analysis.settings["hp_combinations_reproducible"]))
            if "ensemble" in analysis.settings:
                script_py_file.write(
                    '\naisy.set_ge_sr_random_states_ensemble({})'.format(analysis.settings["ge_sr_random_states_ensemble"]))
            if "early_stopping" in analysis.settings:
                script_py_file.write(
                    '\naisy.set_ge_sr_random_states_early_stopping({})'.format(analysis.settings["ge_sr_random_states_early_stopping"]))
            if "ge_sr_random_states" in analysis.settings:
                script_py_file.write('\naisy.set_ge_sr_random_states({})'.format(analysis.settings["ge_sr_random_states"]))

    script_py_file.write('\n\naisy.run(')
    script_py_file.write('\n    key_rank_executions={},'.format(analysis.settings["key_rank_executions"]))
    script_py_file.write('\n    key_rank_report_interval={},'.format(analysis.settings["key_rank_report_interval"]))
    script_py_file.write('\n    key_rank_attack_traces={},'.format(analysis.settings["key_rank_attack_traces"]))
    if "early_stopping" in analysis.settings:
        script_py_file.write('\n    early_stopping=early_stopping,')
    if "ensemble" in analysis.settings:
        script_py_file.write('\n    ensemble=[{}],'.format(analysis.settings["ensemble"]))
    if "visualization" in analysis.settings:
        script_py_file.write('\n    visualization=[{}],'.format(analysis.settings["visualization"]))
    if "save_to_npz" in analysis.settings:
        script_py_file.write('\n    save_to_npz=["{}"]'.format(analysis.settings["save_to_npz"][0]))
    if "callbacks" in analysis.settings:
        script_py_file.write('\n    callbacks=custom_callbacks')
    if "data_augmentation" in analysis.settings:
        script_py_file.write(
            '\n    data_augmentation=[{}, {}]'.format(analysis.settings["data_augmentation"][0], analysis.settings["data_augmentation"][1]))
    if len(hyper_parameters) == 1:
        script_py_file.write('\n)\n')
    else:
        if hyper_parameter_search.search_type == "Grid Search":
            script_py_file.write('\n    grid_search=grid_search')
        else:
            script_py_file.write('\n    random_search=random_search')
        script_py_file.write('\n)\n')
    script_py_file.close()
