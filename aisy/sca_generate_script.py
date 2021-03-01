from aisy.sca_database import ScaDatabase
from aisy.sca_tables import *


def generate_script(script_filename, databases_root_folder, table_name, analysis_id):
    script_py_file = open("scripts/{}_{}.py".format(script_filename, table_name.replace(".sqlite", "")), "w+")

    script_py_file.write("from tensorflow.keras.optimizers import *\n")
    script_py_file.write("from tensorflow.keras.layers import *\n")
    script_py_file.write("from tensorflow.keras.models import *\n")
    script_py_file.write("from aisy.sca_deep_learning_aes import AisyAes\n")

    db = ScaDatabase(databases_root_folder + table_name)

    analysis = db.select_analysis(Analysis, analysis_id)
    neural_network_model = db.select_from_analysis(NeuralNetwork, analysis_id)

    # get training hyper-parameters information from database
    hyper_parameters = db.select_all_from_analysis(HyperParameter, analysis_id)
    hyper_parameters_single = hyper_parameters[0].hyper_parameters[0]

    hyper_parameter_search = None
    if len(hyper_parameters) > 1:
        hyper_parameter_search = db.select_from_analysis(HyperParameterSearch, analysis.id)

    leakage_models = db.select_from_analysis(LeakageModel, analysis_id)
    leakage_model_parameters = leakage_models.leakage_model[0]

    script_py_file.write('\naisy = AisyAes()')
    script_py_file.write('\naisy.set_dataset("{}")'.format(analysis.dataset))
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

    script_py_file.write('\n\naisy.run(')
    script_py_file.write('\n    key_rank_executions={},'.format(analysis.settings["key_rank_executions"]))
    script_py_file.write('\n    key_rank_report_interval={},'.format(analysis.settings["key_rank_report_interval"]))
    script_py_file.write('\n    key_rank_attack_traces={},'.format(analysis.settings["key_rank_attack_traces"]))
    if "early_stopping" in analysis.settings:
        script_py_file.write('\n    early_stopping=early_stopping,')
    if "ensemble" in analysis.settings:
        script_py_file.write('\n    ensemble=[{}],'.format(analysis.settings["ensemble"]))
    if len(hyper_parameters) == 1:
        script_py_file.write('\n)\n')
    else:
        if hyper_parameter_search.search_type == "Grid Search":
            script_py_file.write('\n    grid_search=grid_search')
        else:
            script_py_file.write('\n    random_search=random_search')
        script_py_file.write('\n)\n')
    script_py_file.close()
