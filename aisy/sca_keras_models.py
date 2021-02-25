import os


class ScaKerasModels:

    def __init__(self):
        pass

    def keras_model_as_string(self, keras_method_name):

        dir_name = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(dir_name, 'custom\\custom_models')
        filename = file_path + "\\neural_networks.py"

        file_contents = ""

        file = open(filename, 'r')
        lines = file.readlines()
        print_line = False
        for line in lines:

            if keras_method_name + "(" in line:
                print_line = True

            if print_line:
                file_contents += line

                if "return" in line:
                    print_line = False

        if file_contents == "":
            file_contents = "# only neural network models defined in /neural_networks/neural_networks.py can be displayed here."
        return file_contents
