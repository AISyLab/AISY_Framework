import sys, os


class ScaKerasModels:

    def __init__(self):
        pass

    def keras_model_as_string(self, keras_method, keras_method_name):

        filename = os.path.abspath(sys.modules[keras_method.__module__].__file__)

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
            file_contents = "# to display the keras model definition, you need to pass the method name to set_neural_network() method (instead of an object)"
        return file_contents
