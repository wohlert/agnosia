from types import FunctionType

class Pipeline(object):
    """
    Extremely simplified pipeline class for
    uniform preprocessing of multiple inputs
    such as training and test data.
    """

    def __init__(self):
        self.functions = {}

    def add(self, function: FunctionType, arguments: list = []):
        """
        Adds a function to the pipeline.
        The function's auxillary arguments
        must be passed as a list of values.
        """
        self.functions[function] = tuple(arguments)

    def run(self, data):
        """
        Apply all functions in order to the input data.
        """
        for f, args in self.functions.items():
            data = f(data, *args)

        return data
