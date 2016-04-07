"""
pipeline

Provides the Pipeline class for chaining actions.
"""

from types import FunctionType
from collections import OrderedDict


class Pipeline(object):
    """
    Extremely simplified pipeline class for
    uniform preprocessing of multiple inputs
    such as training and test data.
    """

    def __init__(self, functions={}):
        self.functions = OrderedDict(functions)

    def add(self, function: FunctionType, arguments: list=[]):
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
        for function, args in self.functions.items():
            data = function(data, *args)

        return data
