"""
pipeline

Provides the Pipeline class for chaining actions.
"""

from types import FunctionType
from collections import OrderedDict
import json
import numpy as np


class Pipeline(object):
    """
    Extremely simplified pipeline class for
    uniform preprocessing of multiple inputs
    such as training and test data.
    """

    def __init__(self, functions: dict=None):
        if functions is None:
            functions = {}

        self.functions = OrderedDict(functions)

    def add(self, function: FunctionType, arguments: list=None):
        """
        Adds a function to the pipeline.
        The function's auxillary arguments
        must be passed as a list of values.
        """
        if arguments is None:
            arguments = []

        self.functions[function] = tuple(arguments)

    def run(self, data):
        """
        Apply all functions in order to the input data.
        """
        for function, args in self.functions.items():
            data = function(data, *args)

        return data

    def serialise(self):
        """
        Serialise the pipeline in order to save the contained
        functions.
        """
        function_names = OrderedDict({})
        for function, args in self.functions.items():
            function_names[function.__name__] = args

        json_data = json.dumps(function_names)
        return json_data

    def __str__(self):
        return self.serialise()

    def __repr__(self):
        return self.serialise()

