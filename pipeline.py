class Pipeline(object):

    def __init__(self):
        self.functions = []

    def add(self, function, arguments=()):
        self.functions[function] = arguments

    def run(self, data):
        for f, args in self.functions.items():
            data = f(data, *args)

        return data
