from json import JSONEncoder
from numpy import integer, floating, ndarray

class NumpyEncoder(JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, integer):
            return int(obj)
        elif isinstance(obj, floating):
            return float(obj)
        elif isinstance(obj, ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)