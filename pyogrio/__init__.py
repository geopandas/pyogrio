from pyogrio._io import ogr_read, ogr_list_layers
from pyogrio.pandas import read_dataframe


def read(path, layer=None, columns=None):

    return ogr_read(str(path), layer, columns)


def write(path, meta, data, driver=None):
    raise NotImplementedError("Not built!")


def list_layers(path):
    return ogr_list_layers(str(path))
