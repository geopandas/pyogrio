import os


print("Incoming PATH: ", os.environ['PATH'])

from pyogrio.core import list_layers, read_info
from pyogrio.geopandas import read_dataframe, write_dataframe

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
