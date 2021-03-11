from pyogrio._env import GDALEnv
from pyogrio._version import get_versions

with GDALEnv():
    from pyogrio.core import list_layers, read_info
    from pyogrio.geopandas import read_dataframe, write_dataframe


__version__ = get_versions()['version']
del get_versions
