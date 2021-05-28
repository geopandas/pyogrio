from pyogrio.core import (
    list_drivers,
    list_layers,
    read_bounds,
    read_info,
    set_gdal_config_options,
    get_gdal_config_option,
    __gdal_version__,
    __gdal_version_string__,
)
from pyogrio.geopandas import read_dataframe, write_dataframe
from pyogrio._version import get_versions


__version__ = get_versions()["version"]
del get_versions
