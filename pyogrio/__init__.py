"""Vectorized vector I/O using OGR."""

try:
    # we try importing shapely, to ensure it is imported (and it can load its
    # own GEOS copy) before we load GDAL and its linked GEOS
    import shapely
    import shapely.geos  # noqa: F401
except Exception:
    pass

from pyogrio._version import get_versions
from pyogrio.core import (
    __gdal_geos_version__,
    __gdal_version__,
    __gdal_version_string__,
    detect_write_driver,
    get_gdal_config_option,
    get_gdal_data_path,
    list_drivers,
    list_layers,
    read_bounds,
    read_info,
    set_gdal_config_options,
    vsi_listtree,
    vsi_rmtree,
    vsi_unlink,
)
from pyogrio.geopandas import read_dataframe, write_dataframe
from pyogrio.raw import open_arrow, read_arrow, write_arrow

__version__ = get_versions()["version"]
del get_versions

__all__ = [
    "list_drivers",
    "detect_write_driver",
    "list_layers",
    "read_bounds",
    "read_info",
    "set_gdal_config_options",
    "get_gdal_config_option",
    "get_gdal_data_path",
    "open_arrow",
    "read_arrow",
    "read_dataframe",
    "vsi_listtree",
    "vsi_rmtree",
    "vsi_unlink",
    "write_arrow",
    "write_dataframe",
    "__gdal_version__",
    "__gdal_version_string__",
    "__gdal_geos_version__",
    "__version__",
]
