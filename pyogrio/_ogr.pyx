import os
import sys
import warnings

from pyogrio._err cimport exc_wrap_int, exc_wrap_ogrerr
from pyogrio._err import CPLE_BaseError


cdef get_string(const char *c_str, str encoding="UTF-8"):
    """Get Python string from a char *

    IMPORTANT: the char * must still be freed by the caller.

    Parameters
    ----------
    c_str : char *
    encoding : str, optional (default: UTF-8)

    Returns
    -------
    Python string
    """
    cdef bytes py_str

    py_str = c_str
    return py_str.decode(encoding)


def get_gdal_version():
    """Convert GDAL version number into tuple of (major, minor, revision)"""
    version = int(GDALVersionInfo("VERSION_NUM"))
    major = version // 1000000
    minor = (version - (major * 1000000)) // 10000
    revision = (version - (major * 1000000) - (minor * 10000)) // 100
    return (major, minor, revision)


def get_gdal_version_string():
    cdef const char* version = GDALVersionInfo("RELEASE_NAME")
    return get_string(version)


def set_gdal_config_options(dict options):
    for name, value in options.items():
        name_b = name.encode('utf-8')
        name_c = name_b

        # None is a special case; this is used to clear the previous value
        if value is None:
            CPLSetConfigOption(<const char*>name_c, NULL)
            continue

        # normalize bool to ON/OFF
        if isinstance(value, bool):
            value_b = b'ON' if value else b'OFF'
        else:
            value_b = str(value).encode('utf-8')

        value_c = value_b
        CPLSetConfigOption(<const char*>name_c, <const char*>value_c)


def get_gdal_config_option(str name):
    name_b = name.encode('utf-8')
    name_c = name_b
    value = CPLGetConfigOption(<const char*>name_c, NULL)

    if not value:
        return None

    if value.isdigit():
        return int(value)

    if value == b'ON':
        return True
    if value == b'OFF':
        return False

    str_value = get_string(value)

    return str_value


### Drivers
# mapping of driver:mode
# see full list at https://gdal.org/drivers/vector/index.html
# only for drivers specifically known to operate correctly with pyogrio
DRIVERS = {
    # "CSV": "rw",  # TODO: needs geometry conversion method
    "ESRI Shapefile": "rw",
    "FlatGeobuf": "rw",
    "GeoJSON": "rw",
    "GeoJSONSeq": "rw",
    "GML": "rw",
    # "GPX": "rw", # TODO: supports limited geometry types
    "GPKG": "rw",
    "OAPIF": "r",
    "OpenFileGDB": "r",
    "TopoJSON": "r",
    # "XLSX": "rw",  # TODO: needs geometry conversion method
}


def ogr_list_drivers():
    cdef OGRSFDriverH driver = NULL
    cdef int i
    cdef char *name_c

    # Register all drivers
    GDALAllRegister()

    drivers = dict()
    for i in range(OGRGetDriverCount()):
        driver = OGRGetDriver(i)
        name_c = <char *>OGR_Dr_GetName(driver)

        name = get_string(name_c)
        # drivers that are not specifically listed have unknown support
        # this omits any drivers from supported list that are not installed
        drivers[name] = DRIVERS.get(name, '?')

    return drivers


cdef void set_proj_search_path(str path):
    """Set PROJ library data file search path for use in GDAL."""
    cdef char **paths = NULL
    cdef const char *path_c = NULL
    path_b = path.encode("utf-8")
    path_c = path_b
    paths = CSLAddString(paths, path_c)
    OSRSetPROJSearchPaths(<const char *const *>paths)


cdef char has_proj_data():
    """Verify if PROJ library data files are loaded by GDAL.

    Returns
    -------
    bool
        True if a test spatial reference object could be created, which verifies
        that data files are correctly loaded.

    Adapted from Fiona (_env.pyx).
    """
    cdef OGRSpatialReferenceH srs = OSRNewSpatialReference(NULL)

    try:
        exc_wrap_ogrerr(exc_wrap_int(OSRImportFromEPSG(srs, 4326)))
    except CPLE_BaseError:
        return 0
    else:
        return 1
    finally:
        if srs != NULL:
            OSRRelease(srs)


def init_proj_data():
    """Set Proj search directories in the following precedence:
    - PROJ_LIB env var
    - wheel copy of proj
    - default install of proj found by GDAL
    - search other well-known paths

    Adapted from Fiona (env.py, _env.pyx).
    """

    if "PROJ_LIB" in os.environ:
        set_proj_search_path(os.environ["PROJ_LIB"])
        # verify that this now resolves
        if not has_proj_data():
            raise ValueError("PROJ_LIB did not resolve to a path containing PROJ data files")
        return

    # wheels are packaged to include PROJ data files at pyogrio/proj_data
    wheel_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "proj_data"))
    if os.path.exists(wheel_dir):
        set_proj_search_path(wheel_dir)
        # verify that this now resolves
        if not has_proj_data():
            raise ValueError("Could not correctly detect PROJ data files installed by pyogrio wheel")
        return

    # GDAL correctly found PROJ based on compiled-in paths
    if has_proj_data():
        return

    wk_path = os.path.join(sys.prefix, 'share', 'proj')
    if os.path.exists(wk_path):
        set_proj_search_path(wk_path)
        # verify that this now resolves
        if not has_proj_data():
            raise ValueError(f"Found PROJ data directory at {wk_path} but it does not appear to correctly contain PROJ data files")
        return

    warnings.warn("Could not detect PROJ data files.  Set PROJ_LIB environment variable to the correct path.", RuntimeWarning)
