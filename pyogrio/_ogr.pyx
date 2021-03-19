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

