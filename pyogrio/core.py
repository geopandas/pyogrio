from pyogrio._env import GDALEnv

with GDALEnv():
    from pyogrio._ogr import (
        get_gdal_version,
        get_gdal_version_string,
        ogr_list_drivers,
        set_gdal_config_options as _set_gdal_config_options,
        get_gdal_config_option as _get_gdal_config_option,
    )
    from pyogrio._io import ogr_list_layers, ogr_read_info

    __gdal_version__ = get_gdal_version()
    __gdal_version_string__ = get_gdal_version_string()


def list_drivers(read=False, write=False):
    """List drivers available in GDAL.

    Parameters
    ----------
    read: bool, optional (default: False)
        If True, will only return drivers that are known to support read capabilities.
    write: bool, optional (default: False)
        If True, will only return drivers that are known to support write capabilities.

    Returns
    -------
    dict
        Mapping of driver name to file mode capabilities: "r": read, "w": write.
        Drivers that are available but with unknown support are marked with "?"
    """

    drivers = ogr_list_drivers()

    if read:
        drivers = {k: v for k, v in drivers.items() if v.startswith("r")}

    if write:
        drivers = {k: v for k, v in drivers.items() if v.endswith("w")}

    return drivers


def list_layers(path):
    """List layers available in an OGR data source.

    NOTE: includes both spatial and nonspatial layers.

    Parameters
    ----------
    path : str or pathlib.Path

    Returns
    -------
    ndarray shape (2, n)
        array of pairs of [<layer name>, <layer geometry type>]
        Note: geometry is `None` for nonspatial layers.
    """

    return ogr_list_layers(str(path))


def read_info(path, layer=None, encoding=None):
    """Read information about an OGR data source.

    `crs` and `geometry` will be `None` and `features` will be 0 for a
    nonspatial layer.

    Parameters
    ----------
    path : str or pathlib.Path
    layer : [type], optional
        Name or index of layer in data source.  Reads the first layer by default.
    encoding : [type], optional (default: None)
        If present, will be used as the encoding for reading string values from
        the data source, unless encoding can be inferred directly from the data
        source.

    Returns
    -------
    dict
        {
            "crs": "<crs>",
            "fields": <ndarray of field names>,
            "encoding": "<encoding>",
            "geometry": "<geometry type>",
            "features": <feature count>
        }
    """
    return ogr_read_info(str(path), layer=layer, encoding=encoding)


def set_gdal_config_options(options):
    """Set GDAL configuration options.

    Options are listed here: https://trac.osgeo.org/gdal/wiki/ConfigOptions

    No error is raised if invalid option names are provided.

    These options are applied for an entire session rather than for individual
    functions.

    Parameters
    ----------
    options : dict
        If present, provides a mapping of option name / value pairs for GDAL
        configuration options.  True / False are normalized to 'ON' / 'OFF'.
        A value of None for a config option can be used to clear out a previously
        set value.
    """

    _set_gdal_config_options(options)


def get_gdal_config_option(name):
    """Get the value for a GDAL configuration option.

    Parameters
    ----------
    name : str
        name of the option to retrive

    Returns
    -------
    value of the option or None if not set
        'ON' / 'OFF' are normalized to True / False.
    """

    return _get_gdal_config_option(name)
