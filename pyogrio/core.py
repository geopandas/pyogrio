from pyogrio._env import GDALEnv
from pyogrio.util import get_vsi_path, _preprocess_options_key_value, _mask_to_wkb


with GDALEnv():
    from pyogrio._ogr import (
        get_gdal_version,
        get_gdal_version_string,
        get_gdal_geos_version,
        ogr_list_drivers,
        set_gdal_config_options as _set_gdal_config_options,
        get_gdal_config_option as _get_gdal_config_option,
        get_gdal_data_path as _get_gdal_data_path,
        init_gdal_data as _init_gdal_data,
        init_proj_data as _init_proj_data,
        remove_virtual_file,
        _register_drivers,
        _get_drivers_for_path,
    )
    from pyogrio._err import _register_error_handler
    from pyogrio._io import ogr_list_layers, ogr_read_bounds, ogr_read_info

    _init_gdal_data()
    _init_proj_data()
    _register_drivers()
    _register_error_handler()

    __gdal_version__ = get_gdal_version()
    __gdal_version_string__ = get_gdal_version_string()
    __gdal_geos_version__ = get_gdal_geos_version()


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
        Mapping of driver name to file mode capabilities: ``"r"``: read, ``"w"``: write.
        Drivers that are available but with unknown support are marked with ``"?"``
    """

    drivers = ogr_list_drivers()

    if read:
        drivers = {k: v for k, v in drivers.items() if v.startswith("r")}

    if write:
        drivers = {k: v for k, v in drivers.items() if v.endswith("w")}

    return drivers


def detect_write_driver(path):
    """Attempt to infer the driver for a path by extension or prefix.  Only
    drivers that support write capabilities will be detected.

    If the path cannot be resolved to a single driver, a ValueError will be
    raised.

    Parameters
    ----------
    path : str

    Returns
    -------
    str
        name of the driver, if detected
    """
    # try to infer driver from path
    drivers = _get_drivers_for_path(path)

    if len(drivers) == 0:
        raise ValueError(
            f"Could not infer driver from path: {path}; please specify driver "
            "explicitly"
        )

    # if there are multiple drivers detected, user needs to specify the correct
    # one manually
    elif len(drivers) > 1:
        raise ValueError(
            f"Could not infer driver from path: {path}; multiple drivers are "
            f"available for that extension: {', '.join(drivers)}.  Please "
            "specify driver explicitly."
        )

    return drivers[0]


def list_layers(path_or_buffer, /):
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
    path, buffer = get_vsi_path(path_or_buffer)

    try:
        result = ogr_list_layers(path)
    finally:
        if buffer is not None:
            remove_virtual_file(path)
    return result


def read_bounds(
    path_or_buffer,
    /,
    layer=None,
    skip_features=0,
    max_features=None,
    where=None,
    bbox=None,
    mask=None,
):
    """Read bounds of each feature.

    This can be used to assist with spatial indexing and partitioning, in
    order to avoid reading all features into memory.  It is roughly 2-3x faster
    than reading the full geometry and attributes of a dataset.

    Parameters
    ----------
    path : pathlib.Path or str
        data source path
    layer : int or str, optional (default: first layer)
        If an integer is provided, it corresponds to the index of the layer
        with the data source.  If a string is provided, it must match the name
        of the layer in the data source.  Defaults to first layer in data source.
    skip_features : int, optional (default: 0)
        Number of features to skip from the beginning of the file before returning
        features.  Must be less than the total number of features in the file.
    max_features : int, optional (default: None)
        Number of features to read from the file.  Must be less than the total
        number of features in the file minus ``skip_features`` (if used).
    where : str, optional (default: None)
        Where clause to filter features in layer by attribute values.  Uses a
        restricted form of SQL WHERE clause, defined here:
        http://ogdi.sourceforge.net/prop/6.2.CapabilitiesMetadata.html
        Examples: ``"ISO_A3 = 'CAN'"``, ``"POP_EST > 10000000 AND POP_EST < 100000000"``
    bbox : tuple of (xmin, ymin, xmax, ymax), optional (default: None)
        If present, will be used to filter records whose geometry intersects this
        box.  This must be in the same CRS as the dataset.  If GEOS is present
        and used by GDAL, only geometries that intersect this bbox will be
        returned; if GEOS is not available or not used by GDAL, all geometries
        with bounding boxes that intersect this bbox will be returned.
    mask : Shapely geometry, optional (default: None)
        If present, will be used to filter records whose geometry intersects
        this geometry.  This must be in the same CRS as the dataset.  If GEOS is
        present and used by GDAL, only geometries that intersect this geometry
        will be returned; if GEOS is not available or not used by GDAL, all
        geometries with bounding boxes that intersect the bounding box of this
        geometry will be returned.  Requires Shapely >= 2.0.
        Cannot be combined with ``bbox`` keyword.

    Returns
    -------
    tuple of (fids, bounds)
        fids are global IDs read from the FID field of the dataset
        bounds are ndarray of shape(4, n) containing ``xmin``, ``ymin``, ``xmax``,
        ``ymax``
    """
    path, buffer = get_vsi_path(path_or_buffer)

    try:
        result = ogr_read_bounds(
            path,
            layer=layer,
            skip_features=skip_features,
            max_features=max_features or 0,
            where=where,
            bbox=bbox,
            mask=_mask_to_wkb(mask),
        )
    finally:
        if buffer is not None:
            remove_virtual_file(path)
    return result


def read_info(
    path_or_buffer,
    /,
    layer=None,
    encoding=None,
    force_feature_count=False,
    force_total_bounds=False,
    **kwargs,
):
    """Read information about an OGR data source.

    ``crs``, ``geometry`` and ``total_bounds`` will be ``None`` and ``features`` will be
    0 for a nonspatial layer.

    ``features`` will be -1 if this is an expensive operation for this driver. You can
    force it to be calculated using the ``force_feature_count`` parameter.

    ``total_bounds`` is the 2-dimensional extent of all features within the dataset:
    (xmin, ymin, xmax, ymax). It will be None if this is an expensive operation for this
    driver or if the data source is nonspatial. You can force it to be calculated using
    the ``force_total_bounds`` parameter.

    Parameters
    ----------
    path : str or pathlib.Path
    layer : [type], optional
        Name or index of layer in data source.  Reads the first layer by default.
    encoding : [type], optional (default: None)
        If present, will be used as the encoding for reading string values from
        the data source, unless encoding can be inferred directly from the data
        source.
    force_feature_count : bool, optional (default: False)
        True if the feature count should be computed even if it is expensive.
    force_total_bounds : bool, optional (default: False)
        True if the total bounds should be computed even if it is expensive.
    **kwargs
        Additional driver-specific dataset open options passed to OGR.  Invalid
        options will trigger a warning.

    Returns
    -------
    dict
        A dictionary with the following keys::

            {
                "crs": "<crs>",
                "fields": <ndarray of field names>,
                "dtypes": <ndarray of field dtypes>,
                "encoding": "<encoding>",
                "geometry": "<geometry type>",
                "features": <feature count or -1>,
                "total_bounds": <tuple with total bounds or None>,
                "driver": "<driver>",
                "capabilities": "<dict of driver capabilities>"
                "dataset_metadata": "<dict of dataset metadata or None>"
                "layer_metadata": "<dict of layer metadata or None>"
            }
    """
    path, buffer = get_vsi_path(path_or_buffer)

    dataset_kwargs = _preprocess_options_key_value(kwargs) if kwargs else {}

    try:
        result = ogr_read_info(
            path,
            layer=layer,
            encoding=encoding,
            force_feature_count=force_feature_count,
            force_total_bounds=force_total_bounds,
            dataset_kwargs=dataset_kwargs,
        )
    finally:
        if buffer is not None:
            remove_virtual_file(path)
    return result


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
        configuration options.  ``True`` / ``False`` are normalized to ``'ON'``
        / ``'OFF'``. A value of ``None`` for a config option can be used to clear out a
        previously set value.
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
        ``'ON'`` / ``'OFF'`` are normalized to ``True`` / ``False``.
    """

    return _get_gdal_config_option(name)


def get_gdal_data_path():
    """Get the path to the directory GDAL uses to read data files.

    Returns
    -------
    str, or None if data directory was not found
    """
    return _get_gdal_data_path()
