import warnings
import os

from pyogrio._env import GDALEnv
from pyogrio.errors import DataSourceError
from pyogrio.util import get_vsi_path

with GDALEnv():
    from pyogrio._io import ogr_read, ogr_read_arrow, ogr_write
    from pyogrio._ogr import (
        get_gdal_version,
        get_gdal_version_string,
        ogr_driver_supports_write,
        remove_virtual_file,
        _get_driver_metadata_item,
    )


DRIVERS = {
    ".fgb": "FlatGeobuf",
    ".geojson": "GeoJSON",
    ".geojsonl": "GeoJSONSeq",
    ".geojsons": "GeoJSONSeq",
    ".gpkg": "GPKG",
    ".json": "GeoJSON",
    ".shp": "ESRI Shapefile",
}


DRIVERS_NO_MIXED_SINGLE_MULTI = {
    "FlatGeobuf",
    "GPKG",
}


def read(
    path_or_buffer,
    /,
    layer=None,
    encoding=None,
    columns=None,
    read_geometry=True,
    force_2d=False,
    skip_features=0,
    max_features=None,
    where=None,
    bbox=None,
    fids=None,
    sql=None,
    sql_dialect=None,
    return_fids=False,
):
    """Read OGR data source into numpy arrays.

    IMPORTANT: non-linear geometry types (e.g., MultiSurface) are converted
    to their linear approximations.

    Parameters
    ----------
    path_or_buffer : pathlib.Path or str, or bytes buffer
        A dataset path or URI, or raw buffer.
    layer : int or str, optional (default: first layer)
        If an integer is provided, it corresponds to the index of the layer
        with the data source.  If a string is provided, it must match the name
        of the layer in the data source.  Defaults to first layer in data source.
    encoding : str, optional (default: None)
        If present, will be used as the encoding for reading string values from
        the data source, unless encoding can be inferred directly from the data
        source.
    columns : list-like, optional (default: all columns)
        List of column names to import from the data source.  Column names must
        exactly match the names in the data source, and will be returned in
        the order they occur in the data source.  To avoid reading any columns,
        pass an empty list-like.
    read_geometry : bool, optional (default: True)
        If True, will read geometry into WKB.  If False, geometry will be None.
    force_2d : bool, optional (default: False)
        If the geometry has Z values, setting this to True will cause those to
        be ignored and 2D geometries to be returned
    skip_features : int, optional (default: 0)
        Number of features to skip from the beginning of the file before returning
        features.  Must be less than the total number of features in the file.
    max_features : int, optional (default: None)
        Number of features to read from the file.  Must be less than the total
        number of features in the file minus skip_features (if used).
    where : str, optional (default: None)
        Where clause to filter features in layer by attribute values.  Uses a
        restricted form of SQL WHERE clause, defined here:
        http://ogdi.sourceforge.net/prop/6.2.CapabilitiesMetadata.html
        Examples: "ISO_A3 = 'CAN'", "POP_EST > 10000000 AND POP_EST < 100000000"
    bbox : tuple of (xmin, ymin, xmax, ymax), optional (default: None)
        If present, will be used to filter records whose geometry intersects this
        box.  This must be in the same CRS as the dataset.  If GEOS is present
        and used by GDAL, only geometries that intersect this bbox will be
        returned; if GEOS is not available or not used by GDAL, all geometries
        with bounding boxes that intersect this bbox will be returned.
    fids : array-like, optional (default: None)
        Array of integer feature id (FID) values to select. Cannot be combined
        with other keywords to select a subset (`skip_features`, `max_features`,
        `where` or `bbox`). Note that the starting index is driver and file
        specific (e.g. typically 0 for Shapefile and 1 for GeoPackage, but can
        still depend on the specific file). The performance of reading a large
        number of features usings FIDs is also driver specific.
    return_fids : bool, optional (default: False)
        If True, will return the FIDs of the feature that were read.

    Returns
    -------
    (dict, fids, geometry, data fields)
        Returns a tuple of meta information about the data source in a dict,
        an ndarray of FIDs corresponding to the features that were read or None
        (if return_fids is False),
        an ndarray of geometry objects or None (if data source does not include
        geometry or read_geometry is False), a tuple of ndarrays for each field
        in the data layer.

        Meta is: {
            "crs": "<crs>",
            "fields": <ndarray of field names>,
            "encoding": "<encoding>",
            "geometry": "<geometry type>"
        }
    """
    path, buffer = get_vsi_path(path_or_buffer)

    try:
        result = ogr_read(
            path,
            layer=layer,
            encoding=encoding,
            columns=columns,
            read_geometry=read_geometry,
            force_2d=force_2d,
            skip_features=skip_features,
            max_features=max_features or 0,
            where=where,
            bbox=bbox,
            fids=fids,
            sql=sql,
            sql_dialect=sql_dialect,
            return_fids=return_fids,
        )
    finally:
        if buffer is not None:
            remove_virtual_file(path)

    return result


def read_arrow(
    path_or_buffer,
    /,
    layer=None,
    encoding=None,
    columns=None,
    read_geometry=True,
    force_2d=False,
    skip_features=0,
    max_features=None,
    where=None,
    bbox=None,
    fids=None,
    sql=None,
    sql_dialect=None,
    return_fids=False,
):
    """
    Read OGR data source into a pyarrow Table.

    See docstring of `read` for details.
    """
    try:
        import pyarrow  # noqa
    except ImportError:
        raise RuntimeError("the 'pyarrow' package is required to read using arrow")

    path, buffer = get_vsi_path(path_or_buffer)

    try:
        result = ogr_read_arrow(
            path,
            layer=layer,
            encoding=encoding,
            columns=columns,
            read_geometry=read_geometry,
            force_2d=force_2d,
            skip_features=skip_features,
            max_features=max_features or 0,
            where=where,
            bbox=bbox,
            fids=fids,
            sql=sql,
            sql_dialect=sql_dialect,
            return_fids=return_fids,
        )
    finally:
        if buffer is not None:
            remove_virtual_file(path)

    return result


def detect_driver(path):
    # try to infer driver from path
    parts = os.path.splitext(path)
    if len(parts) != 2:
        raise ValueError(
            f"Could not infer driver from path: {path}; please specify driver "
            "explicitly"
        )

    ext = parts[1].lower()
    driver = DRIVERS.get(ext, None)
    if driver is None:
        raise ValueError(
            f"Could not infer driver from path: {path}; please specify driver "
            "explicitly"
        )

    return driver


def _parse_options_names(xml):
    """Convert metadata xml to list of names"""
    # Based on Fiona's meta.py
    # (https://github.com/Toblerity/Fiona/blob/91c13ad8424641557a4e5f038f255f9b657b1bc5/fiona/meta.py)
    import xml.etree.ElementTree as ET

    options = []
    if xml:
        root = ET.fromstring(xml)
        for option in root.iter("Option"):
            # some options explicitly have scope='raster'
            if option.attrib.get("scope", "vector") != "raster":
                options.append(option.attrib["name"])

    return options


def _preprocess_options_key_value(options):
    """
    Preprocess options, eg `spatial_index=True` gets converted
    to `SPATIAL_INDEX="YES"`.
    """
    if not isinstance(options, dict):
        raise TypeError(f"Expected options to be a dict, got {type(options)}")

    result = {}
    for k, v in options.items():
        if v is None:
            continue
        k = k.upper()
        if isinstance(v, bool):
            v = "ON" if v else "OFF"
        else:
            v = str(v)
        result[k] = v
    return result


def write(
    path,
    geometry,
    field_data,
    fields,
    layer=None,
    driver=None,
    # derived from meta if roundtrip
    geometry_type=None,
    crs=None,
    encoding=None,
    promote_to_multi=None,
    nan_as_null=True,
    append=False,
    dataset_options=None,
    layer_options=None,
    **kwargs,
):
    if geometry_type is None:
        raise ValueError("geometry_type must be provided")

    if driver is None:
        driver = detect_driver(path)

    # verify that driver supports writing
    if not ogr_driver_supports_write(driver):
        raise DataSourceError(
            f"{driver} does not support write functionality in GDAL "
            f"{get_gdal_version_string()}"
        )

    # prevent segfault from: https://github.com/OSGeo/gdal/issues/5739
    if append and driver == "FlatGeobuf" and get_gdal_version() <= (3, 5, 0):
        raise RuntimeError(
            "append to FlatGeobuf is not supported for GDAL <= 3.5.0 due to segfault"
        )

    if promote_to_multi is None:
        promote_to_multi = (
            geometry_type.startswith("Multi")
            and driver in DRIVERS_NO_MIXED_SINGLE_MULTI
        )

    if crs is None:
        warnings.warn(
            "'crs' was not provided.  The output dataset will not have "
            "projection information defined and may not be usable in other "
            "systems."
        )

    # preprocess kwargs and split in dataset and layer creation options
    dataset_kwargs = _preprocess_options_key_value(dataset_options or {})
    layer_kwargs = _preprocess_options_key_value(layer_options or {})
    if kwargs:
        kwargs = _preprocess_options_key_value(kwargs)
        dataset_option_names = _parse_options_names(
            _get_driver_metadata_item(driver, "DMD_CREATIONOPTIONLIST")
        )
        layer_option_names = _parse_options_names(
            _get_driver_metadata_item(driver, "DS_LAYER_CREATIONOPTIONLIST")
        )
        for k, v in kwargs.items():
            if k in dataset_option_names:
                dataset_kwargs[k] = v
            elif k in layer_option_names:
                layer_kwargs[k] = v
            else:
                raise ValueError(f"unrecognized option '{k}' for driver '{driver}'")

    ogr_write(
        str(path),
        layer=layer,
        driver=driver,
        geometry=geometry,
        geometry_type=geometry_type,
        field_data=field_data,
        fields=fields,
        crs=crs,
        encoding=encoding,
        promote_to_multi=promote_to_multi,
        nan_as_null=nan_as_null,
        append=append,
        dataset_kwargs=dataset_kwargs,
        layer_kwargs=layer_kwargs,
    )
