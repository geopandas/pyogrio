import warnings

from pyogrio._env import GDALEnv
from pyogrio._compat import HAS_ARROW_API
from pyogrio.core import detect_write_driver
from pyogrio.errors import DataSourceError
from pyogrio.util import (
    get_vsi_path,
    vsi_path,
    _preprocess_options_key_value,
    _mask_to_wkb,
)

with GDALEnv():
    from pyogrio._io import ogr_open_arrow, ogr_read, ogr_write
    from pyogrio._ogr import (
        get_gdal_version,
        get_gdal_version_string,
        ogr_driver_supports_write,
        remove_virtual_file,
        _get_driver_metadata_item,
    )


DRIVERS_NO_MIXED_SINGLE_MULTI = {
    "FlatGeobuf",
    "GPKG",
}

DRIVERS_NO_MIXED_DIMENSIONS = {
    "FlatGeobuf",
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
    mask=None,
    fids=None,
    sql=None,
    sql_dialect=None,
    return_fids=False,
    datetime_as_string=False,
    **kwargs,
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
        Number of features to skip from the beginning of the file before
        returning features.  If greater than available number of features, an
        empty DataFrame will be returned.  Using this parameter may incur
        significant overhead if the driver does not support the capability to
        randomly seek to a specific feature, because it will need to iterate
        over all prior features.
    max_features : int, optional (default: None)
        Number of features to read from the file.
    where : str, optional (default: None)
        Where clause to filter features in layer by attribute values. If the data source
        natively supports SQL, its specific SQL dialect should be used (eg. SQLite and
        GeoPackage: `SQLITE`_, PostgreSQL). If it doesn't, the `OGRSQL WHERE`_ syntax
        should be used. Note that it is not possible to overrule the SQL dialect, this
        is only possible when you use the SQL parameter.
        Examples: ``"ISO_A3 = 'CAN'"``, ``"POP_EST > 10000000 AND POP_EST < 100000000"``
    bbox : tuple of (xmin, ymin, xmax, ymax), optional (default: None)
        If present, will be used to filter records whose geometry intersects this
        box.  This must be in the same CRS as the dataset.  If GEOS is present
        and used by GDAL, only geometries that intersect this bbox will be
        returned; if GEOS is not available or not used by GDAL, all geometries
        with bounding boxes that intersect this bbox will be returned.
        Cannot be combined with ``mask`` keyword.
    mask : Shapely geometry, optional (default: None)
        If present, will be used to filter records whose geometry intersects
        this geometry.  This must be in the same CRS as the dataset.  If GEOS is
        present and used by GDAL, only geometries that intersect this geometry
        will be returned; if GEOS is not available or not used by GDAL, all
        geometries with bounding boxes that intersect the bounding box of this
        geometry will be returned.  Requires Shapely >= 2.0.
        Cannot be combined with ``bbox`` keyword.
    fids : array-like, optional (default: None)
        Array of integer feature id (FID) values to select. Cannot be combined
        with other keywords to select a subset (``skip_features``,
        ``max_features``, ``where``, ``bbox``, or ``mask``). Note that the
        starting index is driver and file specific (e.g. typically 0 for
        Shapefile and 1 for GeoPackage, but can still depend on the specific
        file). The performance of reading a large number of features usings FIDs
        is also driver specific.
    sql : str, optional (default: None)
        The SQL statement to execute. Look at the sql_dialect parameter for more
        information on the syntax to use for the query. When combined with other
        keywords like ``columns``, ``skip_features``, ``max_features``,
        ``where``, ``bbox``, or ``mask``, those are applied after the SQL query.
        Be aware that this can have an impact on performance, (e.g. filtering
        with the ``bbox`` or ``mask`` keywords may not use spatial indexes).
        Cannot be combined with the ``layer`` or ``fids`` keywords.
    sql_dialect : str, optional (default: None)
        The SQL dialect the ``sql`` statement is written in. Possible values:

          - **None**: if the data source natively supports SQL, its specific SQL dialect
            will be used by default (eg. SQLite and Geopackage: `SQLITE`_, PostgreSQL).
            If the data source doesn't natively support SQL, the `OGRSQL`_ dialect is
            the default.
          - '`OGRSQL`_': can be used on any data source. Performance can suffer
            when used on data sources with native support for SQL.
          - '`SQLITE`_': can be used on any data source. All spatialite_
            functions can be used. Performance can suffer on data sources with
            native support for SQL, except for Geopackage and SQLite as this is
            their native SQL dialect.

    return_fids : bool, optional (default: False)
        If True, will return the FIDs of the feature that were read.
    datetime_as_string : bool, optional (default: False)
        If True, will return datetime dtypes as detected by GDAL as a string
        array (which can be used to extract timezone info), instead of
        a datetime64 array.

    **kwargs
        Additional driver-specific dataset open options passed to OGR.  Invalid
        options will trigger a warning.

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
            "dtypes": <ndarray of numpy dtypes corresponding to fields>
            "encoding": "<encoding>",
            "geometry_type": "<geometry type>"
        }

    .. _OGRSQL:

        https://gdal.org/user/ogr_sql_dialect.html#ogr-sql-dialect

    .. _OGRSQL WHERE:

        https://gdal.org/user/ogr_sql_dialect.html#where

    .. _SQLITE:

        https://gdal.org/user/sql_sqlite_dialect.html#sql-sqlite-dialect

    .. _spatialite:

        https://www.gaia-gis.it/gaia-sins/spatialite-sql-latest.html

    """
    path, buffer = get_vsi_path(path_or_buffer)

    dataset_kwargs = _preprocess_options_key_value(kwargs) if kwargs else {}

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
            mask=_mask_to_wkb(mask),
            fids=fids,
            sql=sql,
            sql_dialect=sql_dialect,
            return_fids=return_fids,
            dataset_kwargs=dataset_kwargs,
            datetime_as_string=datetime_as_string,
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
    mask=None,
    fids=None,
    sql=None,
    sql_dialect=None,
    return_fids=False,
    **kwargs,
):
    """
    Read OGR data source into a pyarrow Table.

    See docstring of `read` for parameters.

    Returns
    -------
    (dict, pyarrow.Table)

        Returns a tuple of meta information about the data source in a dict,
        and a pyarrow Table with data.

        Meta is: {
            "crs": "<crs>",
            "fields": <ndarray of field names>,
            "encoding": "<encoding>",
            "geometry_type": "<geometry_type>",
            "geometry_name": "<name of geometry column in arrow table>",
        }
    """
    from pyarrow import Table

    if skip_features < 0:
        raise ValueError("'skip_features' must be >= 0")

    if max_features is not None and max_features < 0:
        raise ValueError("'max_features' must be >= 0")

    # limit batch size to max_features if set
    if "batch_size" in kwargs:
        batch_size = kwargs.pop("batch_size")
    else:
        batch_size = 65_536

    if max_features is not None and max_features < batch_size:
        batch_size = max_features

    # handle skip_features internally within open_arrow if GDAL >= 3.8.0
    gdal_skip_features = 0
    if get_gdal_version() >= (3, 8, 0):
        gdal_skip_features = skip_features
        skip_features = 0

    with open_arrow(
        path_or_buffer,
        layer=layer,
        encoding=encoding,
        columns=columns,
        read_geometry=read_geometry,
        force_2d=force_2d,
        where=where,
        bbox=bbox,
        mask=mask,
        fids=fids,
        sql=sql,
        sql_dialect=sql_dialect,
        return_fids=return_fids,
        skip_features=gdal_skip_features,
        batch_size=batch_size,
        **kwargs,
    ) as source:
        meta, reader = source

        if max_features is not None:
            batches = []
            count = 0
            while True:
                try:
                    batch = reader.read_next_batch()
                    batches.append(batch)

                    count += len(batch)
                    if count >= (skip_features + max_features):
                        break

                except StopIteration:
                    break

            # use combine_chunks to release the original memory that included
            # too many features
            table = (
                Table.from_batches(batches, schema=reader.schema)
                .slice(skip_features, max_features)
                .combine_chunks()
            )

        elif skip_features > 0:
            table = reader.read_all().slice(skip_features).combine_chunks()

        else:
            table = reader.read_all()

    return meta, table


def open_arrow(
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
    mask=None,
    fids=None,
    sql=None,
    sql_dialect=None,
    return_fids=False,
    batch_size=65_536,
    **kwargs,
):
    """
    Open OGR data source as a stream of pyarrow record batches.

    See docstring of `read` for parameters.

    The RecordBatchStreamReader is reading from a stream provided by OGR and must not be
    accessed after the OGR dataset has been closed, i.e. after the context manager has
    been closed.

    Examples
    --------

    >>> from pyogrio.raw import open_arrow
    >>> import pyarrow as pa
    >>> import shapely
    >>>
    >>> with open_arrow(path) as source:
    >>>     meta, reader = source
    >>>     for table in reader:
    >>>         geometries = shapely.from_wkb(table[meta["geometry_name"]])

    Returns
    -------
    (dict, pyarrow.RecordBatchStreamReader)

        Returns a tuple of meta information about the data source in a dict,
        and a pyarrow RecordBatchStreamReader with data.

        Meta is: {
            "crs": "<crs>",
            "fields": <ndarray of field names>,
            "encoding": "<encoding>",
            "geometry_type": "<geometry_type>",
            "geometry_name": "<name of geometry column in arrow table>",
        }
    """
    if not HAS_ARROW_API:
        raise RuntimeError("pyarrow and GDAL>= 3.6 required to read using arrow")

    path, buffer = get_vsi_path(path_or_buffer)

    dataset_kwargs = _preprocess_options_key_value(kwargs) if kwargs else {}

    try:
        return ogr_open_arrow(
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
            mask=_mask_to_wkb(mask),
            fids=fids,
            sql=sql,
            sql_dialect=sql_dialect,
            return_fids=return_fids,
            dataset_kwargs=dataset_kwargs,
            batch_size=batch_size,
        )
    finally:
        if buffer is not None:
            remove_virtual_file(path)


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


def write(
    path,
    geometry,
    field_data,
    fields,
    field_mask=None,
    layer=None,
    driver=None,
    # derived from meta if roundtrip
    geometry_type=None,
    crs=None,
    encoding=None,
    promote_to_multi=None,
    nan_as_null=True,
    append=False,
    dataset_metadata=None,
    layer_metadata=None,
    metadata=None,
    dataset_options=None,
    layer_options=None,
    gdal_tz_offsets=None,
    **kwargs,
):
    # if dtypes is given, remove it from kwargs (dtypes is included in meta returned by
    # read, and it is convenient to pass meta directly into write for round trip tests)
    kwargs.pop("dtypes", None)
    path = vsi_path(str(path))

    if driver is None:
        driver = detect_write_driver(path)

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

    if metadata is not None:
        if layer_metadata is not None:
            raise ValueError("Cannot pass both metadata and layer_metadata")
        layer_metadata = metadata

    # validate metadata types
    for metadata in [dataset_metadata, layer_metadata]:
        if metadata is not None:
            for k, v in metadata.items():
                if not isinstance(k, str):
                    raise ValueError(f"metadata key {k} must be a string")

                if not isinstance(v, str):
                    raise ValueError(f"metadata value {v} must be a string")

    if geometry is not None and promote_to_multi is None:
        promote_to_multi = (
            geometry_type.startswith("Multi")
            and driver in DRIVERS_NO_MIXED_SINGLE_MULTI
        )

    if geometry is not None and crs is None:
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
        path,
        layer=layer,
        driver=driver,
        geometry=geometry,
        geometry_type=geometry_type,
        field_data=field_data,
        field_mask=field_mask,
        fields=fields,
        crs=crs,
        encoding=encoding,
        promote_to_multi=promote_to_multi,
        nan_as_null=nan_as_null,
        append=append,
        dataset_metadata=dataset_metadata,
        layer_metadata=layer_metadata,
        dataset_kwargs=dataset_kwargs,
        layer_kwargs=layer_kwargs,
        gdal_tz_offsets=gdal_tz_offsets,
    )
