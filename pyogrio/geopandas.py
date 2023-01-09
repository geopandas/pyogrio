from pyogrio.raw import DRIVERS_NO_MIXED_SINGLE_MULTI
from pyogrio.raw import detect_driver, read, read_arrow, write


def _stringify_path(path):
    """
    Convert path-like to a string if possible, pass-through other objects
    """
    if isinstance(path, str):
        return path

    # checking whether path implements the filesystem protocol
    if hasattr(path, "__fspath__"):
        return path.__fspath__()

    # pass-though other objects
    return path


def read_dataframe(
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
    fid_as_index=False,
    use_arrow=False,
):
    """Read from an OGR data source to a GeoPandas GeoDataFrame or Pandas DataFrame.
    If the data source does not have a geometry column or ``read_geometry`` is False,
    a DataFrame will be returned.

    Requires ``geopandas`` >= 0.8.

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
        If True, will read geometry into a GeoSeries.  If False, a Pandas DataFrame
        will be returned instead.
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
        Examples: ``"ISO_A3 = 'CAN'"``, ``"POP_EST > 10000000 AND POP_EST < 100000000"``
    bbox : tuple of (xmin, ymin, xmax, ymax) (default: None)
        If present, will be used to filter records whose geometry intersects this
        box.  This must be in the same CRS as the dataset.  If GEOS is present
        and used by GDAL, only geometries that intersect this bbox will be
        returned; if GEOS is not available or not used by GDAL, all geometries
        with bounding boxes that intersect this bbox will be returned.
    fids : array-like, optional (default: None)
        Array of integer feature id (FID) values to select. Cannot be combined
        with other keywords to select a subset (``skip_features``, ``max_features``,
        ``where``, ``bbox`` or ``sql``). Note that the starting index is driver and file
        specific (e.g. typically 0 for Shapefile and 1 for GeoPackage, but can
        still depend on the specific file). The performance of reading a large
        number of features usings FIDs is also driver specific.
    sql : str, optional (default: None)
        The sql statement to execute. Look at the sql_dialect parameter for
        more information on the syntax to use for the query. When combined
        with other keywords like ``columns``, ``skip_features``,
        ``max_features``, ``where`` or ``bbox``, those are applied after the
        sql query. Be aware that this can have an impact on performance,
        (e.g. filtering with the ``bbox`` keyword may not use
        spatial indexes).
        Cannot be combined with the ``layer`` or ``fids`` keywords.
    sql_dialect : str, optional (default: None)
        The sql dialect the sql statement is written in. Possible values:

          - **None**: if the datasource natively supports sql, the specific
            sql syntax for this datasource should be used (eg. SQLite,
            PostgreSQL, Oracle,...). If the datasource doesn't natively
            support sql, the 'OGRSQL_' dialect is the
            default.
          - 'OGRSQL_': can be used on any datasource. Performance can suffer
            when used on datasources with native support for sql.
          - 'SQLITE_': can be used on any datasource. All spatialite_
            functions can be used. Performance can suffer on datasources with
            native support for sql, except for GPKG and SQLite as this is
            their native sql dialect.

    fid_as_index : bool, optional (default: False)
        If True, will use the FIDs of the features that were read as the
        index of the GeoDataFrame.  May start at 0 or 1 depending on the driver.
    use_arrow : bool, default False
        Whether to use Arrow as the transfer mechanism of the read data
        from GDAL to Python (requires GDAL >= 3.6 and `pyarrow` to be
        installed). When enabled, this provides a further speed-up.

    Returns
    -------
    GeoDataFrame or DataFrame (if no geometry is present)

    .. _OGRSQL: https://gdal.org/user/ogr_sql_dialect.html#ogr-sql-dialect
    .. _SQLITE: https://gdal.org/user/sql_sqlite_dialect.html#sql-sqlite-dialect
    .. _spatialite: https://www.gaia-gis.it/gaia-sins/spatialite-sql-latest.html

    """
    try:
        import pandas as pd
        import geopandas as gp
        from geopandas.array import from_wkb

    except ImportError:
        raise ImportError("geopandas is required to use pyogrio.read_dataframe()")

    path_or_buffer = _stringify_path(path_or_buffer)

    read_func = read_arrow if use_arrow else read
    result = read_func(
        path_or_buffer,
        layer=layer,
        encoding=encoding,
        columns=columns,
        read_geometry=read_geometry,
        force_2d=force_2d,
        skip_features=skip_features,
        max_features=max_features,
        where=where,
        bbox=bbox,
        fids=fids,
        sql=sql,
        sql_dialect=sql_dialect,
        return_fids=fid_as_index,
    )

    if use_arrow:
        meta, table = result
        df = table.to_pandas()
        geometry_name = meta["geometry_name"] or "wkb_geometry"
        if geometry_name in df.columns:
            df["geometry"] = from_wkb(df.pop(geometry_name), crs=meta["crs"])
            return gp.GeoDataFrame(df, geometry="geometry")
        else:
            return df

    meta, index, geometry, field_data = result

    columns = meta["fields"].tolist()
    data = {columns[i]: field_data[i] for i in range(len(columns))}
    if fid_as_index:
        index = pd.Index(index, name="fid")
    else:
        index = None

    df = pd.DataFrame(data, columns=columns, index=index)

    if geometry is None or not read_geometry:
        return df

    geometry = from_wkb(geometry, crs=meta["crs"])

    return gp.GeoDataFrame(df, geometry=geometry)


# TODO: handle index properly
def write_dataframe(
    df,
    path,
    layer=None,
    driver=None,
    encoding=None,
    geometry_type=None,
    promote_to_multi=None,
    nan_as_null=True,
    append=False,
    dataset_options=None,
    layer_options=None,
    **kwargs,
):
    """
    Write GeoPandas GeoDataFrame to an OGR file format.

    Parameters
    ----------
    df : GeoDataFrame
        The data to write. For attribute columns of the "object" dtype,
        all values will be converted to strings to be written to the
        output file, except None and np.nan, which will be set to NULL
        in the output file.
    path : str
        path to file
    layer :str, optional (default: None)
        layer name
    driver : string, optional (default: None)
        The OGR format driver used to write the vector file. By default write_dataframe
        attempts to infer driver from path.
    encoding : str, optional (default: None)
        If present, will be used as the encoding for writing string values to
        the file.
    geometry_type : string, optional (default: None)
        By default, the geometry type of the layer will be inferred from the
        data, after applying the promote_to_multi logic. If the data only contains a
        single geometry type (after applying the logic of promote_to_multi), this type
        is used for the layer. If the data (still) contains mixed geometry types, the
        output layer geometry type will be set to "Unknown".

        This parameter does not modify the geometry, but it will try to force the layer
        type of the output file to this value. Use this parameter with caution because
        using a non-default layer geometry type may result in errors when writing the
        file, may be ignored by the driver, or may result in invalid files. Possible
        values are: "Unknown", "Point", "LineString", "Polygon", "MultiPoint",
        "MultiLineString", "MultiPolygon" or "GeometryCollection".
    promote_to_multi : bool, optional (default: None)
        If True, will convert singular geometry types in the data to their
        corresponding multi geometry type for writing. By default, will convert
        mixed singular and multi geometry types to multi geometry types for drivers
        that do not support mixed singular and multi geometry types. If False, geometry
        types will not be promoted, which may result in errors or invalid files when
        attempting to write mixed singular and multi geometry types to drivers that do
        not support such combinations.
    nan_as_null : bool, default True
        For floating point columns (float32 / float64), whether NaN values are
        written as "null" (missing value). Defaults to True because in pandas
        NaNs are typically used as missing value. Note that when set to False,
        behaviour is format specific: some formats don't support NaNs by
        default (e.g. GeoJSON will skip this property) or might treat them as
        null anyway (e.g. GeoPackage).
    append : bool, optional (default: False)
        If True, the data source specified by path already exists, and the
        driver supports appending to an existing data source, will cause the
        data to be appended to the existing records in the data source.
        NOTE: append support is limited to specific drivers and GDAL versions.
    dataset_options : dict, optional
        Dataset creation option (format specific) passed to OGR. Specify as
        a key-value dictionary.
    layer_options : dict, optional
        Layer creation option (format specific) passed to OGR. Specify as
        a key-value dictionary.
    **kwargs
        Additional driver-specific dataset or layer creation options passed
        to OGR. pyogrio will attempt to automatically pass those keywords
        either as dataset or as layer creation option based on the known
        options for the specific driver. Alternatively, you can use the
        explicit `dataset_options` or `layer_options` keywords to manually
        do this (for example if an option exists as both dataset and layer
        option).
    """
    # TODO: add examples to the docstring (e.g. OGR kwargs)
    try:
        import geopandas as gp
        from geopandas.array import to_wkb
        import pandas as pd

        # if geopandas is available so is pyproj
        from pyproj.enums import WktVersion

    except ImportError:
        raise ImportError("geopandas is required to use pyogrio.read_dataframe()")

    path = str(path)

    if not isinstance(df, gp.GeoDataFrame):
        raise ValueError("'df' must be a GeoDataFrame")

    if driver is None:
        driver = detect_driver(path)

    geometry_columns = df.columns[df.dtypes == "geometry"]
    if len(geometry_columns) == 0:
        raise ValueError("'df' does not have a geometry column")

    if len(geometry_columns) > 1:
        raise ValueError(
            "'df' must have only one geometry column. "
            "Multiple geometry columns are not supported for output using OGR."
        )

    geometry_column = geometry_columns[0]
    geometry = df[geometry_column]
    fields = [c for c in df.columns if not c == geometry_column]

    # TODO: may need to fill in pd.NA, etc
    field_data = [df[f].values for f in fields]

    # Determine geometry_type and/or promote_to_multi
    if geometry_type is None or promote_to_multi is None:
        tmp_geometry_type = "Unknown"

        # If there is data, infer layer geometry type + promote_to_multi
        if not df.empty:
            geometry_types = pd.Series(geometry.type.unique()).dropna().values
            if len(geometry_types) == 1:
                tmp_geometry_type = geometry_types[0]
                if promote_to_multi and tmp_geometry_type in (
                    "Point",
                    "LineString",
                    "Polygon",
                ):
                    tmp_geometry_type = f"Multi{tmp_geometry_type}"
            elif len(geometry_types) == 2:
                # Check if the types are corresponding multi + single types
                if "Polygon" in geometry_types and "MultiPolygon" in geometry_types:
                    multi_type = "MultiPolygon"
                elif (
                    "LineString" in geometry_types
                    and "MultiLineString" in geometry_types
                ):
                    multi_type = "MultiLineString"
                elif "Point" in geometry_types and "MultiPoint" in geometry_types:
                    multi_type = "MultiPoint"
                else:
                    multi_type = None

                # If they are corresponding multi + single types
                if multi_type is not None:
                    if (
                        promote_to_multi is None
                        and driver in DRIVERS_NO_MIXED_SINGLE_MULTI
                    ):
                        promote_to_multi = True
                    if promote_to_multi:
                        tmp_geometry_type = multi_type

        if geometry_type is None:
            geometry_type = tmp_geometry_type

    crs = None
    if geometry.crs:
        # TODO: this may need to be WKT1, due to issues
        # if possible use EPSG codes instead
        epsg = geometry.crs.to_epsg()
        if epsg:
            crs = f"EPSG:{epsg}"
        else:
            crs = geometry.crs.to_wkt(WktVersion.WKT1_GDAL)

    write(
        path,
        layer=layer,
        driver=driver,
        geometry=to_wkb(geometry.values),
        field_data=field_data,
        fields=fields,
        crs=crs,
        geometry_type=geometry_type,
        encoding=encoding,
        promote_to_multi=promote_to_multi,
        nan_as_null=nan_as_null,
        append=append,
        dataset_options=dataset_options,
        layer_options=layer_options,
        **kwargs,
    )
