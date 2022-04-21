import os

from pyogrio._env import GDALEnv
from pyogrio.raw import read, write


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
    fid_as_index=False,
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
        box.  This must be in the same CRS as the dataset.
    fids : array-like, optional (default: None)
        Array of integer feature id (FID) values to select. Cannot be combined
        with other keywords to select a subset (``skip_features``, ``max_features``,
        ``where`` or ``bbox``). Note that the starting index is driver and file
        specific (e.g. typically 0 for Shapefile and 1 for GeoPackage, but can
        still depend on the specific file). The performance of reading a large
        number of features usings FIDs is also driver specific.
    fid_as_index : bool, optional (default: False)
        If True, will use the FIDs of the features that were read as the
        index of the GeoDataFrame.  May start at 0 or 1 depending on the driver.

    Returns
    -------
    GeoDataFrame or DataFrame (if no geometry is present)
    """
    try:
        with GDALEnv():
            import pandas as pd
            import geopandas as gp
            from geopandas.array import from_wkb

    except ImportError:
        raise ImportError("geopandas is required to use pyogrio.read_dataframe()")

    path_or_buffer = _stringify_path(path_or_buffer)

    meta, index, geometry, field_data = read(
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
        return_fids=fid_as_index,
    )

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
def write_dataframe(df, path, layer=None, driver=None, encoding=None, promote_to_multi=None, **kwargs):
    """
    Write GeoPandas GeoDataFrame to an OGR file format.

    Parameters
    ----------
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
    promote_to_multi: bool, optional (default: None) 
        If True, will convert singular geometry types to their corresponding
        multi geometry type. By default, will convert convert mixed singular
        and multi geometry types to multi geometry types for drivers that do
        not support mixed singular and multi geometry types.

    **kwargs
        The kwargs passed to OGR.
    """
    # TODO: add examples to the docstring (e.g. OGR kwargs)
    try:
        with GDALEnv():
            import geopandas as gp
            from geopandas.array import to_wkb

            # if geopandas is available so is pyproj
            from pyproj.enums import WktVersion

    except ImportError:
        raise ImportError("geopandas is required to use pyogrio.read_dataframe()")

    path = str(path)

    if not isinstance(df, gp.GeoDataFrame):
        raise ValueError("'df' must be a GeoDataFrame")

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

    geometry_type = None
    if not df.empty:
        # TODO: validate geometry types, not all combinations are valid
        geometry_types = geometry.type.unique()
        if len(geometry_types) == 1:
            geometry_type = geometry_types[0]
        elif len(geometry_types == 2):
            if "Polygon" in geometry_types and "MultiPolygon" in geometry_types:
                geometry_type = "MultiPolygon"
            elif "LineString" in geometry_types and "MultiLineString" in geometry_types:
                geometry_type = "MultiLineString"
            elif "Point" in geometry_types and "MultiPoint" in geometry_types:
                geometry_type = "MultiPoint"
            
    if geometry_type is None:
        geometry_type = "Unknown"

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
        **kwargs
    )
