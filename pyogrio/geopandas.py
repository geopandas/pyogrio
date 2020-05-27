import os

from pyogrio import read, write


def read_dataframe(path, read_geometry=True, **kwargs):
    """Read from an OGR data source to a GeoPandas GeoDataFrame or Pandas DataFrame.
    If the data source does not have a geometry column or `read_geometry` is False,
    a DataFrame will be returned.

    Requires geopandas >= 0.8.

    Parameters
    ----------
    path : str
        path to file
    read_geometry : bool (default: True)
        If False, geometry will not be read, and a DataFrame will be returned instead.
    kwargs : dict
        see pyogrio.read() for kwargs

    Returns
    -------
    GeoDataFrame or DataFrame
    """
    try:
        import pandas as pd
        import geopandas as gp
        from geopandas.array import from_wkb

    except ImportError:
        raise ImportError("geopandas is required to use pyogrio.read_dataframe()")

    path = str(path)

    if not os.path.exists(path):
        raise ValueError(f"'{path}' does not exist")

    meta, geometry, field_data = read(path, **kwargs)

    columns = meta["fields"].tolist()
    data = {columns[i]: field_data[i] for i in range(len(columns))}
    df = pd.DataFrame(data, columns=columns)

    if geometry is None or not read_geometry:
        return df

    geometry = from_wkb(geometry, crs=meta["crs"])

    return gp.GeoDataFrame(df, geometry=geometry)


# TODO: handle index properly
def write_dataframe(
    df, path, layer=None, driver="ESRI Shapefile", encoding=None, **kwargs
):
    import geopandas as gp
    from geopandas.array import to_wkb

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

    # TODO: validate geometry types, not all combinations are valid
    geometry_type = geometry.type.unique()[0]

    crs = None
    if geometry.crs:
        crs = geometry.crs.to_wkt()

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
    )
