from pyogrio._io import ogr_read


def read_dataframe(path, read_geometry=True, as_pygeos=False, **kwargs):
    """Read from an OGR data source to a pandas DataFrame.

    Requires pandas.

    Parameters
    ----------
    path : str
        path to file
    read_geometry : bool, optional (default True)
        if True and OGR data source includes geometry, will read geometry into "geometry" column.
    as_pygeos : bool, optional (default False)
        if True, geometry will be converted from WKB to pygeos geometry objects.

    Returns
    -------
    DataFrame
    """
    try:
        import pandas as pd

    except ImportError:
        raise ImportError("pandas is required to use pyogrio.read_dataframe()")

    meta, geometry, field_data = ogr_read(
        str(path), read_geometry=read_geometry, **kwargs
    )

    columns = meta["fields"].tolist()
    data = {columns[i]: field_data[i] for i in range(len(columns))}

    if geometry is not None and read_geometry:
        if as_pygeos:
            try:
                import pygeos as pg

            except ImportError:
                raise ImportError("pygeos is required to use as_pygeos option")

            geometry = pg.from_wkb(geometry)

        data["geometry"] = geometry
        columns += ["geometry"]

    df = pd.DataFrame(data, columns=columns)

    if read_geometry:
        # this is a hack and raises a warning
        # TODO: find better way to store crs
        df.crs = meta["crs"]

    return df
