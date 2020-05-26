import warnings

from pyogrio._io import ogr_read, ogr_read_info, ogr_list_layers, ogr_write


def read(path, layer=None, encoding=None, columns=None, read_geometry=True):
    """Read OGR data source.

    IMPORTANT: non-linear geometry types (e.g., MultiSurface) are converted
    to their linear approximations.

    Parameters
    ----------
    path : pathlib.Path or str
        data source path
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

    Returns
    -------
    (dict, geometry, data fields)
        Returns a tuple of meta information about the data source in a dict,
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

    return ogr_read(
        str(path),
        layer=layer,
        encoding=encoding,
        columns=columns,
        read_geometry=read_geometry,
    )


def read_info(path, layer=None, encoding=None):
    """Read information about an OGR data source.

    Parameters
    ----------
    path : str or pathlib.Path
    layer : [type], optional (default: None)
        name or index of layer in data source
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


def write(
    path,
    geometry,
    field_data,
    fields,
    layer=None,
    driver="ESRI Shapefile",
    # derived from meta if roundtrip
    geometry_type=None,
    crs=None,
    encoding=None,
    **kwargs
):

    if geometry_type is None:
        raise ValueError("geometry_type must be provided")

    if crs is None:
        warnings.warn(
            "'crs' was not provided.  The output dataset will not have "
            "projection information defined and may not be usable in other "
            "systems."
        )

    ogr_write(
        path,
        layer=layer,
        driver=driver,
        geometry=geometry,
        geometry_type=geometry_type,
        field_data=field_data,
        fields=fields,
        crs=crs,
        encoding=encoding,
        **kwargs
    )


def list_layers(path):
    return ogr_list_layers(str(path))
