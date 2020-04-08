from pyogrio._io import ogr_read, ogr_list_layers
from pyogrio.pandas import read_dataframe


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
    columns : list-like, optional
        List of column names to import from the data source.
    read_geometry : bool, optional (default: True)
        If True, will read geometry into WKB.  If False, geometry will be None.

    Returns
    -------
    (dict, geometry, data fields)
        Returns a tuple of meta information about the data source in a dict,
        an ndarray of geometry objects or None (if data source does not include
        geometry or read_geometry is False), a tuple of ndarrays for each field
        in the data layer.
    """
    return ogr_read(
        str(path), layer=layer, encoding=encoding, columns=columns, to_linear=to_linear
    )


def write(path, meta, data, driver=None):
    raise NotImplementedError("Not built!")


def list_layers(path):
    return ogr_list_layers(str(path))
