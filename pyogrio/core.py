from pyogrio._env import GDALEnv

with GDALEnv():
    from pyogrio._io import ogr_list_layers, ogr_read_info


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
