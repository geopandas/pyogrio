import warnings
import os

from pyogrio._env import GDALEnv
from pyogrio.errors import DataSourceError
from pyogrio.util import vsi_path

with GDALEnv():
    from pyogrio._io import ogr_read, ogr_read_info, ogr_list_layers, ogr_write
    from pyogrio._ogr import buffer_to_virtual_file, remove_virtual_file


DRIVERS = {
    ".gpkg": "GPKG",
    ".shp": "ESRI Shapefile",
    ".json": "GeoJSON",
    ".fgb": "FlatGeobuf",
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
    return_fids=False,
):
    """Read OGR data source.

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
        box.  This must be in the same CRS as the dataset.
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
    if hasattr(path_or_buffer, "read"):
        path_or_buffer = path_or_buffer.read()

    from_buffer = False
    if isinstance(path_or_buffer, bytes):
        from_buffer = True
        ext = ""
        is_zipped = path_or_buffer[:4].startswith(b'PK\x03\x04')
        if is_zipped:
            ext = ".zip"
        path = buffer_to_virtual_file(path_or_buffer, ext=ext)
        if is_zipped:
            path = "/vsizip/" + path
    else:
        path = vsi_path(str(path_or_buffer))

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
            return_fids=return_fids,
        )
    finally:
        if from_buffer:
            remove_virtual_file(path)

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
    **kwargs,
):

    if geometry_type is None:
        raise ValueError("geometry_type must be provided")

    if driver is None:
        # try to infer driver from path
        parts = os.path.splitext(path)
        if len(parts) != 2:
            raise ValueError(
                f"Could not infer driver from path: {path}; please specify driver explicitly"
            )

        ext = parts[1].lower()
        driver = DRIVERS.get(ext, None)
        if driver is None:
            raise ValueError(
                f"Could not infer driver from path: {path}; please specify driver explicitly"
            )

    if crs is None:
        warnings.warn(
            "'crs' was not provided.  The output dataset will not have "
            "projection information defined and may not be usable in other "
            "systems."
        )

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
        **kwargs,
    )
