import logging

from pyogrio._ogr cimport *
from pyogrio._err cimport *
from pyogrio._err import CPLE_BaseError, NullPointerError
from pyogrio.errors import DriverError

import numpy as np
cimport numpy as np


log = logging.getLogger(__name__)



# TODO: port drivers from fiona::drvsupport.py

# driver:mode
# minimally ported from fiona::drvsupport.py
DRIVERS = {
    "ESRI Shapefile": "raw",
    "GeoJSON": "raw",
    "GeoJSONSeq": "rw",
    "GPKG": "raw",
    "OpenFileGDB": "r",
}


def get_string(char *c_str, str encoding="UTF-8"):
    """Get Python string from a char *

    IMPORTANT: the char * must still be freed
    by the caller.

    Parameters
    ----------
    c_str : char *
    encoding : str, optional (default: UTF-8)

    Returns
    -------
    Python string
    """
    py_str = c_str
    return py_str.decode(encoding)


# ported from fiona::_shim22.pyx::gdal_open_vector
cdef void* ogr_open(const char* path_c, int mode, drivers, options) except NULL:
    cdef void* cogr_ds = NULL
    cdef char **ogr_drivers = NULL
    cdef void* ogr_driver = NULL
    cdef char **open_opts = NULL

    # TODO: move to env?
    GDALAllRegister()

    flags = GDAL_OF_VECTOR | GDAL_OF_VERBOSE_ERROR
    if mode == 1:
        flags |= GDAL_OF_UPDATE
    else:
        flags |= GDAL_OF_READONLY

    # TODO: specific driver support may not be needed
    # for name in drivers:
    #     name_b = name.encode()
    #     name_c = name_b
    #     ogr_driver = GDALGetDriverByName(name_c)
    #     if ogr_driver != NULL:
    #         ogr_drivers = CSLAddString(ogr_drivers, name_c)

    # TODO: other open opts from fiona
    open_opts = CSLAddNameValue(open_opts, "VALIDATE_OPEN_OPTIONS", "NO")

    try:
        # When GDAL complains that file is not a supported file format, it is
        # most likely because we didn't call GDALAllRegister() prior to getting here

        cogr_ds = exc_wrap_pointer(
            GDALOpenEx(path_c, flags, <const char *const *>ogr_drivers, <const char *const *>open_opts, NULL)
        )
        return cogr_ds

    except NullPointerError:
        raise DriverError("Failed to open dataset (mode={}): {}".format(mode, path_c.decode("utf-8")))

    except CPLE_BaseError as exc:
        raise DriverError(str(exc))

    finally:
        CSLDestroy(ogr_drivers)
        CSLDestroy(open_opts)



cdef get_crs(void *ogr_layer):
    cdef void *ogr_crs = NULL
    cdef const char *authority_key = NULL
    cdef const char *authority_val = NULL
    cdef char *ogr_wkt = NULL

    try:
        ogr_crs = exc_wrap_pointer(OGR_L_GetSpatialRef(ogr_layer))
    except NullPointerError:
        # no coordinate system defined
        log.debug("No CRS defined")
        return None

    retval = OSRAutoIdentifyEPSG(ogr_crs)
    if retval > 0:
        log.info("Failed to auto identify EPSG: %d", retval)

    try:
        authority_key = <const char *>exc_wrap_pointer(<void *>OSRGetAuthorityName(ogr_crs, NULL))
        authority_val = <const char *>exc_wrap_pointer(<void *>OSRGetAuthorityCode(ogr_crs, NULL))

    except CPLE_BaseError as exc:
        log.debug("{}".format(exc))

    if authority_key != NULL and authority_val != NULL:
        key = get_string(authority_key)
        if key == 'EPSG':
            value = get_string(authority_val)
            return f"EPSG:{value}"

    try:
        OSRExportToWkt(ogr_crs, &ogr_wkt)
        if ogr_wkt == NULL:
            raise ValueError("Null projection")

        wkt = get_string(ogr_wkt)

    finally:
        CPLFree(ogr_wkt)
        return wkt


def ogr_read(path, layer=None, **kwargs):
    cdef const char *path_c = NULL
    cdef void *ogr_dataset = NULL
    cdef void *ogr_layer = NULL

    path_b = path.encode('utf-8')
    path_c = path_b

    # layer defaults to index 0
    if layer is None:
        layer = 0

    # all DRIVERS support read
    ogr_dataset = ogr_open(path_c, 0, DRIVERS, kwargs)

    if isinstance(layer, str):
        name_b = layer.encode('utf-8')
        name_c = name_b
        ogr_layer = GDALDatasetGetLayerByName(ogr_dataset, name_c)

    elif isinstance(layer, int):
        ogr_layer = GDALDatasetGetLayer(ogr_dataset, layer)

    if ogr_layer == NULL:
        raise ValueError(f"Layer '{layer}' could not be opened")

    crs = get_crs(ogr_layer)


    if ogr_dataset != NULL:
        GDALClose(ogr_dataset)
    ogr_dataset = NULL


def ogr_list_layers(str path):
    cdef const char *path_c = NULL
    cdef const char *ogr_name = NULL
    cdef void *ogr_dataset = NULL
    cdef void *ogr_layer = NULL

    path_b = path.encode('utf-8')
    path_c = path_b

    ogr_dataset = ogr_open(path_c, 0, DRIVERS, None)

    layer_count = GDALDatasetGetLayerCount(ogr_dataset)

    layer_names = np.empty(shape=(layer_count, ), dtype=np.object)
    view = layer_names[:]
    for i in range(layer_count):
        ogr_layer = GDALDatasetGetLayer(ogr_dataset, i)
        ogr_name = OGR_L_GetName(ogr_layer)

        name = get_string(ogr_name)
        # layer_names[i] = name
        view[i] = name

    if ogr_dataset != NULL:
        GDALClose(ogr_dataset)
    ogr_dataset = NULL

    return layer_names