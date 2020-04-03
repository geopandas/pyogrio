"""IO support for OGR vector data sources

TODO:
* better handling of drivers
* select fields
* handle null field values
* handle date / time fields
"""

import datetime
import locale
import logging

from libc.stdint cimport uint8_t
from libc.stdlib cimport malloc, free

cimport cython
import numpy as np
cimport numpy as np

from pyogrio._ogr cimport *
from pyogrio._err cimport *
from pyogrio._err import CPLE_BaseError, NullPointerError
from pyogrio._geometry cimport get_geometry_type
from pyogrio.errors import DriverError



log = logging.getLogger(__name__)


cdef const char * STRINGSASUTF8 = "StringsAsUTF8"


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

# Mapping of OGR integer field types to Python field type names.
# TODO: numpy types
FIELD_TYPES = [
    'int32',        # OFTInteger, Simple 32bit integer
    None,           # OFTIntegerList, List of 32bit integers, not supported
    'float64',      # OFTReal, Double Precision floating point
    None,           # OFTRealList, List of doubles, not supported
    'object',          # OFTString, String of UTF-8 chars
    None,           # OFTStringList, Array of strings, not supported
    None,           # OFTWideString, deprecated, not supported
    None,           # OFTWideStringList, deprecated, not supported
    'object', # 'bytes',        # OFTBinary, Raw Binary data
    'datetime64[D]',# OFTDate, Date
    None,       # OFTTime, Time, NOTE: not directly supported in numpy
    'datetime64[s]',# OFTDateTime, Date and Time
    'int64',        # OFTInteger64, Single 64bit integer
    None            # OFTInteger64List, List of 64bit integers, not supported
]


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
    cdef void* ogr_dataset = NULL
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

        ogr_dataset = exc_wrap_pointer(
            GDALOpenEx(path_c, flags, <const char *const *>ogr_drivers, <const char *const *>open_opts, NULL)
        )
        return ogr_dataset

    except NullPointerError:
        raise DriverError("Failed to open dataset (mode={}): {}".format(mode, path_c.decode("utf-8")))

    except CPLE_BaseError as exc:
        raise DriverError(str(exc))

    finally:
        CSLDestroy(ogr_drivers)
        CSLDestroy(open_opts)



cdef get_crs(void *ogr_layer):
    """Read CRS from layer as EPSG:<code> if available or WKT.

    Parameters
    ----------
    ogr_layer : pointer to open OGR layer

    Returns
    -------
    str or None
        EPSG:<code> or WKT
    """
    cdef void *ogr_crs = NULL
    cdef const char *authority_key = NULL
    cdef const char *authority_val = NULL
    cdef char *ogr_wkt = NULL

    try:
        ogr_crs = exc_wrap_pointer(OGR_L_GetSpatialRef(ogr_layer))
    except NullPointerError:

        # no coordinate system defined
        print("No CRS defined")
        return None

    # If CRS can be decoded to an EPSG code, use that.
    # The following pointers will be NULL if it cannot be decoded.
    retval = OSRAutoIdentifyEPSG(ogr_crs)
    authority_key = <const char *>OSRGetAuthorityName(ogr_crs, NULL)
    authority_val = <const char *>OSRGetAuthorityCode(ogr_crs, NULL)

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


cdef detect_encoding(void *ogr_dataset, void *ogr_layer):
    """Attempt to detect the encoding of the layer.
    If it supports UTF-8, use that.
    If it is a shapefile, it must be ISO-8859-1.

    Parameters
    ----------
    ogr_dataset : pointer to open OGR dataset
    ogr_layer : pointer to open OGR layer

    Returns
    -------
    str or None
    """
    cdef void *ogr_driver

    if OGR_L_TestCapability(ogr_layer, STRINGSASUTF8):
        return 'UTF-8'

    ogr_driver = GDALGetDatasetDriver(ogr_dataset)
    driver = OGR_Dr_GetName(ogr_driver).decode("UTF-8")
    if driver == 'ESRI Shapefile':
        return 'ISO-8859-1'

    return None


# TODO: ignore fields?
cdef get_fields(void *ogr_layer, encoding):
    """Get field names and types for layer.

    Parameters
    ----------
    ogr_layer : pointer to open OGR layer
    encoding : str
        encoding to use when reading field name

    Returns
    -------
    ndarray(n, 4)
        array of index, ogr type, name, numpy type
    """
    cdef int i
    cdef int field_count
    cdef void *ogr_featuredef = NULL
    cdef void *ogr_fielddef = NULL
    cdef const char *key_c

    # if self.collection.ignore_fields:
    #     ignore_fields = self.collection.ignore_fields
    # else:
    #     ignore_fields = set()

    ogr_featuredef = OGR_L_GetLayerDefn(ogr_layer)
    if ogr_featuredef == NULL:
        raise ValueError("Null feature definition")

    field_count = OGR_FD_GetFieldCount(ogr_featuredef)

    fields = np.empty(shape=(field_count, 4), dtype=np.object)
    fields_view = fields[:,:]

    for i in range(field_count):
        ogr_fielddef = OGR_FD_GetFieldDefn(ogr_featuredef, i)
        if ogr_fielddef == NULL:
            raise ValueError("Null field definition")

        field_name = get_string(OGR_Fld_GetNameRef(ogr_fielddef), encoding=encoding)

    #     # TODO:
    #     if name in ignore_fields:
    #         continue

        field_type = OGR_Fld_GetType(ogr_fielddef)
        np_type = FIELD_TYPES[field_type]
        if not np_type:
            log.warning(
                f"Skipping field {field_name}: unsupported OGR type: {field_type}")
            continue

        fields_view[i,0] = i
        fields_view[i,1] = field_type
        fields_view[i,2] = field_name
        fields_view[i,3] = np_type

    return fields


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef get_features(void *ogr_layer, object[:,:] fields, encoding, uint8_t read_geometry, ):
    cdef void * ogr_feature = NULL
    cdef void * ogr_geometry = NULL
    cdef char * wkb = NULL
    cdef int i
    cdef int j
    cdef int success
    cdef int field_index
    cdef int count
    cdef int ret_length
    cdef unsigned char *bin_value
    cdef int year = 0
    cdef int month = 0
    cdef int day = 0
    cdef int hour = 0
    cdef int minute = 0
    cdef int second = 0
    cdef int timezone = 0


    # make sure layer is read from beginning
    OGR_L_ResetReading(ogr_layer)

    count = OGR_L_GetFeatureCount(ogr_layer, 1)

    if read_geometry:
        geometries = np.empty(shape=(count, ), dtype='object')
        geom_view = geometries[:]

    else:
        geometries = None

    field_iter = range(fields.shape[0])
    field_indexes = fields[:,0]
    field_ogr_types = fields[:,1]

    field_data = [
        np.empty(shape=(count, ),
        dtype=fields[field_index,3]) for field_index in field_indexes
    ]

    field_data_view = [field_data[field_index][:] for field_index in field_indexes]

    for i in range(count):
        ogr_feature = OGR_L_GetNextFeature(ogr_layer)

        if ogr_feature == NULL:
            raise ValueError("Failed to read feature {}".format(i))

        if read_geometry:
            ogr_geometry = OGR_F_GetGeometryRef(ogr_feature)

            if ogr_geometry == NULL:
                geom_view[i] = None

            else:
                try:
                    ret_length = OGR_G_WkbSize(ogr_geometry)
                    wkb = <char*>malloc(sizeof(char)*ret_length)
                    OGR_G_ExportToWkb(ogr_geometry, 1, wkb)
                    geom_view[i] = wkb[:ret_length]

                finally:
                    free(wkb)

        for j in field_iter:
            field_index = field_indexes[j]
            field_type = field_ogr_types[j]
            data = field_data_view[j]

            # TODO: handle null values
            if field_type == OFTInteger:
                data[i] = OGR_F_GetFieldAsInteger(ogr_feature, field_index)

            elif field_type == OFTInteger64:
                data[i] = OGR_F_GetFieldAsInteger64(ogr_feature, field_index)

            elif field_type == OFTReal:
                data[i] = OGR_F_GetFieldAsDouble(ogr_feature, field_index)

            elif field_type == OFTString:
                value = get_string(OGR_F_GetFieldAsString(ogr_feature, field_index), encoding=encoding)
                data[i] = value

            elif field_type == OFTBinary:
                bin_value = OGR_F_GetFieldAsBinary(ogr_feature, field_index, &ret_length)
                data[i] = bin_value[:ret_length]

            elif field_type == OFTDateTime or field_type == OFTDate:
                success = OGR_F_GetFieldAsDateTime(
                    ogr_feature, field_index, &year, &month, &day, &hour, &minute, &second, &timezone)

                if not success:
                    data[i] = None

                elif field_type == OFTDate:
                    data[i] = datetime.date(year, month, day).isoformat()

                elif field_type == OFTDateTime:
                    data[i] = datetime.datetime(year, month, day, hour, minute, second).isoformat()

    return (geometries, field_data)



def ogr_read(path, layer=None, encoding=None, read_geometry=True, **kwargs):
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

    # Encoding is derived from the dataset, from the user, or from the system locale
    encoding = (
        detect_encoding(ogr_dataset, ogr_layer)
        or encoding
        or locale.getpreferredencoding()
    )

    fields = get_fields(ogr_layer, encoding)

    geometry_type = get_geometry_type(ogr_layer)

    meta = {
        'crs': crs,
        'encoding': encoding,
        'fields': fields[:,2:],
        'geometry': geometry_type
    }

    # FIXME:
    geometries, field_data = get_features(
        ogr_layer,
        fields,
        encoding,
        read_geometry=read_geometry and geometry_type is not None
    )


    if ogr_dataset != NULL:
        GDALClose(ogr_dataset)
    ogr_dataset = NULL


    return (
        meta,
        geometries,
        field_data
    )


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