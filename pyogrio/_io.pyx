#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

"""IO support for OGR vector data sources
"""


import datetime
import locale
import logging
import os
import warnings

from libc.stdint cimport uint8_t
from libc.stdlib cimport malloc, free
from libc.string cimport strlen

cimport cython
import numpy as np
cimport numpy as np

from pyogrio._ogr cimport *
from pyogrio._err cimport *
from pyogrio._err import CPLE_BaseError, CPLE_NotSupportedError, NullPointerError
from pyogrio._geometry cimport get_geometry_type, get_geometry_type_code
from pyogrio.errors import CRSError, DataSourceError, DataLayerError, GeometryError, FieldError, FeatureError


log = logging.getLogger(__name__)


# Mapping of OGR integer field types to Python field type names
# (index in array is the integer field type)
FIELD_TYPES = [
    'int32',        # OFTInteger, Simple 32bit integer
    None,           # OFTIntegerList, List of 32bit integers, not supported
    'float64',      # OFTReal, Double Precision floating point
    None,           # OFTRealList, List of doubles, not supported
    'object',       # OFTString, String of UTF-8 chars
    None,           # OFTStringList, Array of strings, not supported
    None,           # OFTWideString, deprecated, not supported
    None,           # OFTWideStringList, deprecated, not supported
    'object',       #  OFTBinary, Raw Binary data
    'datetime64[D]',# OFTDate, Date
    None,           # OFTTime, Time, NOTE: not directly supported in numpy
    'datetime64[s]',# OFTDateTime, Date and Time
    'int64',        # OFTInteger64, Single 64bit integer
    None            # OFTInteger64List, List of 64bit integers, not supported
]

FIELD_SUBTYPES = {
    OFSTNone: None,           # No subtype
    OFSTBoolean: "bool",      # Boolean integer
    OFSTInt16: "int16",       # Signed 16-bit integer
    OFSTFloat32: "float32",   # Single precision (32 bit) floating point
}

# Mapping of numpy ndarray dtypes to (field type, subtype)
DTYPE_OGR_FIELD_TYPES = {
    'int8': (OFTInteger, OFSTInt16),
    'int16': (OFTInteger, OFSTInt16),
    'int32': (OFTInteger, OFSTNone),
    'int': (OFTInteger64, OFSTNone),
    'int64': (OFTInteger64, OFSTNone),
    # unsigned ints have to be converted to ints; these are converted
    # to the next largest integer size
    'uint8': (OFTInteger, OFSTInt16),
    'uint16': (OFTInteger, OFSTNone),
    'uint32': (OFTInteger64, OFSTNone),
    # TODO: these might get truncated, check maximum value and raise error
    'uint': (OFTInteger64, OFSTNone),
    'uint64': (OFTInteger64, OFSTNone),

    # bool is handled as integer with boolean subtype
    'bool': (OFTInteger, OFSTBoolean),

    'float32': (OFTReal,OFSTFloat32),
    'float': (OFTReal, OFSTNone),
    'float64': (OFTReal, OFSTNone)
}



cdef int start_transaction(OGRDataSourceH ogr_dataset, int force) except 1:
    cdef int err = GDALDatasetStartTransaction(ogr_dataset, force)
    if err == OGRERR_FAILURE:
        raise DataSourceError("Failed to start transaction")

    return 0


cdef int commit_transaction(OGRDataSourceH ogr_dataset) except 1:
    cdef int err = GDALDatasetCommitTransaction(ogr_dataset)
    if err == OGRERR_FAILURE:
        raise DataSourceError("Failed to commit transaction")

    return 0


# Not currently used; uncomment when used
# cdef int rollback_transaction(OGRDataSourceH ogr_dataset) except 1:
#     cdef int err = GDALDatasetRollbackTransaction(ogr_dataset)
#     if err == OGRERR_FAILURE:
#         raise DataSourceError("Failed to rollback transaction")

#     return 0


# ported from fiona::_shim22.pyx::gdal_open_vector
cdef void* ogr_open(const char* path_c, int mode, options) except NULL:
    cdef void* ogr_dataset = NULL
    cdef char **ogr_drivers = NULL
    cdef void* ogr_driver = NULL
    cdef char **open_opts = NULL

    # Register all drivers
    GDALAllRegister()

    # Force linear approximations in all cases
    OGRSetNonLinearGeometriesEnabledFlag(0)

    flags = GDAL_OF_VECTOR | GDAL_OF_VERBOSE_ERROR
    if mode == 1:
        flags |= GDAL_OF_UPDATE
    else:
        flags |= GDAL_OF_READONLY

    # TODO: other open opts from fiona
    open_opts = CSLAddNameValue(open_opts, "VALIDATE_OPEN_OPTIONS", "NO")

    try:
        ogr_dataset = exc_wrap_pointer(
            GDALOpenEx(path_c, flags, <const char *const *>ogr_drivers, <const char *const *>open_opts, NULL)
        )
        return ogr_dataset

    except NullPointerError:
        raise DataSourceError("Failed to open dataset (mode={}): {}".format(mode, path_c.decode("utf-8"))) from None

    except CPLE_BaseError as exc:
        raise DataSourceError(str(exc))

    finally:
        CSLDestroy(ogr_drivers)
        CSLDestroy(open_opts)


cdef OGRLayerH get_ogr_layer(GDALDatasetH ogr_dataset, layer) except NULL:
    """Open OGR layer by index or name.

    Parameters
    ----------
    ogr_dataset : pointer to open OGR dataset
    layer : str or int
        name or index of layer

    Returns
    -------
    pointer to OGR layer
    """
    cdef OGRLayerH ogr_layer = NULL

    try:
        if isinstance(layer, str):
            name_b = layer.encode('utf-8')
            name_c = name_b
            ogr_layer = exc_wrap_pointer(GDALDatasetGetLayerByName(ogr_dataset, name_c))

        elif isinstance(layer, int):
            ogr_layer = exc_wrap_pointer(GDALDatasetGetLayer(ogr_dataset, layer))

    # GDAL does not always raise exception messages in this case
    except NullPointerError:
        raise DataLayerError(f"Layer '{layer}' could not be opened") from None

    except CPLE_BaseError as exc:
        raise DataLayerError(str(exc))

    return ogr_layer


cdef OGRLayerH execute_sql(GDALDatasetH ogr_dataset, str sql, str sql_dialect=None) except NULL:
    """Execute an SQL statement on a dataset.

    Parameters
    ----------
    ogr_dataset : pointer to open OGR dataset
    sql : str
        The sql statement to execute
    sql_dialect : str, optional (default: None)
        The sql dialect the sql statement is written in

    Returns
    -------
    pointer to OGR layer
    """

    try:
        sql_b = sql.encode('utf-8')
        sql_c = sql_b
        if sql_dialect is None:
            return exc_wrap_pointer(GDALDatasetExecuteSQL(ogr_dataset, sql_c, NULL, NULL))

        sql_dialect_b = sql_dialect.encode('utf-8')
        sql_dialect_c = sql_dialect_b
        return exc_wrap_pointer(GDALDatasetExecuteSQL(ogr_dataset, sql_c, NULL, sql_dialect_c))

    # GDAL does not always raise exception messages in this case
    except NullPointerError:
        raise DataLayerError(f"Error executing sql '{sql}'") from None

    except CPLE_BaseError as exc:
        raise DataLayerError(str(exc))


cdef str get_crs(OGRLayerH ogr_layer):
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
        # No coordinate system defined.
        # This is expected and valid for nonspatial tables.
        return None

    except CPLE_BaseError as exc:
        raise CRSError(str(exc))

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
            raise CRSError("CRS could not be extracted as WKT") from None

        wkt = get_string(ogr_wkt)

    finally:
        CPLFree(ogr_wkt)
        return wkt


cdef detect_encoding(OGRDataSourceH ogr_dataset, OGRLayerH ogr_layer):
    """Attempt to detect the encoding of the layer.
    If it supports UTF-8, use that.
    If it is a shapefile, it must otherwise be ISO-8859-1.

    Parameters
    ----------
    ogr_dataset : pointer to open OGR dataset
    ogr_layer : pointer to open OGR layer

    Returns
    -------
    str or None
    """
    cdef void *ogr_driver

    if OGR_L_TestCapability(ogr_layer, OLCStringsAsUTF8):
        return 'UTF-8'

    try:
        ogr_driver = exc_wrap_pointer(GDALGetDatasetDriver(ogr_dataset))

    except NullPointerError:
        raise DataLayerError(f"Could not detect encoding of layer") from None

    except CPLE_BaseError as exc:
        raise DataLayerError(str(exc))

    driver = OGR_Dr_GetName(ogr_driver).decode("UTF-8")
    if driver == 'ESRI Shapefile':
        return 'ISO-8859-1'

    return None


cdef get_fields(OGRLayerH ogr_layer, str encoding):
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
    cdef OGRFeatureDefnH ogr_featuredef = NULL
    cdef OGRFieldDefnH ogr_fielddef = NULL
    cdef int field_subtype
    cdef const char *key_c

    try:
        ogr_featuredef = exc_wrap_pointer(OGR_L_GetLayerDefn(ogr_layer))

    except NullPointerError:
        raise DataLayerError("Could not get layer definition") from None

    except CPLE_BaseError as exc:
        raise DataLayerError(str(exc))

    field_count = OGR_FD_GetFieldCount(ogr_featuredef)

    fields = np.empty(shape=(field_count, 4), dtype=object)
    fields_view = fields[:,:]

    for i in range(field_count):
        try:
            ogr_fielddef = exc_wrap_pointer(OGR_FD_GetFieldDefn(ogr_featuredef, i))

        except NullPointerError:
            raise FieldError(f"Could not get field definition for field at index {i}") from None

        except CPLE_BaseError as exc:
            raise FieldError(str(exc))

        field_name = get_string(OGR_Fld_GetNameRef(ogr_fielddef), encoding=encoding)

        field_type = OGR_Fld_GetType(ogr_fielddef)
        np_type = FIELD_TYPES[field_type]
        if not np_type:
            log.warning(
                f"Skipping field {field_name}: unsupported OGR type: {field_type}")
            continue

        field_subtype = OGR_Fld_GetSubType(ogr_fielddef)
        subtype = FIELD_SUBTYPES.get(field_subtype)
        if subtype is not None:
            # bool, int16, float32 dtypes
            np_type = subtype

        fields_view[i,0] = i
        fields_view[i,1] = field_type
        fields_view[i,2] = field_name
        fields_view[i,3] = np_type

    return fields


cdef apply_where_filter(OGRLayerH ogr_layer, str where):
    """Applies where filter to layer.

    WARNING: GDAL does not raise an error for GPKG when SQL query is invalid
    but instead only logs to stderr.

    Parameters
    ----------
    ogr_layer : pointer to open OGR layer
    where : str
        See http://ogdi.sourceforge.net/prop/6.2.CapabilitiesMetadata.html
        restricted_where for more information about valid expressions.

    Raises
    ------
    ValueError: if SQL query is not valid
    """

    where_b = where.encode('utf-8')
    where_c = where_b
    err = OGR_L_SetAttributeFilter(ogr_layer, where_c)
    # WARNING: GDAL does not raise this error for GPKG but instead only
    # logs to stderr
    if err != OGRERR_NONE:
        try:
            exc_check()
            name = OGR_L_GetName(ogr_layer)
        except CPLE_BaseError as exc:
            raise ValueError(str(exc))

        raise ValueError(f"Invalid SQL query for layer '{name}': '{where}'")


cdef apply_spatial_filter(OGRLayerH ogr_layer, bbox):
    """Applies spatial filter to layer.

    Parameters
    ----------
    ogr_layer : pointer to open OGR layer
    bbox: list or tuple of xmin, ymin, xmax, ymax

    Raises
    ------
    ValueError: if bbox is not a list or tuple or does not have proper number of
        items
    """

    if not (isinstance(bbox, (tuple, list)) and len(bbox) == 4):
        raise ValueError(f"Invalid bbox: {bbox}")

    xmin, ymin, xmax, ymax = bbox
    OGR_L_SetSpatialFilterRect(ogr_layer, xmin, ymin, xmax, ymax)


cdef validate_feature_range(OGRLayerH ogr_layer, int skip_features=0, int max_features=0):
    """Limit skip_features and max_features to bounds available for dataset.

    This is typically performed after applying where and spatial filters, which
    reduce the available range of features.

    Parameters
    ----------
    ogr_layer : pointer to open OGR layer
    skip_features : number of features to skip from beginning of available range
    max_features : number of features to read from available range
    """
    feature_count = OGR_L_GetFeatureCount(ogr_layer, 1)
    if feature_count <= 0:
        # the count comes back as -1 if the where clause above is invalid but not rejected as error
        name = OGR_L_GetName(ogr_layer)
        warnings.warn(f"Layer '{name}' does not have any features to read")
        feature_count = 0
        skip_features = 0
        max_features = 0

    else:
        # validate skip_features, max_features
        if skip_features < 0 or skip_features >= feature_count:
            raise ValueError(f"'skip_features' must be between 0 and {feature_count-1}")

        if max_features < 0:
            raise ValueError("'max_features' must be >= 0")

        if max_features > feature_count:
            max_features = feature_count

    return skip_features, max_features


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef process_geometry(OGRFeatureH ogr_feature, int i, geom_view, uint8_t force_2d):

    cdef OGRGeometryH ogr_geometry = NULL
    cdef unsigned char *wkb = NULL
    cdef int ret_length

    ogr_geometry = OGR_F_GetGeometryRef(ogr_feature)

    if ogr_geometry == NULL:
        geom_view[i] = None
    else:
        try:
            # if geometry has M values, these need to be removed first
            if (OGR_G_IsMeasured(ogr_geometry)):
                OGR_G_SetMeasured(ogr_geometry, 0)

            if force_2d and OGR_G_Is3D(ogr_geometry):
                OGR_G_Set3D(ogr_geometry, 0)

            ret_length = OGR_G_WkbSize(ogr_geometry)
            wkb = <unsigned char*>malloc(sizeof(unsigned char)*ret_length)
            OGR_G_ExportToWkb(ogr_geometry, 1, wkb)
            geom_view[i] = wkb[:ret_length]

        finally:
            free(wkb)


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef process_fields(
    OGRFeatureH ogr_feature,
    int i,
    int n_fields,
    object field_data,
    object field_data_view,
    object field_indexes,
    object field_ogr_types,
    encoding
):
    cdef int j
    cdef int success
    cdef int field_index
    cdef int ret_length
    cdef GByte *bin_value
    cdef int year = 0
    cdef int month = 0
    cdef int day = 0
    cdef int hour = 0
    cdef int minute = 0
    cdef int second = 0
    cdef int timezone = 0

    for j in range(n_fields):
        field_index = field_indexes[j]
        field_type = field_ogr_types[j]
        data = field_data_view[j]

        isnull = OGR_F_IsFieldSetAndNotNull(ogr_feature, field_index) == 0
        if isnull:
            if field_type in (OFTInteger, OFTInteger64, OFTReal):
                # if a boolean or integer type, have to cast to float to hold
                # NaN values
                if data.dtype.kind in ('b', 'i', 'u'):
                    field_data[j] = field_data[j].astype(np.float64)
                    field_data_view[j] = field_data[j][:]
                    field_data_view[j][i] = np.nan
                else:
                    data[i] = np.nan

            elif field_type in ( OFTDate, OFTDateTime):
                data[i] = np.datetime64('NaT')

            else:
                data[i] = None

            continue

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
                data[i] = np.datetime64('NaT')

            elif field_type == OFTDate:
                data[i] = datetime.date(year, month, day).isoformat()

            elif field_type == OFTDateTime:
                data[i] = datetime.datetime(year, month, day, hour, minute, second).isoformat()


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef get_features(
    OGRLayerH ogr_layer,
    object[:,:] fields,
    encoding,
    uint8_t read_geometry,
    uint8_t force_2d,
    int skip_features,
    int max_features,
    uint8_t return_fids
):

    cdef OGRFeatureH ogr_feature = NULL
    cdef int n_fields
    cdef int i
    cdef int field_index
    cdef int count

    # make sure layer is read from beginning
    OGR_L_ResetReading(ogr_layer)

    count = OGR_L_GetFeatureCount(ogr_layer, 1)
    if count < 0:
        # sometimes this comes back as -1 if there is an error with the where clause, etc
        count = 0

    if skip_features > 0:
        count = count - skip_features
        OGR_L_SetNextByIndex(ogr_layer, skip_features)

    if max_features > 0:
        count = max_features

    if return_fids:
        fid_data = np.empty(shape=(count), dtype=np.int64)
        fid_view = fid_data[:]
    else:
        fid_data = None

    if read_geometry:
        geometries = np.empty(shape=(count, ), dtype='object')
        geom_view = geometries[:]

    else:
        geometries = None

    n_fields = fields.shape[0]
    field_indexes = fields[:,0]
    field_ogr_types = fields[:,1]

    field_data = [
        np.empty(shape=(count, ),
        dtype=fields[field_index,3]) for field_index in range(n_fields)
    ]

    field_data_view = [field_data[field_index][:] for field_index in range(n_fields)]

    for i in range(count):
        try:
            ogr_feature = exc_wrap_pointer(OGR_L_GetNextFeature(ogr_layer))

        except NullPointerError:
            raise FeatureError(f"Could not read feature at index {i}") from None

        except CPLE_BaseError as exc:
            raise FeatureError(str(exc))

        if return_fids:
            fid_view[i] = OGR_F_GetFID(ogr_feature)

        if read_geometry:
            process_geometry(ogr_feature, i, geom_view, force_2d)

        process_fields(
            ogr_feature, i, n_fields, field_data, field_data_view,
            field_indexes, field_ogr_types, encoding
        )

    return fid_data, geometries, field_data


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef get_features_by_fid(
    OGRLayerH ogr_layer,
    int[:] fids,
    object[:,:] fields,
    encoding,
    uint8_t read_geometry,
    uint8_t force_2d
):

    cdef OGRFeatureH ogr_feature = NULL
    cdef int n_fields
    cdef int i
    cdef int fid
    cdef int field_index
    cdef int count

    # make sure layer is read from beginning
    OGR_L_ResetReading(ogr_layer)

    count = len(fids)

    if read_geometry:
        geometries = np.empty(shape=(count, ), dtype='object')
        geom_view = geometries[:]

    else:
        geometries = None

    n_fields = fields.shape[0]
    field_indexes = fields[:,0]
    field_ogr_types = fields[:,1]

    field_data = [
        np.empty(shape=(count, ),
        dtype=fields[field_index,3]) for field_index in range(n_fields)
    ]

    field_data_view = [field_data[field_index][:] for field_index in range(n_fields)]

    for i in range(count):
        fid = fids[i]

        try:
            ogr_feature = exc_wrap_pointer(OGR_L_GetFeature(ogr_layer, fid))

        except NullPointerError:
            raise FeatureError(f"Could not read feature with fid {fid}") from None

        except CPLE_BaseError as exc:
            raise FeatureError(str(exc))

        if read_geometry:
            process_geometry(ogr_feature, i, geom_view, force_2d)

        process_fields(
            ogr_feature, i, n_fields, field_data, field_data_view,
            field_indexes, field_ogr_types, encoding
        )

    return (geometries, field_data)


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef get_bounds(
    OGRLayerH ogr_layer,
    int skip_features,
    int max_features):

    cdef OGRFeatureH ogr_feature = NULL
    cdef OGRGeometryH ogr_geometry = NULL
    cdef OGREnvelope ogr_envelope # = NULL
    cdef int i
    cdef int count

    # make sure layer is read from beginning
    OGR_L_ResetReading(ogr_layer)

    count = OGR_L_GetFeatureCount(ogr_layer, 1)
    if count < 0:
        # sometimes this comes back as -1 if there is an error with the where clause, etc
        count = 0

    if skip_features > 0:
        count = count - skip_features
        OGR_L_SetNextByIndex(ogr_layer, skip_features)

    if max_features > 0:
        count = max_features

    fid_data = np.empty(shape=(count), dtype=np.int64)
    fid_view = fid_data[:]

    bounds_data = np.empty(shape=(4, count), dtype='float64')
    bounds_view = bounds_data[:]

    for i in range(count):
        try:
            ogr_feature = exc_wrap_pointer(OGR_L_GetNextFeature(ogr_layer))

        except NullPointerError:
            raise FeatureError(f"Could not read feature at index {i}") from None

        except CPLE_BaseError as exc:
            raise FeatureError(str(exc))

        fid_view[i] = OGR_F_GetFID(ogr_feature)

        ogr_geometry = OGR_F_GetGeometryRef(ogr_feature)

        if ogr_geometry == NULL:
            bounds_view[:,i] = np.nan

        else:
            OGR_G_GetEnvelope(ogr_geometry, &ogr_envelope)
            bounds_view[0, i] = ogr_envelope.MinX
            bounds_view[1, i] = ogr_envelope.MinY
            bounds_view[2, i] = ogr_envelope.MaxX
            bounds_view[3, i] = ogr_envelope.MaxY

    return fid_data, bounds_data


def ogr_read(
    str path,
    object layer=None,
    object encoding=None,
    int read_geometry=True,
    int force_2d=False,
    object columns=None,
    int skip_features=0,
    int max_features=0,
    object where=None,
    tuple bbox=None,
    object fids=None,
    str sql=None,
    str sql_dialect=None,
    int return_fids=False,
    **kwargs):

    cdef int err = 0
    cdef const char *path_c = NULL
    cdef const char *where_c = NULL
    cdef OGRDataSourceH ogr_dataset = NULL
    cdef OGRLayerH ogr_layer = NULL
    cdef int feature_count = 0
    cdef double xmin, ymin, xmax, ymax

    path_b = path.encode('utf-8')
    path_c = path_b

    if fids is not None:
        if where is not None or bbox is not None or sql is not None or skip_features or max_features:
            raise ValueError(
                "cannot set both 'fids' and any of 'where', 'bbox', 'sql', "
                "'skip_features' or 'max_features'"
            )
        fids = np.asarray(fids, dtype=np.intc)

    if sql is not None and layer is not None:
        raise ValueError("'sql' paramater cannot be combined with 'layer'")

    ogr_dataset = ogr_open(path_c, 0, kwargs)
    try:
        if sql is None:
            # layer defaults to index 0
            if layer is None:
                layer = 0
            ogr_layer = get_ogr_layer(ogr_dataset, layer)
        else:
            ogr_layer = execute_sql(ogr_dataset, sql, sql_dialect)

        crs = get_crs(ogr_layer)

        # Encoding is derived from the dataset, from the user, or from the system locale
        encoding = (
            detect_encoding(ogr_dataset, ogr_layer)
            or encoding
            or locale.getpreferredencoding()
        )

        fields = get_fields(ogr_layer, encoding)

        if columns is not None:
            # Fields are matched exactly by name, duplicates are dropped.
            # Find index of each field into fields
            idx = np.intersect1d(fields[:,2], columns, return_indices=True)[1]
            fields = fields[idx, :]

        geometry_type = get_geometry_type(ogr_layer)

        if fids is not None:
            geometries, field_data = get_features_by_fid(
                ogr_layer,
                fids,
                fields,
                encoding,
                read_geometry=read_geometry and geometry_type is not None,
                force_2d=force_2d,
            )

            # bypass reading fids since these should match fids used for read
            if return_fids:
                fid_data = fids.astype(np.int64)
            else:
                fid_data = None
        else:
            # Apply the attribute filter
            if where is not None and where != "":
                apply_where_filter(ogr_layer, where)

            # Apply the spatial filter
            if bbox is not None:
                apply_spatial_filter(ogr_layer, bbox)

            # Limit feature range to available range
            skip_features, max_features = validate_feature_range(
                ogr_layer, skip_features, max_features
            )

            fid_data, geometries, field_data = get_features(
                ogr_layer,
                fields,
                encoding,
                read_geometry=read_geometry and geometry_type is not None,
                force_2d=force_2d,
                skip_features=skip_features,
                max_features=max_features,
                return_fids=return_fids
            )

        meta = {
            'crs': crs,
            'encoding': encoding,
            'fields': fields[:,2], # return only names
            'geometry_type': geometry_type
        }

    finally:
        if ogr_dataset != NULL:
            if sql is not None:
                GDALDatasetReleaseResultSet(ogr_dataset, ogr_layer)

            GDALClose(ogr_dataset)
            ogr_dataset = NULL

    return (
        meta,
        fid_data,
        geometries,
        field_data
    )


def ogr_read_bounds(
    str path,
    object layer=None,
    object encoding=None,
    int read_geometry=True,
    int force_2d=False,
    object columns=None,
    int skip_features=0,
    int max_features=0,
    object where=None,
    tuple bbox=None,
    **kwargs):

    cdef int err = 0
    cdef const char *path_c = NULL
    cdef const char *where_c = NULL
    cdef OGRDataSourceH ogr_dataset = NULL
    cdef OGRLayerH ogr_layer = NULL
    cdef int feature_count = 0
    cdef double xmin, ymin, xmax, ymax

    path_b = path.encode('utf-8')
    path_c = path_b

    # layer defaults to index 0
    if layer is None:
        layer = 0

    ogr_dataset = ogr_open(path_c, 0, kwargs)
    ogr_layer = get_ogr_layer(ogr_dataset, layer)

    # Apply the attribute filter
    if where is not None and where != "":
        apply_where_filter(ogr_layer, where)

    # Apply the spatial filter
    if bbox is not None:
        apply_spatial_filter(ogr_layer, bbox)

    # Limit feature range to available range
    skip_features, max_features = validate_feature_range(ogr_layer, skip_features, max_features)

    return get_bounds(ogr_layer, skip_features, max_features)


def ogr_read_info(str path, object layer=None, object encoding=None, **kwargs):
    cdef const char *path_c = NULL
    cdef OGRDataSourceH ogr_dataset = NULL
    cdef OGRLayerH ogr_layer = NULL

    path_b = path.encode('utf-8')
    path_c = path_b

    # layer defaults to index 0
    if layer is None:
        layer = 0

    ogr_dataset = ogr_open(path_c, 0, kwargs)
    ogr_layer = get_ogr_layer(ogr_dataset, layer)

    # Encoding is derived from the dataset, from the user, or from the system locale
    encoding = (
        detect_encoding(ogr_dataset, ogr_layer)
        or encoding
        or locale.getpreferredencoding()
    )

    fields = get_fields(ogr_layer, encoding)

    meta = {
        'crs': get_crs(ogr_layer),
        'encoding': encoding,
        'fields': fields[:,2], # return only names
        'dtypes': fields[:,3],
        'geometry_type': get_geometry_type(ogr_layer),
        'features': OGR_L_GetFeatureCount(ogr_layer, 1),
        "capabilities": {
            "random_read": OGR_L_TestCapability(ogr_layer, OLCRandomRead),
            "fast_set_next_by_index": OGR_L_TestCapability(ogr_layer, OLCFastSetNextByIndex),
            "fast_spatial_filter": OGR_L_TestCapability(ogr_layer, OLCFastSpatialFilter),
        }
    }

    if ogr_dataset != NULL:
        GDALClose(ogr_dataset)
    ogr_dataset = NULL

    return meta


def ogr_list_layers(str path):
    cdef const char *path_c = NULL
    cdef const char *ogr_name = NULL
    cdef OGRDataSourceH ogr_dataset = NULL
    cdef OGRLayerH ogr_layer = NULL

    path_b = path.encode('utf-8')
    path_c = path_b

    ogr_dataset = ogr_open(path_c, 0, None)

    layer_count = GDALDatasetGetLayerCount(ogr_dataset)

    data = np.empty(shape=(layer_count, 2), dtype=object)
    data_view = data[:]
    for i in range(layer_count):
        ogr_layer = GDALDatasetGetLayer(ogr_dataset, i)

        data_view[i, 0] = get_string(OGR_L_GetName(ogr_layer))
        data_view[i, 1] = get_geometry_type(ogr_layer)

    if ogr_dataset != NULL:
        GDALClose(ogr_dataset)
    ogr_dataset = NULL

    return data


# NOTE: all modes are write-only
# some data sources have multiple layers
cdef void * ogr_create(const char* path_c, const char* driver_c) except NULL:
    cdef void *ogr_driver = NULL
    cdef OGRDataSourceH ogr_dataset = NULL

    # Get the driver
    try:
        ogr_driver = exc_wrap_pointer(GDALGetDriverByName(driver_c))

    except NullPointerError:
        raise DataSourceError(f"Could not obtain driver: {driver_c.decode('utf-8')} (check that it was installed correctly into GDAL)")

    except CPLE_BaseError as exc:
        raise DataSourceError(str(exc))

    # Create the dataset
    try:
        ogr_dataset = exc_wrap_pointer(GDALCreate(ogr_driver, path_c, 0, 0, 0, GDT_Unknown, NULL))

    except NullPointerError:
        raise DataSourceError(f"Failed to create dataset with driver: {path_c.decode('utf-8')} {driver_c.decode('utf-8')}") from None

    except CPLE_NotSupportedError as exc:
        raise DataSourceError(f"Driver {driver_c.decode('utf-8')} does not support write functionality") from None

    except CPLE_BaseError as exc:
        raise DataSourceError(str(exc))

    return ogr_dataset


cdef void * create_crs(str crs) except NULL:
    cdef char *crs_c = NULL
    cdef void *ogr_crs = NULL

    crs_b = crs.encode('UTF-8')
    crs_c = crs_b

    try:
        ogr_crs = exc_wrap_pointer(OSRNewSpatialReference(NULL))
        err = OSRSetFromUserInput(ogr_crs, crs_c)
        if err:
            raise CRSError("Could not set CRS: {}".format(crs_c.decode('UTF-8'))) from None

    except CPLE_BaseError as exc:
        OSRRelease(ogr_crs)
        raise CRSError("Could not set CRS: {}".format(exc))

    return ogr_crs


cdef infer_field_types(list dtypes):
    cdef int field_type = 0
    cdef int field_subtype = 0
    cdef int width = 0
    cdef int precision = 0

    field_types = np.zeros(shape=(len(dtypes), 4), dtype=int)
    field_types_view = field_types[:]

    for i in range(len(dtypes)):
        dtype = dtypes[i]

        if dtype.name in DTYPE_OGR_FIELD_TYPES:
            field_type, field_subtype = DTYPE_OGR_FIELD_TYPES[dtype.name]
            field_types_view[i, 0] = field_type
            field_types_view[i, 1] = field_subtype

        # Determine field type from ndarray values
        elif dtype == np.dtype('O'):
            # Object type is ambiguous: could be a string or binary data
            # TODO: handle binary or other types
            # for now fall back to string (same as Geopandas)
            field_types_view[i, 0] = OFTString
            # Convert to unicode string then take itemsize
            # TODO: better implementation of this
            # width = values.astype(np.unicode_).dtype.itemsize // 4
            # DO WE NEED WIDTH HERE?

        elif dtype.type is np.unicode_ or dtype.type is np.string_:
            field_types_view[i, 0] = OFTString
            field_types_view[i, 2] = int(dtype.itemsize // 4)

        # TODO: datetime types

        else:
            raise NotImplementedError(f"field type is not supported {dtype.name} (field index: {i})")

    return field_types


# TODO: handle updateable data sources, like GPKG
# TODO: set geometry and field data as memory views?
def ogr_write(str path, str layer, str driver, geometry, field_data, fields,
    str crs, str geometry_type, str encoding, bint promote_to_multi=False, **kwargs):

    cdef const char *path_c = NULL
    cdef const char *layer_c = NULL
    cdef const char *driver_c = NULL
    cdef const char *crs_c = NULL
    cdef const char *encoding_c = NULL
    cdef char **options = NULL
    cdef const char *ogr_name = NULL
    cdef OGRDataSourceH ogr_dataset = NULL
    cdef OGRLayerH ogr_layer = NULL
    cdef OGRFeatureH ogr_feature = NULL
    cdef OGRGeometryH ogr_geometry = NULL
    cdef OGRGeometryH ogr_geometry_multi = NULL
    cdef OGRFeatureDefnH ogr_featuredef = NULL
    cdef OGRFieldDefnH ogr_fielddef = NULL
    cdef unsigned char *wkb_buffer = NULL
    cdef OGRSpatialReferenceH ogr_crs = NULL
    cdef int layer_idx = -1
    cdef int geometry_code
    cdef int err = 0
    cdef int i = 0
    cdef int num_records = len(geometry)
    cdef int num_fields = len(field_data) if field_data else 0

    if len(field_data) != len(fields):
        raise ValueError("field_data and fields must be same length")

    if num_fields:
        for i in range(1, len(field_data)):
            if len(field_data[i]) != num_records:
                raise ValueError("field_data arrays must be same length as geometry array")

    path_b = path.encode('UTF-8')
    path_c = path_b

    driver_b = driver.encode('UTF-8')
    driver_c = driver_b

    if not layer:
        layer = os.path.splitext(os.path.split(path)[1])[0]

    layer_b = layer.encode('UTF-8')
    layer_c = layer_b

    # if shapefile, GeoJSON, or FlatGeobuf, always delete first
    # for other types, check if we can create layers
    # GPKG might be the only multi-layer writeable type.  TODO: check this
    if driver in ('ESRI Shapefile', 'GeoJSON', 'GeoJSONSeq', 'FlatGeobuf') and os.path.exists(path):
        os.unlink(path)

    # TODO: invert this: if exists then try to update it, if that doesn't work then always create
    if os.path.exists(path):
        try:
            ogr_dataset = ogr_open(path_c, 1, None)

            # If layer exists, delete it.
            for i in range(GDALDatasetGetLayerCount(ogr_dataset)):
                name = OGR_L_GetName(GDALDatasetGetLayer(ogr_dataset, i))
                if layer == name.decode('UTF-8'):
                    layer_idx = i
                    break

            if layer_idx >= 0:
                GDALDatasetDeleteLayer(ogr_dataset, layer_idx)

        except DataSourceError:
            # open failed, so create from scratch
            # force delete it first
            os.unlink(path)
            ogr_dataset = NULL

    # either it didn't exist or could not open it in write mode
    if ogr_dataset == NULL:
        ogr_dataset = ogr_create(path_c, driver_c)

    ### Create the CRS
    if crs is not None:
        try:
            ogr_crs = create_crs(crs)

        except Exception as exc:
            OGRReleaseDataSource(ogr_dataset)
            raise exc


    ### Create options
    if not encoding:
        encoding = locale.getpreferredencoding()

    if driver == 'ESRI Shapefile':
        # Fiona only sets encoding for shapefiles; other drivers do not support
        # encoding as an option.
        encoding_b = encoding.upper().encode('UTF-8')
        encoding_c = encoding_b
        options = CSLSetNameValue(options, "ENCODING", encoding_c)

    # Setup other layer creation options
    for k, v in kwargs.items():
        if v is None:
            continue

        k = k.upper().encode('UTF-8')

        if isinstance(v, bool):
            v = ('ON' if v else 'OFF').encode('utf-8')
        else:
            v = str(v).encode('utf-8')

        options = CSLAddNameValue(options, <const char *>k, <const char *>v)


    ### Get geometry type
    # TODO: this is brittle for 3D / ZM / M types
    # TODO: fail on M / ZM types
    geometry_code = get_geometry_type_code(geometry_type or "Unknown")

    ### Create the layer
    try:
        ogr_layer = exc_wrap_pointer(
                GDALDatasetCreateLayer(ogr_dataset, layer_c, ogr_crs,
                        <OGRwkbGeometryType>geometry_code, options))

    except Exception as exc:
        OGRReleaseDataSource(ogr_dataset)
        ogr_dataset = NULL
        raise DataLayerError(str(exc))

    finally:
        if ogr_crs != NULL:
            OSRRelease(ogr_crs)
            ogr_crs = NULL

        if options != NULL:
            CSLDestroy(<char**>options)
            options = NULL

    ### Create the fields
    field_types = infer_field_types([field.dtype for field in field_data])
    for i in range(num_fields):
        field_type, field_subtype, width, precision = field_types[i]

        name_b = fields[i].encode(encoding)
        try:
            ogr_fielddef = exc_wrap_pointer(OGR_Fld_Create(name_b, field_type))

            # subtypes, see: https://gdal.org/development/rfc/rfc50_ogr_field_subtype.html
            if field_subtype != OFSTNone:
                OGR_Fld_SetSubType(ogr_fielddef, field_subtype)

            if width:
                OGR_Fld_SetWidth(ogr_fielddef, width)

            # TODO: set precision

        except:
            if ogr_fielddef != NULL:
                OGR_Fld_Destroy(ogr_fielddef)
                ogr_fielddef = NULL

            OGRReleaseDataSource(ogr_dataset)
            ogr_dataset = NULL
            raise FieldError(f"Error creating field '{fields[i]}' from field_data") from None

        try:
            exc_wrap_int(OGR_L_CreateField(ogr_layer, ogr_fielddef, 1))

        except:
            OGRReleaseDataSource(ogr_dataset)
            ogr_dataset = NULL
            raise FieldError(f"Error adding field '{fields[i]}' to layer") from None

        finally:
            if ogr_fielddef != NULL:
                OGR_Fld_Destroy(ogr_fielddef)


    ### Create the features
    ogr_featuredef = OGR_L_GetLayerDefn(ogr_layer)

    start_transaction(ogr_dataset, 0)
    for i in range(num_records):
        try:
            # create the feature
            ogr_feature = OGR_F_Create(ogr_featuredef)
            if ogr_feature == NULL:
                raise FeatureError(f"Could not create feature at index {i}") from None

            # create the geometry based on specific WKB type (there might be mixed types in geometries)
            # TODO: geometry must not be null or errors
            wkb = geometry[i]
            wkbtype = bytearray(wkb)[1]
            # may need to consider all 4 bytes: int.from_bytes(wkb[0][1:4], byteorder="little")
            # use "little" if the first byte == 1
            ogr_geometry = OGR_G_CreateGeometry(<OGRwkbGeometryType>wkbtype)
            if ogr_geometry == NULL:
                raise GeometryError(f"Could not create geometry at index {i} for WKB type {wkbtype}") from None

            # import the WKB
            wkb_buffer = wkb
            err = OGR_G_ImportFromWkb(ogr_geometry, wkb_buffer, len(wkb))
            if err:
                if ogr_geometry != NULL:
                    OGR_G_DestroyGeometry(ogr_geometry)
                    ogr_geometry = NULL
                raise GeometryError(f"Could not create geometry from WKB at index {i}") from None

            # Convert to multi type
            if promote_to_multi:
                if wkbtype in (wkbPoint, wkbPoint25D, wkbPointM, wkbPointZM):
                    ogr_geometry = OGR_G_ForceToMultiPoint(ogr_geometry)
                elif wkbtype in (wkbLineString, wkbLineString25D, wkbLineStringM, wkbLineStringZM):
                    ogr_geometry = OGR_G_ForceToMultiLineString(ogr_geometry)
                elif wkbtype in (wkbPolygon, wkbPolygon25D, wkbPolygonM, wkbPolygonZM):
                    ogr_geometry = OGR_G_ForceToMultiPolygon(ogr_geometry)

            # Set the geometry on the feature
            # this assumes ownership of the geometry and it's cleanup
            err = OGR_F_SetGeometryDirectly(ogr_feature, ogr_geometry)
            if err:
                raise GeometryError(f"Could not set geometry for feature at index {i}") from None

            # Set field values
            for field_idx in range(num_fields):
                field_value = field_data[field_idx][i]
                field_type = field_types[field_idx][0]

                if field_value is None:
                    OGR_F_SetFieldNull(ogr_feature, field_idx)

                elif field_type == OFTString:
                    # TODO: encode string using approach from _get_internal_encoding which checks layer capabilities
                    try:
                        # this will fail for strings mixed with nans
                        value_b = field_value.encode("UTF-8")

                    except AttributeError:
                        raise ValueError(f"Could not encode value '{field_value}' in field '{fields[field_idx]}' to string")

                    except Exception:
                        raise

                    OGR_F_SetFieldString(ogr_feature, field_idx, value_b)

                elif field_type == OFTInteger:
                    OGR_F_SetFieldInteger(ogr_feature, field_idx, field_value)

                elif field_type == OFTInteger64:
                    OGR_F_SetFieldInteger64(ogr_feature, field_idx, field_value)

                elif field_type == OFTReal:
                    OGR_F_SetFieldDouble(ogr_feature, field_idx, field_value)

                else:
                    raise NotImplementedError(f"OGR field type is not supported for writing: {field_type}")


            # Add feature to the layer
            try:
                exc_wrap_int(OGR_L_CreateFeature(ogr_layer, ogr_feature))
            except CPLE_BaseError as exc:
                raise FeatureError(f"Could not add feature to layer at index {i}: {exc}") from None

        finally:
            if ogr_feature != NULL:
                OGR_F_Destroy(ogr_feature)
                ogr_feature = NULL

    commit_transaction(ogr_dataset)

    log.info(f"Created {num_records:,} records" )

    ### Final cleanup
    if ogr_dataset != NULL:
        GDALClose(ogr_dataset)
