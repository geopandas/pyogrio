from pyogrio._ogr cimport *
from pyogrio._err cimport *
from pyogrio.errors import UnsupportedGeometryTypeError


# Mapping of OGR integer geometry types to GeoJSON type names.
GEOMETRY_TYPES = {
    0: 'Unknown',
    1: 'Point',
    2: 'LineString',
    3: 'Polygon',
    4: 'MultiPoint',
    5: 'MultiLineString',
    6: 'MultiPolygon',
    7: 'GeometryCollection',
    # Unsupported types.
    #8: 'CircularString',
    #9: 'CompoundCurve',
    #10: 'CurvePolygon',
    #11: 'MultiCurve',
    #12: 'MultiSurface',
    #13: 'Curve',
    #14: 'Surface',
    #15: 'PolyhedralSurface',
    #16: 'TIN',
    #17: 'Triangle',
    100: None,
    101: 'LinearRing',
    0x80000001: '3D Point',
    0x80000002: '3D LineString',
    0x80000003: '3D Polygon',
    0x80000004: '3D MultiPoint',
    0x80000005: '3D MultiLineString',
    0x80000006: '3D MultiPolygon',
    0x80000007: '3D GeometryCollection'
}



cdef str get_geometry_type(void *ogr_layer):
    """Get geometry type for layer.

    Parameters
    ----------
    ogr_layer : pointer to open OGR layer

    Returns
    -------
    str
        geometry type
    """
    cdef void *cogr_featuredef = NULL
    cdef int ogr_type

    ogr_featuredef = OGR_L_GetLayerDefn(ogr_layer)
    if ogr_featuredef == NULL:
        raise ValueError("Null feature definition")

    ogr_type = OGR_FD_GetGeomType(ogr_featuredef)

    # Normalize 'M' types to 2D types.
    if 2000 <= ogr_type < 3000:
        ogr_type = ogr_type % 1000
    elif ogr_type == 3000:
        ogr_type = 0
    # Normalize 'ZM' types to 3D types.
    elif 3000 < ogr_type < 4000:
        ogr_type = (ogr_type % 1000) | 0x80000000

    if ogr_type not in GEOMETRY_TYPES:
        raise UnsupportedGeometryTypeError(ogr_type)

    return GEOMETRY_TYPES[ogr_type]
