from pyogrio._ogr cimport *
from pyogrio._err cimport *
from pyogrio.errors import UnsupportedGeometryTypeError


# Mapping of OGR integer geometry types to GeoJSON type names.
# GEOMETRY_TYPES = {
#     0: 'Unknown',
#     1: 'Point',
#     2: 'LineString',
#     3: 'Polygon',
#     4: 'MultiPoint',
#     5: 'MultiLineString',
#     6: 'MultiPolygon',
#     7: 'GeometryCollection',
#     # Unsupported types.
#     #8: 'CircularString',
#     #9: 'CompoundCurve',
#     #10: 'CurvePolygon',
#     #11: 'MultiCurve',
#     #12: 'MultiSurface',
#     #13: 'Curve',
#     #14: 'Surface',
#     #15: 'PolyhedralSurface',
#     #16: 'TIN',
#     #17: 'Triangle',
#     100: None,
#     101: 'LinearRing',
#     0x80000001: '3D Point',
#     0x80000002: '3D LineString',
#     0x80000003: '3D Polygon',
#     0x80000004: '3D MultiPoint',
#     0x80000005: '3D MultiLineString',
#     0x80000006: '3D MultiPolygon',
#     0x80000007: '3D GeometryCollection'
# }

GEOMETRY_TYPES = {
    wkbUnknown: 'Unknown',
    wkbPoint: 'Point',
    wkbLineString: 'LineString',
    wkbPolygon: 'Polygon',
    wkbMultiPoint: 'MultiPoint',
    wkbMultiLineString: 'MultiLineString',
    wkbMultiPolygon: 'MultiPolygon',
    wkbGeometryCollection: 'GeometryCollection',
    wkbNone: None,
    wkbLinearRing: 'LinearRing',
    wkbPointM: 'PointM',
    wkbLineStringM: 'Measured LineString',
    wkbPolygonM: 'Measured Polygon',
    wkbMultiPointM: 'Measured MultiPoint',
    wkbMultiLineStringM: 'Measured MultiLineString',
    wkbMultiPolygonM: 'Measured MultiPolygon',
    wkbGeometryCollectionM: 'Measured GeometryCollection',
    wkbPointZM: 'Measured 3D Point',
    wkbLineStringZM: 'Measured 3D LineString',
    wkbPolygonZM: 'Measured 3D Polygon',
    wkbMultiPointZM: 'Measured 3D MultiPoint',
    wkbMultiLineStringZM: 'Measured 3D MultiLineString',
    wkbMultiPolygonZM: 'Measured 3D MultiPolygon',
    wkbGeometryCollectionZM: 'Measured 3D GeometryCollection',
    wkbPoint25D: '2.5D Point',
    wkbLineString25D: '2.5D LineString',
    wkbPolygon25D: '2.5D Polygon',
    wkbMultiPoint25D: '2.5D MultiPoint',
    wkbMultiLineString25D: '2.5D MultiLineString',
    wkbMultiPolygon25D: '2.5D MultiPolygon',
    wkbGeometryCollection25D: '2.5D GeometryCollection',

    # 2.5D also represented using negative numbers not enumerated above
    -2147483647: '2.5D Point',
    -2147483646: '2.5D LineString',
    -2147483645: '2.5D Polygon',
    -2147483644: '2.5D MultiPoint',
    -2147483643: '2.5D MultiLineString',
    -2147483642: '2.5D MultiPolygon',
    -2147483641: '2.5D GeometryCollection',
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

    if ogr_type not in GEOMETRY_TYPES:
        raise UnsupportedGeometryTypeError(ogr_type)

    return GEOMETRY_TYPES[ogr_type]
