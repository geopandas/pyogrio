# from fiona::ogrext3.pxd

cdef extern from "ogr_core.h":

    ctypedef int OGRErr

    ctypedef enum OGRwkbGeometryType:
        wkbUnknown
        wkbPoint
        wkbLineString
        wkbPolygon
        wkbMultiPoint
        wkbMultiLineString
        wkbMultiPolygon
        wkbGeometryCollection
        wkbCircularString
        wkbCompoundCurve
        wkbCurvePolygon
        wkbMultiCurve
        wkbMultiSurface
        wkbCurve
        wkbSurface
        wkbPolyhedralSurface
        wkbTIN
        wkbTriangle
        wkbNone
        wkbLinearRing
        wkbCircularStringZ
        wkbCompoundCurveZ
        wkbCurvePolygonZ
        wkbMultiCurveZ
        wkbMultiSurfaceZ
        wkbCurveZ
        wkbSurfaceZ
        wkbPolyhedralSurfaceZ
        wkbTINZ
        wkbTriangleZ
        wkbPointM
        wkbLineStringM
        wkbPolygonM
        wkbMultiPointM
        wkbMultiLineStringM
        wkbMultiPolygonM
        wkbGeometryCollectionM
        wkbCircularStringM
        wkbCompoundCurveM
        wkbCurvePolygonM
        wkbMultiCurveM
        wkbMultiSurfaceM
        wkbCurveM
        wkbSurfaceM
        wkbPolyhedralSurfaceM
        wkbTINM
        wkbTriangleM
        wkbPointZM
        wkbLineStringZM
        wkbPolygonZM
        wkbMultiPointZM
        wkbMultiLineStringZM
        wkbMultiPolygonZM
        wkbGeometryCollectionZM
        wkbCircularStringZM
        wkbCompoundCurveZM
        wkbCurvePolygonZM
        wkbMultiCurveZM
        wkbMultiSurfaceZM
        wkbCurveZM
        wkbSurfaceZM
        wkbPolyhedralSurfaceZM
        wkbTINZM
        wkbTriangleZM
        wkbPoint25D
        wkbLineString25D
        wkbPolygon25D
        wkbMultiPoint25D
        wkbMultiLineString25D
        wkbMultiPolygon25D
        wkbGeometryCollection25D

    ctypedef enum OGRFieldType:
        OFTInteger
        OFTIntegerList
        OFTReal
        OFTRealList
        OFTString
        OFTStringList
        OFTWideString
        OFTWideStringList
        OFTBinary
        OFTDate
        OFTTime
        OFTDateTime
        OFTInteger64
        OFTInteger64List
        OFTMaxType



cdef extern from "ogr_api.h":
    void *  OGR_Dr_Open (void *driver, const char *path, int bupdate)
    const char * OGR_Dr_GetName (void *driver)

    long    OGR_F_GetFID (void *feature)
    void *  OGR_F_GetGeometryRef (void *feature)

    void *  OGR_FD_Create (char *name)
    int     OGR_FD_GetFieldCount (void *featuredefn)
    void *  OGR_FD_GetFieldDefn (void *featuredefn, int n)
    int     OGR_FD_GetGeomType (void *featuredefn)

    char *  OGR_Fld_GetNameRef (void *fielddefn)
    int     OGR_Fld_GetPrecision (void *fielddefn)
    int     OGR_Fld_GetType (void *fielddefn)
    int     OGR_Fld_GetWidth (void *fielddefn)

    void    OGR_G_ExportToWkb (void *geometry, int endianness, char *buffer)
    int     OGR_G_WkbSize (void *geometry)

    const char *  OGR_L_GetName (void *layer)
    void *  OGR_L_GetSpatialRef (void *layer)
    int     OGR_L_TestCapability (void *layer, char *name)
    void *  OGR_L_GetLayerDefn (void *layer)
    void *  OGR_L_GetNextFeature (void *layer)
    void    OGR_L_ResetReading (void *layer)
    int     OGR_L_GetFeatureCount (void *layer, int m)

cdef extern from "ogr_srs_api.h":

    ctypedef void * OGRSpatialReferenceH

    int     OSRAutoIdentifyEPSG (OGRSpatialReferenceH srs)
    const char * OSRGetAuthorityName (OGRSpatialReferenceH srs, const char *key)
    const char * OSRGetAuthorityCode (OGRSpatialReferenceH srs, const char *key)
    int     OSRExportToWkt (OGRSpatialReferenceH srs, char **params)



cdef extern from "gdal.h":
    void GDALAllRegister()
    char * GDALGetDatasetDriver (void * hDataset)
    void * GDALGetDriverByName(const char * pszName)
    void * GDALOpenEx(const char * pszFilename,
                      unsigned int nOpenFlags,
                      const char *const *papszAllowedDrivers,
                      const char *const *papszOpenOptions,
                      const char *const *papszSiblingFiles
                      )
    int GDAL_OF_UPDATE
    int GDAL_OF_READONLY
    int GDAL_OF_VECTOR
    int GDAL_OF_VERBOSE_ERROR
    int GDALDatasetGetLayerCount(void * hds)
    void * GDALDatasetGetLayer(void * hDS, int iLayer)
    void * GDALDatasetGetLayerByName(void * hDS, char * pszName)
    void GDALClose(void * hDS)
    char * GDALVersionInfo (char *pszRequest)


cdef extern from "cpl_string.h":
    char ** CSLAddNameValue (char **list, const char *name, const char *value)
    char ** CSLSetNameValue (char **list, const char *name, const char *value)
    void CSLDestroy (char **list)
    char ** CSLAddString(char **list, const char *string)

cdef extern from "cpl_conv.h":
    void *  CPLMalloc (size_t)
    void    CPLFree (void *ptr)
