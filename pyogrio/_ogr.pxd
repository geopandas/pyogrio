# from fiona::ogrext3.pxd

cdef extern from "ogr_core.h":
    ctypedef int OGRErr
    ctypedef void* OGRDataSourceH
    ctypedef void* OGRGeometryH
    ctypedef void* OGRLayerH

    # Field subtype values
    ctypedef int OGRFieldSubType
    cdef int OFSTNone = 0
    cdef int OFSTBoolean = 1
    cdef int OFSTInt16 = 2
    cdef int OFSTFloat32 = 3

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

    void *  OGR_F_Create (void *featuredefn)
    void    OGR_F_Destroy (void *feature)

    long    OGR_F_GetFID (void *feature)
    void *  OGR_F_GetGeometryRef (void *feature)
    unsigned char * OGR_F_GetFieldAsBinary(void *feature, int n, int *s)
    int     OGR_F_GetFieldAsDateTime (void *feature, int n, int *y, int *m, int *d, int *h, int *m, int *s, int *z)
    double  OGR_F_GetFieldAsDouble (void *feature, int n)
    int OGR_F_GetFieldAsInteger (void *feature, int n)
    long OGR_F_GetFieldAsInteger64 (void *feature, int n)
    char *  OGR_F_GetFieldAsString (void *feature, int n)
    int     OGR_F_IsFieldSetAndNotNull (void *feature, int n)
    void    OGR_F_SetFieldDateTime (void *feature, int n, int y, int m, int d, int hh, int mm, int ss, int tz)
    void    OGR_F_SetFieldDouble (void *feature, int n, double value)
    void    OGR_F_SetFieldInteger (void *feature, int n, int value)
    void    OGR_F_SetFieldInteger64 (void *feature, int n, long long value)
    void    OGR_F_SetFieldString (void *feature, int n, char *value)
    void    OGR_F_SetFieldBinary (void *feature, int n, int l, unsigned char *value)
    void    OGR_F_SetFieldNull (void *feature, int n)  # new in GDAL 2.2
    OGRErr  OGR_F_SetGeometryDirectly (void *feature, void *geometry)

    void *  OGR_FD_Create (char *name)
    int     OGR_FD_GetFieldCount (void *featuredefn)
    void *  OGR_FD_GetFieldDefn (void *featuredefn, int n)
    OGRwkbGeometryType     OGR_FD_GetGeomType (void *featuredefn)

    void *  OGR_Fld_Create (char *name, OGRFieldType fieldtype)
    void    OGR_Fld_Destroy (void *fielddefn)
    char *  OGR_Fld_GetNameRef (void *fielddefn)
    int     OGR_Fld_GetPrecision (void *fielddefn)
    OGRFieldSubType OGR_Fld_GetSubType(void *fielddefn)
    int     OGR_Fld_GetType (void *fielddefn)
    int     OGR_Fld_GetWidth (void *fielddefn)
    void    OGR_Fld_Set (void *fielddefn, char *name, int fieldtype, int width, int precision, int justification)
    void    OGR_Fld_SetPrecision (void *fielddefn, int n)
    void    OGR_Fld_SetWidth (void *fielddefn, int n)

    void    OGR_Fld_SetSubType(void *fielddefn, OGRFieldSubType subtype)

    void *  OGR_G_CreateGeometry (int wkbtypecode)
    void    OGR_G_DestroyGeometry (void *geometry)
    void    OGR_G_ExportToWkb (void *geometry, int endianness, unsigned char *buffer)
    OGRErr  OGR_G_ImportFromWkb (void *geometry, unsigned char *bytes, int nbytes)
    int     OGR_G_WkbSize (void *geometry)
    int     OGR_G_IsMeasured(void *geometry)
    void    OGR_G_SetMeasured(void *geometry, int isMeasured)
    int     OGR_G_Is3D(void *geometry)
    void    OGR_G_Set3D(void *geoemtry, int is3D)

    int     OGR_GT_HasM(OGRwkbGeometryType eType)
    int     OGR_GT_HasZ(OGRwkbGeometryType eType)
    OGRwkbGeometryType  OGR_GT_SetModifier(OGRwkbGeometryType eType, int setZ, int setM)

    OGRErr  OGR_L_CreateFeature (void *layer, void *feature)
    OGRErr  OGR_L_CreateField (void *layer, void *fielddefn, int flexible)
    const char *  OGR_L_GetName (void *layer)
    void *  OGR_L_GetSpatialRef (void *layer)
    int     OGR_L_TestCapability (void *layer, char *name)
    void *  OGR_L_GetLayerDefn (void *layer)
    void *  OGR_L_GetNextFeature (void *layer)
    void    OGR_L_ResetReading (void *layer)
    OGRErr  OGR_L_SetAttributeFilter(OGRLayerH hLayer, const char* pszQuery)
    OGRErr  OGR_L_SetNextByIndex(void *layer, int nIndex)
    int     OGR_L_GetFeatureCount (void *layer, int m)

    void    OGRSetNonLinearGeometriesEnabledFlag (int bFlag)
    int     OGRGetNonLinearGeometriesEnabledFlag ()

    int     OGRReleaseDataSource (void *datasource)

cdef extern from "ogr_srs_api.h":

    ctypedef void * OGRSpatialReferenceH

    int     OSRAutoIdentifyEPSG (OGRSpatialReferenceH srs)
    const char * OSRGetAuthorityName (OGRSpatialReferenceH srs, const char *key)
    const char * OSRGetAuthorityCode (OGRSpatialReferenceH srs, const char *key)
    OGRErr  OSRExportToWkt (OGRSpatialReferenceH srs, char **params)

    int     OSRSetFromUserInput(OGRSpatialReferenceH srs, const char *pszDef)
    OGRSpatialReferenceH  OSRNewSpatialReference (char *wkt)
    void    OSRRelease (OGRSpatialReferenceH srs)



cdef extern from "gdal.h":
    ctypedef enum GDALDataType:
        GDT_Unknown
        GDT_Byte
        GDT_UInt16
        GDT_Int16
        GDT_UInt32
        GDT_Int32
        GDT_Float32
        GDT_Float64
        GDT_CInt16
        GDT_CInt32
        GDT_CFloat32
        GDT_CFloat64
        GDT_TypeCount

    int GDAL_OF_UPDATE
    int GDAL_OF_READONLY
    int GDAL_OF_VECTOR
    int GDAL_OF_VERBOSE_ERROR

    void GDALAllRegister()

    void * GDALCreate(void * hDriver,
                      const char * pszFilename,
                      int nXSize,
                      int     nYSize,
                      int     nBands,
                      GDALDataType eBandType,
                      char ** papszOptions)
    void * GDALDatasetCreateLayer(void * hDS,
                                  const char * pszName,
                                  void * hSpatialRef,
                                  int eType,
                                  char ** papszOptions)
    int GDALDatasetDeleteLayer(void * hDS, int iLayer)

    char * GDALGetDatasetDriver (void * hDataset)
    void * GDALGetDriverByName(const char * pszName)
    void * GDALOpenEx(const char * pszFilename,
                      unsigned int nOpenFlags,
                      const char *const *papszAllowedDrivers,
                      const char *const *papszOpenOptions,
                      const char *const *papszSiblingFiles
                      )

    void GDALClose(void * hDS)
    int GDALDatasetGetLayerCount(void * hds)
    void * GDALDatasetGetLayer(void * hDS, int iLayer)
    void * GDALDatasetGetLayerByName(void * hDS, char * pszName)
    OGRErr GDALDatasetStartTransaction (void * hDataset, int bForce)
    OGRErr GDALDatasetCommitTransaction (void * hDataset)
    OGRErr GDALDatasetRollbackTransaction (void * hDataset)
    char * GDALVersionInfo (char *pszRequest)


cdef extern from "cpl_string.h":
    char ** CSLAddNameValue (char **list, const char *name, const char *value)
    char ** CSLSetNameValue (char **list, const char *name, const char *value)
    void CSLDestroy (char **list)
    char ** CSLAddString(char **list, const char *string)

cdef extern from "cpl_conv.h":
    void *  CPLMalloc (size_t)
    void    CPLFree (void *ptr)
