
from pyogrio._ogr cimport *
from pyogrio._err cimport *
from pyogrio._err import CPLE_BaseError, NullPointerError
from pyogrio.errors import DriverError


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


def ogr_read(path, **kwargs):
    cdef const char *path_c = NULL
    cdef void *cogr_ds
    # cdef void *cogr_layer

    path_b = path.encode('utf-8')
    path_c = path_b

    # all DRIVERS support read
    cogr_ds = ogr_open(path_c, 0, DRIVERS, kwargs)