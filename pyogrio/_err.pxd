# ported from fiona::_err.pxd

cdef extern from "cpl_error.h":
    ctypedef enum CPLErr:
        CE_None
        CE_Debug
        CE_Warning
        CE_Failure
        CE_Fatal

    int CPLGetLastErrorNo()
    const char* CPLGetLastErrorMsg()
    int CPLGetLastErrorType()
    void CPLErrorReset()


cdef extern from "ogr_core.h":
    ctypedef enum OGRErr:
        OGRERR_NONE  # success
        OGRERR_NOT_ENOUGH_DATA
        OGRERR_NOT_ENOUGH_MEMORY
        OGRERR_UNSUPPORTED_GEOMETRY_TYPE
        OGRERR_UNSUPPORTED_OPERATION
        OGRERR_CORRUPT_DATA
        OGRERR_FAILURE
        OGRERR_UNSUPPORTED_SRS
        OGRERR_INVALID_HANDLE
        OGRERR_NON_EXISTING_FEATURE


cdef object exc_check()
cdef int exc_wrap_int(int retval) except -1
# cdef OGRErr exc_wrap_ogrerr(OGRErr retval) except -1
cdef void *exc_wrap_pointer(void *ptr) except NULL

