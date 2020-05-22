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
    ctypedef int OGRErr

cdef int exc_wrap_int(int retval) except -1
# cdef OGRErr exc_wrap_ogrerr(OGRErr retval) except -1
cdef void *exc_wrap_pointer(void *ptr) except NULL

