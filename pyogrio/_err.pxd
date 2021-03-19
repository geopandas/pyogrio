# ported from fiona::_err.pxd




cdef object exc_check()
cdef int exc_wrap_int(int retval) except -1
# cdef OGRErr exc_wrap_ogrerr(OGRErr retval) except -1
cdef void *exc_wrap_pointer(void *ptr) except NULL

