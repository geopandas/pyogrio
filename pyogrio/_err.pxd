cdef object exc_check()
cdef int exc_wrap_int(int retval) except -1
cdef int exc_wrap_ogrerr(int retval) except -1
cdef void *exc_wrap_pointer(void *ptr) except NULL
