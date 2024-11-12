cdef object exc_check()
cdef int exc_wrap_int(int retval) except -1
cdef int exc_wrap_ogrerr(int retval) except -1
cdef void *exc_wrap_pointer(void *ptr) except NULL

cdef class ErrorHandler:
    cdef object error_stack
    cdef int exc_wrap_int(self, int retval) except -1
    cdef void *exc_wrap_pointer(self, void *ptr) except NULL
