# ported from fiona::_err.pyx
from enum import IntEnum
import warnings

from pyogrio._ogr cimport (
    CE_None, CE_Debug, CE_Warning, CE_Failure, CE_Fatal, CPLErrorReset,
    CPLGetLastErrorType, CPLGetLastErrorNo, CPLGetLastErrorMsg, OGRErr,
    CPLErr, CPLErrorHandler, CPLDefaultErrorHandler, CPLPushErrorHandler)


# CPL Error types as an enum.
class GDALError(IntEnum):
    none = CE_None
    debug = CE_Debug
    warning = CE_Warning
    failure = CE_Failure
    fatal = CE_Fatal



class CPLE_BaseError(Exception):
    """Base CPL error class.
    For internal use within Cython only.
    """

    def __init__(self, error, errno, errmsg):
        self.error = error
        self.errno = errno
        self.errmsg = errmsg

    def __str__(self):
        return self.__unicode__()

    def __unicode__(self):
        return u"{}".format(self.errmsg)

    @property
    def args(self):
        return self.error, self.errno, self.errmsg


class CPLE_AppDefinedError(CPLE_BaseError):
    pass


class CPLE_OutOfMemoryError(CPLE_BaseError):
    pass


class CPLE_FileIOError(CPLE_BaseError):
    pass


class CPLE_OpenFailedError(CPLE_BaseError):
    pass


class CPLE_IllegalArgError(CPLE_BaseError):
    pass


class CPLE_NotSupportedError(CPLE_BaseError):
    pass


class CPLE_AssertionFailedError(CPLE_BaseError):
    pass


class CPLE_NoWriteAccessError(CPLE_BaseError):
    pass


class CPLE_UserInterruptError(CPLE_BaseError):
    pass


class ObjectNullError(CPLE_BaseError):
    pass


class CPLE_HttpResponseError(CPLE_BaseError):
    pass


class CPLE_AWSBucketNotFoundError(CPLE_BaseError):
    pass


class CPLE_AWSObjectNotFoundError(CPLE_BaseError):
    pass


class CPLE_AWSAccessDeniedError(CPLE_BaseError):
    pass


class CPLE_AWSInvalidCredentialsError(CPLE_BaseError):
    pass


class CPLE_AWSSignatureDoesNotMatchError(CPLE_BaseError):
    pass


class NullPointerError(CPLE_BaseError):
    """
    Returned from exc_wrap_pointer when a NULL pointer is passed, but no GDAL
    error was raised.
    """
    pass



# Map of GDAL error numbers to the Python exceptions.
exception_map = {
    1: CPLE_AppDefinedError,
    2: CPLE_OutOfMemoryError,
    3: CPLE_FileIOError,
    4: CPLE_OpenFailedError,
    5: CPLE_IllegalArgError,
    6: CPLE_NotSupportedError,
    7: CPLE_AssertionFailedError,
    8: CPLE_NoWriteAccessError,
    9: CPLE_UserInterruptError,
    10: ObjectNullError,

    # error numbers 11-16 are introduced in GDAL 2.1. See
    # https://github.com/OSGeo/gdal/pull/98.
    11: CPLE_HttpResponseError,
    12: CPLE_AWSBucketNotFoundError,
    13: CPLE_AWSObjectNotFoundError,
    14: CPLE_AWSAccessDeniedError,
    15: CPLE_AWSInvalidCredentialsError,
    16: CPLE_AWSSignatureDoesNotMatchError
}


cdef inline object exc_check():
    """Checks GDAL error stack for fatal or non-fatal errors
    Returns
    -------
    An Exception, SystemExit, or None
    """
    cdef const char *msg_c = NULL

    err_type = CPLGetLastErrorType()
    err_no = CPLGetLastErrorNo()
    err_msg = CPLGetLastErrorMsg()

    if err_msg == NULL:
        msg = "No error message."
    else:
        # Reformat messages.
        msg_b = err_msg

        try:
            msg = msg_b.decode('utf-8')
            msg = msg.replace("`", "'")
            msg = msg.replace("\n", " ")
        except UnicodeDecodeError as exc:
            msg = f"Could not decode error message to UTF-8.  Raw error: {msg_b}"

    if err_type == 3:
        CPLErrorReset()
        return exception_map.get(
            err_no, CPLE_BaseError)(err_type, err_no, msg)

    if err_type == 4:
        return SystemExit("Fatal error: {0}".format((err_type, err_no, msg)))

    else:
        return


cdef void *exc_wrap_pointer(void *ptr) except NULL:
    """Wrap a GDAL/OGR function that returns GDALDatasetH etc (void *)
    Raises an exception if a non-fatal error has be set or if pointer is NULL.
    """
    if ptr == NULL:
        exc = exc_check()
        if exc:
            raise exc
        else:
            # null pointer was passed, but no error message from GDAL
            raise NullPointerError(-1, -1, "NULL pointer error")
    return ptr


cdef int exc_wrap_int(int err) except -1:
    """Wrap a GDAL/OGR function that returns CPLErr or OGRErr (int)
    Raises an exception if a non-fatal error has be set.

    Copied from Fiona (_err.pyx).
    """
    if err:
        exc = exc_check()
        if exc:
            raise exc
        else:
            # no error message from GDAL
            raise CPLE_BaseError(-1, -1, "Unspecified OGR / GDAL error")
    return err


cdef int exc_wrap_ogrerr(int err) except -1:
    """Wrap a function that returns OGRErr (int) but does not use the
    CPL error stack.

    Adapted from Fiona (_err.pyx).
    """
    if err != 0:
        raise CPLE_BaseError(3, err, f"OGR Error code {err}")

    return err


cdef void error_handler(CPLErr err_class, int err_no, const char* err_msg) nogil:
    """Custom CPL error handler to match the Python behaviour.

    Generally we want to suppress error printing to stderr (behaviour of the
    default GDAL error handler) because we already raise a Python exception
    that includes the error message.
    """
    if err_class == CE_Fatal:
        # If the error class is CE_Fatal, we want to have a message issued
        # because the CPL support code does an abort() before any exception
        # can be generated
        CPLDefaultErrorHandler(err_class, err_no, err_msg)
        return

    elif err_class == CE_Failure:
        # For Failures, do nothing as those are explicitly caught
        # with error return codes and translated into Python exceptions
        return

    elif err_class == CE_Warning:
        with gil:
            msg_b = err_msg
            msg = msg_b.decode('utf-8')
            warnings.warn(msg, RuntimeWarning)
        return

    # Fall back to the default handler for non-failure messages since
    # they won't be translated into exceptions.
    CPLDefaultErrorHandler(err_class, err_no, err_msg)


def _register_error_handler():
    CPLPushErrorHandler(<CPLErrorHandler>error_handler)
