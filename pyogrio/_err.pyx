"""Error handling code for GDAL/OGR.

Ported from fiona::_err.pyx
"""

import contextlib
import logging
import warnings
from contextvars import ContextVar
from enum import IntEnum
from itertools import zip_longest

from pyogrio._ogr cimport (
    CE_None, CE_Debug, CE_Warning, CE_Failure, CE_Fatal, CPLErrorReset,
    CPLGetLastErrorType, CPLGetLastErrorNo, CPLGetLastErrorMsg, OGRErr,
    CPLErr, CPLErrorHandler, CPLDefaultErrorHandler, CPLPopErrorHandler,
    CPLPushErrorHandler, CPLPushErrorHandlerEx)

log = logging.getLogger(__name__)

_ERROR_STACK = ContextVar("error_stack")
_ERROR_STACK.set([])


# CPL Error types as an enum.
class GDALError(IntEnum):
    """GDAL error types/classes.
    
    GDAL doc: https://gdal.org/en/latest/doxygen/cpl__error_8h.html#a463ba7c7202a505416ff95b1aeefa2de
    """
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


class CPLE_AWSError(CPLE_BaseError):
    pass


class NullPointerError(CPLE_BaseError):
    """
    Returned from exc_wrap_pointer when a NULL pointer is passed, but no GDAL
    error was raised.
    """
    pass


class CPLError(CPLE_BaseError):
    """
    Returned from exc_wrap_int when a error code is returned, but no GDAL
    error was set.
    """
    pass


cdef dict _LEVEL_MAP = {
    0: 0,
    1: logging.DEBUG,
    2: logging.WARNING,
    3: logging.ERROR,
    4: logging.CRITICAL
}

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
    16: CPLE_AWSSignatureDoesNotMatchError,
    17: CPLE_AWSError
}

cdef dict _CODE_MAP = {
    0: 'CPLE_None',
    1: 'CPLE_AppDefined',
    2: 'CPLE_OutOfMemory',
    3: 'CPLE_FileIO',
    4: 'CPLE_OpenFailed',
    5: 'CPLE_IllegalArg',
    6: 'CPLE_NotSupported',
    7: 'CPLE_AssertionFailed',
    8: 'CPLE_NoWriteAccess',
    9: 'CPLE_UserInterrupt',
    10: 'ObjectNull',
    11: 'CPLE_HttpResponse',
    12: 'CPLE_AWSBucketNotFound',
    13: 'CPLE_AWSObjectNotFound',
    14: 'CPLE_AWSAccessDenied',
    15: 'CPLE_AWSInvalidCredentials',
    16: 'CPLE_AWSSignatureDoesNotMatch',
    17: 'CPLE_AWSError'
}


cdef class GDALErrCtxManager:
    """A manager for GDAL error handling contexts."""

    def __enter__(self):
        CPLErrorReset()
        return self

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        cdef int err_type = CPLGetLastErrorType()
        cdef int err_no = CPLGetLastErrorNo()
        cdef const char *msg = CPLGetLastErrorMsg()
        # TODO: warn for err_type 2?
        if err_type >= 2:
            raise exception_map[err_no](err_type, err_no, msg)


cdef inline object exc_check():
    """Checks GDAL error stack for fatal or non-fatal errors.

    Returns
    -------
    An Exception, SystemExit, or None
    """
    err_type = CPLGetLastErrorType()
    err_no = CPLGetLastErrorNo()
    err_msg = get_last_error_msg()

    if err_type == CE_Failure:
        CPLErrorReset()
        return exception_map.get(
            err_no, CPLE_BaseError)(err_type, err_no, err_msg)

    if err_type == CE_Fatal:
        return SystemExit("Fatal error: {0}".format((err_type, err_no, err_msg)))

    else:
        return


cdef get_last_error_msg():
    """Checks GDAL error stack for the latest error message.

    Returns
    -------
    An error message or empty string

    """
    return clean_error_message(CPLGetLastErrorMsg())


cdef clean_error_message(const char* err_msg):
    """Cleans up error messages from GDAL.

    Parameters
    ----------
    err_msg : const char*
        The error message to clean up.

    Returns
    -------
    str
        The cleaned up error message or empty string
    """
    if err_msg != NULL:
        # Reformat message.
        msg_b = err_msg
        try:
            msg = msg_b.decode("utf-8")
            msg = msg.replace("`", "'")
            msg = msg.replace("\n", " ")
        except UnicodeDecodeError as exc:
            msg = f"Could not decode error message to UTF-8. Raw error: {msg_b}"

    else:
        msg = ""

    return msg


cdef void *exc_wrap_pointer(void *ptr) except NULL:
    """Wrap a GDAL/OGR function that returns GDALDatasetH etc (void *).

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
    """Wrap a GDAL/OGR function that returns CPLErr or OGRErr (int).

    Raises an exception if a non-fatal error has be set.
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


cdef void error_handler(CPLErr err_class, int err_no, const char* err_msg) noexcept nogil:
    """Custom CPL error handler to match the Python behaviour.

    For non-fatal errors (CE_Failure), error printing to stderr (behaviour of
    the default GDAL error handler) is suppressed, because we already raise a
    Python exception that includes the error message.

    Warnings are converted to Python warnings.
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
            warnings.warn(clean_error_message(err_msg), RuntimeWarning)
        return

    # Fall back to the default handler for non-failure messages since
    # they won't be translated into exceptions.
    CPLDefaultErrorHandler(err_class, err_no, err_msg)


def _register_error_handler():
    CPLPushErrorHandler(<CPLErrorHandler>error_handler)

cpl_errs = GDALErrCtxManager()


cdef class StackChecker:

    def __init__(self, error_stack=None):
        self.error_stack = error_stack or {}

    cdef int exc_wrap_int(self, int err) except -1:
        """Wrap a GDAL/OGR function that returns CPLErr (int).

        Raises an exception if a non-fatal error has been added to the
        exception stack.
        """
        if err:
            stack = self.error_stack.get()
            for error, cause in zip_longest(stack[::-1], stack[::-1][1:]):
                if error is not None and cause is not None:
                    error.__cause__ = cause

            if stack:
                last = stack.pop()
                if last is not None:
                    raise last

        return err

    cdef void *exc_wrap_pointer(self, void *ptr) except NULL:
        """Wrap a GDAL/OGR function that returns a pointer.

        Raises an exception if a non-fatal error has been added to the
        exception stack.
        """
        if ptr == NULL:
            stack = self.error_stack.get()
            for error, cause in zip_longest(stack[::-1], stack[::-1][1:]):
                if error is not None and cause is not None:
                    error.__cause__ = cause

            if stack:
                last = stack.pop()
                if last is not None:
                    raise last

        return ptr


cdef void log_error(
    CPLErr err_class,
    int err_no,
    const char* msg,
) noexcept with gil:
    """Send CPL errors to Python's logger.

    Because this function is called by GDAL with no Python context, we
    can't propagate exceptions that we might raise here. They'll be
    ignored.
    """
    if err_no in _CODE_MAP:
        # We've observed that some GDAL functions may emit multiple
        # ERROR level messages and yet succeed. We want to see those
        # messages in our log file, but not at the ERROR level. We
        # turn the level down to INFO.
        if err_class == CE_Failure:
            log.info(
                "GDAL signalled an error: err_no=%r, msg=%r",
                err_no,
                msg.decode("utf-8")
            )
        elif err_no == 0:
            log.log(_LEVEL_MAP[err_class], "%s", msg.decode("utf-8"))
        else:
            log.log(_LEVEL_MAP[err_class], "%s:%s", _CODE_MAP[err_no], msg.decode("utf-8"))
    else:
        log.info("Unknown error number %r", err_no)


IF UNAME_SYSNAME == "Windows":
    cdef void __stdcall stacking_error_handler(
        CPLErr err_class,
        int err_no,
        const char* err_msg
    ) noexcept with gil:
        """Custom CPL error handler that adds non-fatal errors to a stack.

        All non-fatal errors (CE_Failure) are not printed to stderr (behaviour
        of the default GDAL error handler), but they are converted to python
        exceptions and added to a stack, so they can be dealt with afterwards.

        Warnings are converted to Python warnings.
        """
        global _ERROR_STACK
        log_error(err_class, err_no, err_msg)

        if err_class == CE_Fatal:
            # If the error class is CE_Fatal, we want to have a message issued
            # because the CPL support code does an abort() before any exception
            # can be generated
            CPLDefaultErrorHandler(err_class, err_no, err_msg)

            return

        elif err_class == CE_Failure:
            # For Failures, add them to the error exception stack
            stack = _ERROR_STACK.get()
            stack.append(
                exception_map.get(err_no, CPLE_BaseError)(
                    err_class, err_no, clean_error_message(err_msg)
                ),
            )
            _ERROR_STACK.set(stack)

            return

        elif err_class == CE_Warning:
            warnings.warn(clean_error_message(err_msg), RuntimeWarning)

            return

        # Fall back to the default handler for non-failure messages since
        # they won't be translated into exceptions.
        CPLDefaultErrorHandler(err_class, err_no, err_msg)

ELSE:
    cdef void stacking_error_handler(
        CPLErr err_class,
        int err_no,
        const char* err_msg
    ) noexcept with gil:
        """Custom CPL error handler that adds non-fatal errors to a stack.

        All non-fatal errors (CE_Failure) are not printed to stderr (behaviour
        of the default GDAL error handler), but they are converted to python
        exceptions and added to a stack, so they can be dealt with afterwards.

        Warnings are converted to Python warnings.
        """
        global _ERROR_STACK
        log_error(err_class, err_no, err_msg)

        if err_class == CE_Fatal:
            # If the error class is CE_Fatal, we want to have a message issued
            # because the CPL support code does an abort() before any exception
            # can be generated
            CPLDefaultErrorHandler(err_class, err_no, err_msg)

            return

        elif err_class == CE_Failure:
            # For Failures, add them to the error exception stack
            stack = _ERROR_STACK.get()
            stack.append(
                exception_map.get(err_no, CPLE_BaseError)(
                    err_class, err_no, clean_error_message(err_msg)
                ),
            )
            _ERROR_STACK.set(stack)

            return

        elif err_class == CE_Warning:
            warnings.warn(clean_error_message(err_msg), RuntimeWarning)

            return

        # Fall back to the default handler for non-failure messages since
        # they won't be translated into exceptions.
        CPLDefaultErrorHandler(err_class, err_no, err_msg)


@contextlib.contextmanager
def stack_errors():
    """A context manager that captures all GDAL non-fatal errors occuring.

    It adds all errors to a single stack, so it assumes that no more than one
    GDAL function is called.

    Yields a StackChecker object that can be used to check the error stack.
    """
    CPLErrorReset()
    global _ERROR_STACK
    _ERROR_STACK.set([])

    # stacking_error_handler records GDAL errors in the order they occur and
    # converts them to exceptions.
    CPLPushErrorHandlerEx(<CPLErrorHandler>stacking_error_handler, NULL)

    # Run code in the `with` block.
    yield StackChecker(_ERROR_STACK)

    CPLPopErrorHandler()
    _ERROR_STACK.set([])
    CPLErrorReset()
