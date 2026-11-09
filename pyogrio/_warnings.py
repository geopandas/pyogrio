"""Attribute GDAL warnings to callers outside pyogrio."""

import os
import sys
import warnings

_PACKAGE_PREFIXES = (os.path.dirname(__file__) + os.sep,)
_PY_GE_312 = sys.version_info >= (3, 12)


def warn(message):
    """Issue a RuntimeWarning at the first external Python frame."""
    if _PY_GE_312:
        warnings.warn(message, RuntimeWarning, skip_file_prefixes=_PACKAGE_PREFIXES)
    else:
        stacklevel = 2
        frame = sys._getframe(1)
        try:
            while frame is not None and frame.f_code.co_filename.startswith(
                _PACKAGE_PREFIXES
            ):
                stacklevel += 1
                frame = frame.f_back
        finally:
            del frame
        warnings.warn(message, RuntimeWarning, stacklevel=stacklevel)
