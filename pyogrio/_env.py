# With Python >= 3.8 on Windows directories in PATH are not automatically
# searched for DLL dependencies and must be added manually with
# os.add_dll_directory.
# adapted from Fiona: https://github.com/Toblerity/Fiona/pull/875


from contextlib import contextmanager
import logging
import os
from pathlib import Path
import platform
import sys


log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

gdal_dll_dir = None


if platform.system() == "Windows" and sys.version_info >= (3, 8):
    # if loading of extension modules fails, search for gdal dll directory
    try:
        import pyogrio._io

    except ImportError:
        for path in os.getenv("PATH", "").split(os.pathsep):
            if list(Path(path).glob("gdal*.dll")):
                log.info(f"Found GDAL at {path}")
                gdal_dll_dir = path
                break

        if not gdal_dll_dir:
            raise ImportError(
                "GDAL DLL could not be found.  It must be on the system PATH."
            )


# class GDALEnv(object):
#     """Context manager for adding GDAL DLL directory on Windows for Python >= 3.8.
#     Use before importing anything that depends on GDAL:

#         with GDALEnv():
#             import pyogrio._io
#     """

#     def __init__(self):
#         self.dll_dir = None

#     def __enter__(self):
#         if gdal_dll_dir:
#             self.dll_dir = os.add_dll_directory(gdal_dll_dir)

#     def __exit__(self, *args):
#         print("__exit__ called")
#         if self.dll_dir is not None:
#             print("Removing GDAL DLL directory")
#             self.dll_dir.close()


@contextmanager
def GDALEnv():
    dll_dir = None

    if gdal_dll_dir:
        dll_dir = os.add_dll_directory(gdal_dll_dir)

    try:
        yield None
    finally:
        print("finally called")
        if dll_dir is not None:
            print("Removing GDAL DLL directory")
            dll_dir.close()

