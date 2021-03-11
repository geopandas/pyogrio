# With Python >= 3.8 on Windows directories in PATH are not automatically
# searched for DLL dependencies and must be added manually with
# os.add_dll_directory.
# adapted from Fiona: https://github.com/Toblerity/Fiona/pull/875

from pathlib import Path
import os
import platform
import sys


gdal_dll_dir = None


if platform.system() == "Windows" and sys.version_info >= (3, 8):
    # if loading of extension modules fails, search for gdal dll directory
    try:
        import pyogrio._io

    except ImportError as e:
        for path in os.getenv("PATH", "").split(os.pathsep):
            if Path(path).glob("gdal*.dll"):
                print(f"Found GDAL at {path}")
                gdal_dll_dir = path
                break

        if not gdal_dll_dir:
            raise ImportError(
                "GDAL DLL could not be found.  It must be on the system PATH."
            )


class GDALEnv(object):
    """Context manager for adding GDAL DLL directory on Windows for Python >= 3.8.
    Use before importing anything that depends on GDAL:

        with GDALEnv():
            import pyogrio._io
    """

    def __enter__(self):
        if gdal_dll_dir:
            self.dll_dir = os.add_dll_directory(gdal_dll_dir)
            print(f"Added GDAL DLL directory {gdal_dll_dir}")
        else:
            self.dll_dir = None

    def __exit__(self, *args):
        if self.dll_dir:
            self.dll_dir.close()
