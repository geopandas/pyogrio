import sys

if sys.platform == "win32":
    import ctypes
    import imp
    import os

    print("Incoming environment")
    print(os.environ)

    print("Trying to load GDAL dll directly")
    try:
        ctypes.WinDLL("gdal302.dll")
        print("successfully loaded GDAL DLL")
    except Exception as e:
        print("ERROR: unable to load GDAL DLL")
        print(e)

    print("Trying to load GDAL dll directly after setting GDAL path")
    os.add_dll_directory("c:/gdal/bin")
    try:
        ctypes.WinDLL("gdal302.dll")
        print("Successfully loaded GDAL DLL on second try")
    except Exception as e:
        print("ERROR: still unable to load GDAL DLL")
        print(e)

        # Try to load GDAL using system calls
        kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
        res = kernel32.LoadLibraryExW("c:\\gdal\\bin\\gdal302.dll", 0, 0x00001100)
        last_error = ctypes.get_last_error()
        err = ctypes.WinError(last_error)
        print("Error from loading GDAL")
        print(err)

        res = kernel32.LoadLibraryW("c:\\gdal\\bin\\gdal302.dll")
        last_error = ctypes.get_last_error()
        err = ctypes.WinError(last_error)
        print("Error from loading GDAL(2)")
        print(err)


from pyogrio.core import list_layers, read_info
from pyogrio.geopandas import read_dataframe, write_dataframe
from pyogrio._version import get_versions


__version__ = get_versions()["version"]
del get_versions
