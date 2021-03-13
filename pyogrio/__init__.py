import sys

if sys.platform == "win32":
    import ctypes
    import imp
    import os

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

from pyogrio.core import list_layers, read_info
from pyogrio.geopandas import read_dataframe, write_dataframe
from pyogrio._version import get_versions


__version__ = get_versions()["version"]
del get_versions
