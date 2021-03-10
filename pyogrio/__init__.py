import os
import sys


print("Incoming PATH: ", os.environ['PATH'])


# Debug if DLLs are actually found
if sys.platform == "win32":
    import ctypes
    import imp
    for module in ['_err', '_geometry', '_io', '_ogr']:
        print(f"Trying to find module pyogrio.{module}")
        try:
            _, pathname = imp.find_module(f"pyogrio.{module}")
        except:
            print(f"ERROR: could not find module pyogrio.{module}")

        dll = f"{module}.cp39-win_amd64.pyd"
        print(f"Trying to load {dll}")
        try:
            ctypes.WinDLL(dll)
        except:
            print(f"ERROR: unable to load DLL {dll}")

    try:
        ctypes.WinDLL('gdal302.dll')
    except:
        print("ERROR: could not load GDAL DLL")




from pyogrio.core import list_layers, read_info
from pyogrio.geopandas import read_dataframe, write_dataframe

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
