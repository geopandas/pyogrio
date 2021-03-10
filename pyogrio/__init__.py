import os
import sys


print("Incoming PATH: ", os.environ['PATH'])


# Debug if DLLs are actually found
if sys.platform == "win32":
    import ctypes
    import imp
    for dll in ['_err.pyd', '_geometry.pyd', '_io.pyd', '_ogr.pyd']:
        print(f"Trying to find module {dll}")
        try:
            _, pathname = imp.find_module(dll)
        except:
            print("ERROR: could not find module {dll}")

        print(f"Trying to load {dll}")
        try:
            ctypes.WinDLL(dll)
        except:
            print(f"ERROR: unable to load DLL {dll}")






from pyogrio.core import list_layers, read_info
from pyogrio.geopandas import read_dataframe, write_dataframe

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
