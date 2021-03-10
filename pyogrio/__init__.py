import os
import sys


# Debug if DLLs are actually found
if sys.platform == "win32":
    print("Incoming PATH: ", os.environ['PATH'])
    import ctypes
    import imp
    for module in ['_err', '_geometry', '_io', '_ogr']:
        print(f"Trying to find module pyogrio.{module}")
        try:
            _, pathname = imp.find_module(f"pyogrio.{module}")
        except:
            print(f"ERROR: could not find module pyogrio.{module}")
            print("Error was: ", sys.exc_info()[0])

        dll = f"{module}.cp39-win_amd64.pyd"
        print(f"Trying to load {dll}")
        try:
            ctypes.WinDLL(dll)
        except:
            print(f"ERROR: unable to load DLL {dll}")
            print("Error was: ", sys.exc_info()[0])

    try:
        ctypes.WinDLL('gdal302.dll')
    except:
        print("ERROR: could not load GDAL DLL")
        print("Error was: ", sys.exc_info()[0])

    # Try again after setting dll directory
    with os.add_dll_directory("c:/gdal/bin"):
        try:
            ctypes.WinDLL('gdal302.dll')
            print("Successfully loaded GDAL, now trying to load core")
            try:
                from pyogrio.core import list_layers, read_info
            except:
                print("ERROR: couldn't load core after loading GDAL")

        except:
            print("ERROR: could not load GDAL DLL even after setting DLL directory")
            print("Error was: ", sys.exc_info()[0])



from pyogrio.core import list_layers, read_info
from pyogrio.geopandas import read_dataframe, write_dataframe

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
