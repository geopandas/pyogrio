from pathlib import Path
import os
import sys

if sys.platform == "win32":
    # GDAL DLLs are copied to ".libs" folder; make sure this is on the PATH
    libdir = str(Path(__file__) / ".libs")
    os.environ["PATH"] = os.environ["PATH"] + ";" + libdir

from pyogrio.core import list_layers, read_info
from pyogrio.geopandas import read_dataframe, write_dataframe

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
