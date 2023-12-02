from packaging.version import Version

from pyogrio.core import __gdal_version__, __gdal_geos_version__

# detect optional dependencies
try:
    import pyarrow
except ImportError:
    pyarrow = None

try:
    import shapely
except ImportError:
    shapely = None

try:
    import geopandas
except ImportError:
    geopandas = None

try:
    import pandas
except ImportError:
    pandas = None


HAS_ARROW_API = __gdal_version__ >= (3, 6, 0) and pyarrow is not None

HAS_GEOPANDAS = geopandas is not None

PANDAS_GE_15 = pandas is not None and Version(pandas.__version__) >= Version("1.5.0")
PANDAS_GE_20 = pandas is not None and Version(pandas.__version__) >= Version("2.0.0")

HAS_GDAL_GEOS = __gdal_geos_version__ is not None

HAS_SHAPELY = shapely is not None and Version(shapely.__version__) >= Version("2.0.0")
