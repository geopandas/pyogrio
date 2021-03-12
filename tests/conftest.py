from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import pytest

import pygeos
print("pygeos GEOS version", pygeos.geos_version_string)
import shapely.geos
print("Shapely GEOS version", shapely.geos.geos_version_string)


print("Loading pyogrio in conftest")
import pyogrio
print("Loaded pyogrio")

from pyogrio._env import GDALEnv

with GDALEnv():
    print("Loading Fiona")
    # Fiona is required by geopandas, need to load it after setting DLL search path
    import fiona
    print("Fiona GDAL version", fiona.__gdal_version__)



from geopandas.datasets import get_path

print("Successfully loaded geopandas.datasets.get_path")

data_dir = Path(__file__).parent.resolve() / "fixtures"


@pytest.fixture(scope="session")
def naturalearth_lowres():
    return get_path("naturalearth_lowres")


@pytest.fixture
def naturalearth_lowres_vsi(tmp_path):
    """Wrap naturalearth_lowres as a zip file for vsi tests"""

    naturalearth_lowres = Path(get_path("naturalearth_lowres"))

    path = tmp_path / f"{naturalearth_lowres.name}.zip"
    with ZipFile(path, mode="w", compression=ZIP_DEFLATED, compresslevel=5) as out:
        # out.write(naturalearth_lowres, naturalearth_lowres.name)
        for ext in ["dbf", "prj", "shp", "shx"]:
            filename = f"{naturalearth_lowres.stem}.{ext}"
            out.write(naturalearth_lowres.parent / filename, filename)

    return f"/vsizip/{path}/{naturalearth_lowres.name}"


@pytest.fixture(scope="session")
def naturalearth_cities():
    return get_path("naturalearth_cities")


@pytest.fixture(scope="session")
def nybb_vsi():
    return get_path("nybb").replace("zip:/", "/vsizip/")


@pytest.fixture(scope="session")
def test_fgdb_vsi():
    return f'/vsizip/{data_dir}/test_fgdb.gdb.zip'