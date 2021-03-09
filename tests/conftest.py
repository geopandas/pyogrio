import os
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED

if sys.platform == "win32":
    # GDAL DLLs are copied to ".libs" folder; make sure this is on the PATH
    libdir = str(Path(__file__) / ".libs")
    os.environ["PATH"] = os.environ["PATH"] + ";" + libdir

print("Path: ", os.environ['PATH'])

import fiona
from geopandas.datasets import get_path
import pytest


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