from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import pytest

from pyogrio import __gdal_version_string__, __version__, list_drivers
import pyogrio

_data_dir = Path(__file__).parent.resolve() / "fixtures"


def pytest_report_header(config):
    drivers = ", ".join(sorted(list(list_drivers(read=True).keys())))
    return (
        f"pyogrio {__version__}\n"
        f"GDAL {__gdal_version_string__}\n"
        f"Supported drivers: {drivers}"
    )


@pytest.fixture(scope="session")
def data_dir():
    return _data_dir


@pytest.fixture(scope="function")
def naturalearth_lowres():
    return _data_dir / Path("naturalearth_lowres/naturalearth_lowres.shp")


@pytest.fixture(scope="function")
def naturalearth_lowres_ext(tmp_path, request):
    ext = None
    if "callspec" in dir(request._pyfuncitem):
        ext = request._pyfuncitem.callspec.params.get("ext")
    if ext is None:
        ext = ".shp" 

    shp_path = _data_dir / Path("naturalearth_lowres/naturalearth_lowres.shp")
    if ext.lower() == ".shp":
        return shp_path
    else:
        ext_path = tmp_path / f"{shp_path.stem}{ext}"
        if ext_path.exists():
            return ext_path
        gdf = pyogrio.read_dataframe(shp_path)
        pyogrio.write_dataframe(gdf, ext_path)
        return ext_path

@pytest.fixture(scope="function")
def naturalearth_lowres_vsi(tmp_path, naturalearth_lowres):
    """Wrap naturalearth_lowres as a zip file for vsi tests"""

    path = tmp_path / f"{naturalearth_lowres.name}.zip"
    with ZipFile(path, mode="w", compression=ZIP_DEFLATED, compresslevel=5) as out:
        for ext in ["dbf", "prj", "shp", "shx"]:
            filename = f"{naturalearth_lowres.stem}.{ext}"
            out.write(naturalearth_lowres.parent / filename, filename)

    return path, f"/vsizip/{path}/{naturalearth_lowres.name}"


@pytest.fixture(scope="session")
def test_fgdb_vsi():
    return f"/vsizip/{_data_dir}/test_fgdb.gdb.zip"

